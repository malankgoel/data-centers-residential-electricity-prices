#!/usr/bin/env python3
"""Build the service-territory panel and run fixed-effect regressions."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from utility_config import (
    ANALYSIS_STATES,
    MIN_RESIDENTIAL_CUSTOMERS,
    PANEL_END,
    PANEL_START,
    RAW,
    ROOT,
    STATE_NAME_TO_ABBR,
)


OUT = ROOT / "data" / "processed" / "utility_service_territory"
DC_PATH = ROOT / "data" / "processed" / "data_centers" / "data_centers_with_year_imputed.csv"


def normalize_utility_name(name: str) -> str:
    text = str(name).upper().strip()
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_eia_sales_year(year: int) -> pd.DataFrame:
    folder = RAW / "f-861" / f"f861{year}"
    if not folder.exists():
        return pd.DataFrame()
    candidates = list(folder.glob(f"**/Sales_Ult_Cust_{year}.xlsx"))
    candidates += list(folder.glob(f"**/Sales_Ult_Cust_{year}.xls"))
    candidates += list(folder.glob(f"Sales_Ult_Cust*{year}*.xlsx"))
    candidates += list(folder.glob(f"Sales_Ult_Cust*{year}*.xls"))
    candidates = [p for p in candidates if "CS" not in p.name]
    if not candidates:
        return pd.DataFrame()
    path = sorted(candidates, key=lambda p: len(str(p)))[0]
    df = pd.read_excel(path, sheet_name="States", header=2)
    rename = {
        "Data Year": "year",
        "Utility Number": "utility_number",
        "Utility Name": "utility_name",
        "State": "state",
        "Ownership": "ownership",
        "Thousand Dollars": "residential_revenue_thousand_dollars",
        "Megawatthours": "residential_sales_mwh",
        "Count": "residential_customers",
        "Thousand Dollars.1": "commercial_revenue_thousand_dollars",
        "Megawatthours.1": "commercial_sales_mwh",
        "Count.1": "commercial_customers",
        "Thousand Dollars.2": "industrial_revenue_thousand_dollars",
        "Megawatthours.2": "industrial_sales_mwh",
        "Count.2": "industrial_customers",
    }
    missing = [k for k in rename if k not in df.columns]
    if missing:
        return pd.DataFrame()
    if "Part" in df.columns and "Service Type" in df.columns:
        df = df[df["Part"].eq("A") & df["Service Type"].eq("Bundled")]
    out = df[list(rename)].rename(columns=rename)
    out["year"] = year
    out["utility_number"] = pd.to_numeric(out["utility_number"], errors="coerce")
    for col in [
        "residential_revenue_thousand_dollars", "residential_sales_mwh", "residential_customers",
        "commercial_revenue_thousand_dollars", "commercial_sales_mwh", "commercial_customers",
        "industrial_revenue_thousand_dollars", "industrial_sales_mwh", "industrial_customers",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["state"] = out["state"].astype(str).str.strip()
    out = out[out["state"].isin(ANALYSIS_STATES)]
    out = out[out["residential_customers"].fillna(0) >= MIN_RESIDENTIAL_CUSTOMERS]
    out = out[
        (out["residential_revenue_thousand_dollars"] > 0)
        & (out["residential_sales_mwh"] > 0)
    ].copy()
    out["residential_price_cents_kwh"] = (
        out["residential_revenue_thousand_dollars"] * 1000
        / out["residential_sales_mwh"] / 10
    )
    out["commercial_price_cents_kwh"] = (
        out["commercial_revenue_thousand_dollars"] * 1000
        / out["commercial_sales_mwh"].replace(0, np.nan) / 10
    )
    out["industrial_price_cents_kwh"] = (
        out["industrial_revenue_thousand_dollars"] * 1000
        / out["industrial_sales_mwh"].replace(0, np.nan) / 10
    )
    out["avg_monthly_bill_dollars"] = (
        out["residential_revenue_thousand_dollars"] * 1000
        / out["residential_customers"] / 12
    )
    out["utility_name_norm"] = out["utility_name"].map(normalize_utility_name)
    return out


def point_in_ring(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1):
            inside = not inside
    return inside


def point_in_polygon(x: float, y: float, rings: list[list[tuple[float, float]]]) -> bool:
    if not rings:
        return False
    if not point_in_ring(x, y, rings[0]):
        return False
    for hole in rings[1:]:
        if point_in_ring(x, y, hole):
            return False
    return True


@dataclass
class GeoFeature:
    bbox: tuple[float, float, float, float]
    polygons: list[list[list[tuple[float, float]]]]
    props: dict


def geometry_to_polygons(geom: dict) -> list[list[list[tuple[float, float]]]]:
    coords = geom.get("coordinates") or []
    if geom.get("type") == "Polygon":
        raw_polygons = [coords]
    elif geom.get("type") == "MultiPolygon":
        raw_polygons = coords
    else:
        return []

    polygons: list[list[list[tuple[float, float]]]] = []
    for poly in raw_polygons:
        rings = []
        for ring in poly:
            rings.append([(float(a), float(b)) for a, b, *_ in ring])
        if rings:
            polygons.append(rings)
    return polygons


def feature_bbox(polygons: list[list[list[tuple[float, float]]]]) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for poly in polygons:
        for ring in poly:
            xs.extend(x for x, _ in ring)
            ys.extend(y for _, y in ring)
    return min(xs), min(ys), max(xs), max(ys)


def load_features(path: Path) -> list[GeoFeature]:
    with path.open() as f:
        data = json.load(f)
    features: list[GeoFeature] = []
    for feat in data.get("features", []):
        polygons = geometry_to_polygons(feat.get("geometry") or {})
        if not polygons:
            continue
        features.append(GeoFeature(feature_bbox(polygons), polygons, feat.get("properties") or {}))
    return features


def point_matches(x: float, y: float, features: list[GeoFeature]) -> list[GeoFeature]:
    matches = []
    for feat in features:
        minx, miny, maxx, maxy = feat.bbox
        if x < minx or x > maxx or y < miny or y > maxy:
            continue
        if any(point_in_polygon(x, y, poly) for poly in feat.polygons):
            matches.append(feat)
    return matches


def infer_state(lon: float, lat: float, state_features: list[GeoFeature]) -> str | None:
    for feat in point_matches(lon, lat, state_features):
        name = str(feat.props.get("name") or "").upper()
        state = STATE_NAME_TO_ABBR.get(name)
        if state:
            return state
    return None


def clean_state_value(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "NULL"}:
        return None
    return text


def valid_utility_number(value) -> int | None:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return None
    num_int = int(num)
    return num_int if num_int > 0 and num_int != 99999 else None


def choose_service_feature(
    matches: list[GeoFeature],
    point_state: str | None,
    valid_pairs: set[tuple[str, int]],
    valid_utilities: set[int],
) -> GeoFeature | None:
    if not matches:
        return None

    def utility_number(feat: GeoFeature) -> int | None:
        return valid_utility_number(feat.props.get("ID"))

    def area(feat: GeoFeature) -> float:
        val = pd.to_numeric(feat.props.get("Shape__Area"), errors="coerce")
        return float(val) if pd.notna(val) and float(val) > 0 else math.inf

    with_ids = [feat for feat in matches if utility_number(feat) in valid_utilities]
    if point_state:
        in_state_pair = [
            feat for feat in with_ids
            if (point_state, utility_number(feat)) in valid_pairs
        ]
        if in_state_pair:
            return sorted(in_state_pair, key=area)[0]
        same_hifld_state = [
            feat for feat in with_ids
            if str(feat.props.get("STATE") or "").strip() == point_state
        ]
        if same_hifld_state:
            return sorted(same_hifld_state, key=area)[0]
    if with_ids:
        return sorted(with_ids, key=area)[0]
    return sorted(matches, key=area)[0]


def sampled_feature_states(feat: GeoFeature, state_features: list[GeoFeature]) -> set[str]:
    samples: list[tuple[float, float]] = []
    minx, miny, maxx, maxy = feat.bbox
    samples.extend([
        ((minx + maxx) / 2, (miny + maxy) / 2),
        (minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy),
    ])
    for poly in feat.polygons:
        if not poly:
            continue
        exterior = poly[0]
        step = max(1, len(exterior) // 50)
        samples.extend(exterior[::step])

    states = set()
    for lon, lat in samples:
        state = infer_state(lon, lat, state_features)
        if state:
            states.add(state)
    return states


def build_service_area_metadata(panel: pd.DataFrame) -> pd.DataFrame:
    state_features = load_features(RAW / "us_states.geojson")
    service_features = load_features(RAW / "electric-retail-service-territories-geojson.geojson")
    valid_pairs = set(zip(panel["state"], panel["utility_number"]))

    rows = []
    for feat in service_features:
        utility_number = valid_utility_number(feat.props.get("ID"))
        if utility_number is None:
            continue
        hifld_state = clean_state_value(feat.props.get("STATE"))
        sampled_states = sampled_feature_states(feat, state_features)
        candidate_states = sorted({hifld_state, *sampled_states} - {None})
        eia_states = sorted(st for st in candidate_states if (st, utility_number) in valid_pairs)
        if not eia_states and hifld_state and (hifld_state, utility_number) in valid_pairs:
            eia_states = [hifld_state]
        for state in eia_states:
            rows.append({
                "state": state,
                "utility_number": utility_number,
                "hifld_utility_name": feat.props.get("NAME"),
                "hifld_state": hifld_state,
                "sampled_states": ",".join(sorted(sampled_states)),
                "n_sampled_states": len(sampled_states),
                "single_state_service_area": int(sampled_states.issubset({state}) and hifld_state == state),
                "hifld_area": pd.to_numeric(feat.props.get("Shape__Area"), errors="coerce"),
            })

    meta = pd.DataFrame(rows)
    if meta.empty:
        return pd.DataFrame(columns=[
            "state", "utility_number", "single_state_service_area",
            "n_service_area_features", "service_area_states_sampled",
        ])
    collapsed = (
        meta.groupby(["state", "utility_number"], as_index=False)
        .agg(
            single_state_service_area=("single_state_service_area", "min"),
            n_service_area_features=("hifld_utility_name", "size"),
            service_area_states_sampled=("sampled_states", lambda s: ",".join(sorted(set(",".join(s).split(",")) - {""}))),
            hifld_area=("hifld_area", "sum"),
        )
    )
    return collapsed


def build_eia_state_panel() -> pd.DataFrame:
    frames = []
    for year in range(PANEL_START, PANEL_END + 1):
        parsed = parse_eia_sales_year(year)
        if not parsed.empty:
            frames.append(parsed)
    if not frames:
        raise RuntimeError("No EIA 861 Sales_Ult_Cust files were parsed.")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel[panel["utility_number"].notna()].copy()
    panel["utility_number"] = panel["utility_number"].astype(int)
    panel = panel[panel["utility_number"].ne(99999)].copy()
    sum_cols = [
        "residential_revenue_thousand_dollars", "residential_sales_mwh", "residential_customers",
        "commercial_revenue_thousand_dollars", "commercial_sales_mwh", "commercial_customers",
        "industrial_revenue_thousand_dollars", "industrial_sales_mwh", "industrial_customers",
    ]
    panel = (
        panel.groupby(["state", "utility_number", "year"], as_index=False)
        .agg(
            utility_name=("utility_name", "first"),
            ownership=("ownership", "first"),
            **{col: (col, "sum") for col in sum_cols},
        )
    )
    panel = panel[panel["residential_customers"].fillna(0) >= MIN_RESIDENTIAL_CUSTOMERS].copy()
    panel["residential_price_cents_kwh"] = (
        panel["residential_revenue_thousand_dollars"] * 1000
        / panel["residential_sales_mwh"].replace(0, np.nan) / 10
    )
    panel["commercial_price_cents_kwh"] = (
        panel["commercial_revenue_thousand_dollars"] * 1000
        / panel["commercial_sales_mwh"].replace(0, np.nan) / 10
    )
    panel["industrial_price_cents_kwh"] = (
        panel["industrial_revenue_thousand_dollars"] * 1000
        / panel["industrial_sales_mwh"].replace(0, np.nan) / 10
    )
    panel["avg_monthly_bill_dollars"] = (
        panel["residential_revenue_thousand_dollars"] * 1000
        / panel["residential_customers"].replace(0, np.nan) / 12
    )
    panel["utility_name_norm"] = panel["utility_name"].map(normalize_utility_name)
    panel["unit_id"] = panel["state"] + "_" + panel["utility_number"].astype(str)
    panel["state_year"] = panel["state"] + "_" + panel["year"].astype(str)
    panel["log_residential_price"] = np.log(panel["residential_price_cents_kwh"].clip(lower=0.01))
    panel["log_avg_monthly_bill"] = np.log(panel["avg_monthly_bill_dollars"].clip(lower=1.0))
    panel["log_residential_sales"] = np.log(panel["residential_sales_mwh"].clip(lower=1.0))
    panel["log_commercial_price"] = np.log(panel["commercial_price_cents_kwh"].clip(lower=0.01))
    panel["log_industrial_price"] = np.log(panel["industrial_price_cents_kwh"].clip(lower=0.01))
    return panel


def assign_facilities(panel: pd.DataFrame) -> pd.DataFrame:
    state_features = load_features(RAW / "us_states.geojson")
    service_features = load_features(RAW / "electric-retail-service-territories-geojson.geojson")
    valid_pairs = set(zip(panel["state"], panel["utility_number"]))
    valid_utilities = set(panel["utility_number"])

    facilities = pd.read_csv(DC_PATH)
    facilities["lat"] = pd.to_numeric(facilities["lat"], errors="coerce")
    facilities["lon"] = pd.to_numeric(facilities["lon"], errors="coerce")
    facilities["opening_year"] = pd.to_numeric(facilities["opening_year"], errors="coerce")
    facilities = facilities.dropna(subset=["lat", "lon", "opening_year"]).copy()
    facilities["opening_year"] = facilities["opening_year"].astype(int)

    records = []
    for _, row in facilities.iterrows():
        lon, lat = float(row["lon"]), float(row["lat"])
        inferred_state = infer_state(lon, lat, state_features)
        declared_state = clean_state_value(row.get("state"))
        point_state = inferred_state or declared_state
        matches = point_matches(lon, lat, service_features)
        chosen = choose_service_feature(matches, point_state, valid_pairs, valid_utilities)
        utility_number = valid_utility_number(chosen.props.get("ID")) if chosen else None
        utility_name = chosen.props.get("NAME") if chosen else None
        hifld_state = chosen.props.get("STATE") if chosen else None
        records.append({
            "osm_id": row.get("osm_id"),
            "name": row.get("name"),
            "operator": row.get("operator"),
            "lat": lat,
            "lon": lon,
            "declared_state": declared_state,
            "state": point_state,
            "opening_year": int(row["opening_year"]),
            "opening_year_confidence": row.get("opening_year_confidence"),
            "opening_year_method": row.get("opening_year_method"),
            "service_matches": len(matches),
            "utility_number": utility_number,
            "utility_name": utility_name,
            "hifld_state": hifld_state,
            "matched_eia_state_utility": (
                int((point_state, utility_number) in valid_pairs)
                if point_state and utility_number else 0
            ),
        })

    assigned = pd.DataFrame(records)
    assigned["in_analysis_state"] = assigned["state"].isin(ANALYSIS_STATES).astype(int)
    assigned["usable_assignment"] = (
        assigned["in_analysis_state"].eq(1)
        & assigned["utility_number"].notna()
        & assigned["matched_eia_state_utility"].eq(1)
    ).astype(int)
    return assigned


def add_cumulative_exposure(panel: pd.DataFrame, assigned: pd.DataFrame) -> pd.DataFrame:
    usable = assigned[assigned["usable_assignment"].eq(1)].copy()
    yearly = (
        usable.groupby(["state", "utility_number", "opening_year"], as_index=False)
        .size()
        .rename(columns={"size": "new_dc_count", "opening_year": "year"})
    )
    skeleton = panel[["state", "utility_number", "year"]].drop_duplicates()
    exposure = skeleton.merge(yearly, on=["state", "utility_number", "year"], how="left")
    exposure["new_dc_count"] = exposure["new_dc_count"].fillna(0).astype(int)
    exposure = exposure.sort_values(["state", "utility_number", "year"])
    exposure["cum_dc_count"] = exposure.groupby(["state", "utility_number"])["new_dc_count"].cumsum()
    out = panel.merge(exposure, on=["state", "utility_number", "year"], how="left")
    out["new_dc_count"] = out["new_dc_count"].fillna(0).astype(int)
    out["cum_dc_count"] = out["cum_dc_count"].fillna(0).astype(int)
    out["cum_dc_per_100k_customers"] = (
        out["cum_dc_count"] / out["residential_customers"].replace(0, np.nan) * 100_000
    )
    service_meta = build_service_area_metadata(out)
    out = out.merge(service_meta, on=["state", "utility_number"], how="left")
    out["single_state_service_area"] = out["single_state_service_area"].fillna(0).astype(int)
    out["n_service_area_features"] = out["n_service_area_features"].fillna(0).astype(int)
    return out


def residualize_two_way(
    data: pd.DataFrame,
    col: str,
    fe_a: str = "unit_id",
    fe_b: str = "state_year",
    max_iter: int = 200,
    tol: float = 1e-10,
) -> pd.Series:
    resid = data[col].astype(float) - data[col].astype(float).mean()
    for _ in range(max_iter):
        old = resid.copy()
        resid = resid - resid.groupby(data[fe_a]).transform("mean")
        resid = resid - resid.groupby(data[fe_b]).transform("mean")
        if float((resid - old).abs().max()) < tol:
            break
    return resid


def fit_fe_model(df: pd.DataFrame, outcome: str, treatment: str, sample_name: str) -> dict:
    cols = [outcome, treatment, "unit_id", "state_year"]
    work = df.dropna(subset=cols).copy()
    y = residualize_two_way(work, outcome)
    x_resid = residualize_two_way(work, treatment)
    x = pd.DataFrame({treatment: x_resid})
    model = sm.OLS(y, x)
    try:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": work["unit_id"].to_numpy()})
    except Exception:
        res = model.fit(cov_type="HC1")
    coef = res.params.get(treatment, np.nan)
    se = res.bse.get(treatment, np.nan)
    pval = res.pvalues.get(treatment, np.nan)
    return {
        "sample": sample_name,
        "outcome": outcome,
        "treatment": treatment,
        "coefficient": float(coef),
        "std_error": float(se),
        "p_value": float(pval),
        "ci_low": float(coef - 1.96 * se),
        "ci_high": float(coef + 1.96 * se),
        "n_obs": int(len(work)),
        "n_units": int(work["unit_id"].nunique()),
        "n_state_year_fe": int(work["state_year"].nunique()),
        "r_squared": float(res.rsquared),
    }


def run_models(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    samples = [
        ("full_polygon_assigned_panel", panel),
        ("single_state_service_areas", panel[panel["single_state_service_area"].eq(1)].copy()),
    ]
    for sample_name, sample_df in samples:
        for treatment in ["cum_dc_count", "cum_dc_per_100k_customers"]:
            for outcome in [
                "log_residential_price",
                "log_avg_monthly_bill",
                "log_residential_sales",
                "log_commercial_price",
                "log_industrial_price",
            ]:
                rows.append(fit_fe_model(sample_df, outcome, treatment, sample_name))
    return pd.DataFrame(rows)


def build_high_low_comparisons(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest_year = int(panel["year"].max())
    latest = panel[
        panel["year"].eq(latest_year)
        & panel["single_state_service_area"].eq(1)
    ].copy()
    latest = latest[latest["n_service_area_features"].gt(0)].copy()
    latest["dc_group"] = np.where(latest["cum_dc_count"].gt(0), "high_dc", "low_dc")
    candidates = latest[[
        "state", "utility_number", "utility_name", "ownership", "residential_customers",
        "residential_price_cents_kwh", "avg_monthly_bill_dollars", "cum_dc_count",
        "cum_dc_per_100k_customers", "n_service_area_features", "service_area_states_sampled",
        "dc_group",
    ]].sort_values(["state", "cum_dc_count", "residential_customers"], ascending=[True, False, False])

    pairs = []
    for state, grp in latest.groupby("state"):
        high = grp[grp["cum_dc_count"].gt(0)].sort_values(
            ["cum_dc_count", "residential_customers"], ascending=[False, False]
        )
        low = grp[grp["cum_dc_count"].eq(0)].sort_values("residential_customers", ascending=False)
        if high.empty or low.empty:
            continue
        h = high.iloc[0]
        l = low.iloc[0]
        pairs.append({
            "state": state,
            "year": latest_year,
            "high_utility_number": int(h["utility_number"]),
            "high_utility_name": h["utility_name"],
            "high_dc_count": int(h["cum_dc_count"]),
            "high_dc_per_100k": float(h["cum_dc_per_100k_customers"]),
            "high_residential_price_cents_kwh": float(h["residential_price_cents_kwh"]),
            "high_avg_monthly_bill_dollars": float(h["avg_monthly_bill_dollars"]),
            "high_residential_customers": float(h["residential_customers"]),
            "low_utility_number": int(l["utility_number"]),
            "low_utility_name": l["utility_name"],
            "low_dc_count": int(l["cum_dc_count"]),
            "low_dc_per_100k": float(l["cum_dc_per_100k_customers"]),
            "low_residential_price_cents_kwh": float(l["residential_price_cents_kwh"]),
            "low_avg_monthly_bill_dollars": float(l["avg_monthly_bill_dollars"]),
            "low_residential_customers": float(l["residential_customers"]),
            "price_gap_high_minus_low_cents_kwh": float(h["residential_price_cents_kwh"] - l["residential_price_cents_kwh"]),
            "bill_gap_high_minus_low_dollars": float(h["avg_monthly_bill_dollars"] - l["avg_monthly_bill_dollars"]),
        })
    return candidates, pd.DataFrame(pairs)


def write_summaries(panel: pd.DataFrame, assigned: pd.DataFrame, results: pd.DataFrame) -> None:
    assignment_summary = pd.DataFrame([
        {"metric": "all_data_center_points", "value": len(assigned)},
        {"metric": "points_in_analysis_states", "value": int(assigned["in_analysis_state"].sum())},
        {"metric": "points_with_service_polygon", "value": int(assigned["utility_number"].notna().sum())},
        {"metric": "usable_points_matched_to_eia_state_utility", "value": int(assigned["usable_assignment"].sum())},
        {"metric": "points_with_multiple_service_polygon_matches", "value": int(assigned["service_matches"].gt(1).sum())},
        {"metric": "panel_rows", "value": len(panel)},
        {"metric": "panel_units", "value": panel["unit_id"].nunique()},
        {"metric": "panel_states", "value": panel["state"].nunique()},
        {"metric": "single_state_service_area_panel_rows", "value": int(panel["single_state_service_area"].sum())},
        {"metric": "single_state_service_area_units", "value": int(panel.loc[panel["single_state_service_area"].eq(1), "unit_id"].nunique())},
        {"metric": "panel_min_year", "value": int(panel["year"].min())},
        {"metric": "panel_max_year", "value": int(panel["year"].max())},
    ])
    assignment_summary.to_csv(OUT / "assignment_summary.csv", index=False)

    latest_year = int(panel["year"].max())
    top = (
        panel[panel["year"].eq(latest_year)]
        .sort_values("cum_dc_count", ascending=False)
        [[
            "state", "utility_number", "utility_name", "year", "residential_customers",
            "residential_price_cents_kwh", "avg_monthly_bill_dollars",
            "cum_dc_count", "cum_dc_per_100k_customers",
        ]]
        .head(25)
    )
    top.to_csv(OUT / "top_exposed_utilities_latest_year.csv", index=False)

    compact = results.copy()
    compact["percent_effect_approx"] = compact["coefficient"] * 100
    compact["ci_low_percent"] = compact["ci_low"] * 100
    compact["ci_high_percent"] = compact["ci_high"] * 100
    compact.to_csv(OUT / "service_territory_fe_results_readable.csv", index=False)

    candidates, pairs = build_high_low_comparisons(panel)
    candidates.to_csv(OUT / "single_state_utility_candidates_latest_year.csv", index=False)
    pairs.to_csv(OUT / "within_state_high_low_pairs_latest_year.csv", index=False)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_eia_state_panel()
    assigned = assign_facilities(panel)
    service_panel = add_cumulative_exposure(panel, assigned)

    assigned.to_csv(OUT / "facility_service_territory_assignments.csv", index=False)
    service_panel.to_csv(OUT / "utility_state_year_panel.csv", index=False)
    results = run_models(service_panel)
    results.to_csv(OUT / "service_territory_fe_results.csv", index=False)
    write_summaries(service_panel, assigned, results)

    print(f"Wrote {OUT / 'utility_state_year_panel.csv'} ({len(service_panel):,} rows)")
    print(f"Wrote {OUT / 'service_territory_fe_results.csv'}")
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
