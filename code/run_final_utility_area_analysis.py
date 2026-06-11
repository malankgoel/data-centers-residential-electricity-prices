"""Estimate the utility-service-area specifications."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FINAL = PROCESSED / "final"
FINAL.mkdir(parents=True, exist_ok=True)


def _add_result(rows, **kw):
    rows.append(kw)


def _fit_fe_ols(df: pd.DataFrame, formula: str, cluster_var: str, label: str) -> dict:
    df = df.dropna(subset=[cluster_var]).copy()
    res = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster_var]}
    )
    return {
        "label": label,
        "formula": formula,
        "n_obs": int(res.nobs),
        "params": res.params.to_dict(),
        "se": res.bse.to_dict(),
        "pvalue": res.pvalues.to_dict(),
        "r2_adj": float(res.rsquared_adj),
    }


def run_annual_historical():
    """Spec 1: 2010-2024 annual log price ~ cum_dc_count + utility FE + state-year FE."""
    annual = pd.read_csv(PROCESSED / "utility_service_territory" / "utility_state_year_panel.csv")
    annual["unit_id"] = annual["state"] + "_" + annual["utility_number"].astype(str)
    annual["state_year"] = annual["state"] + "_" + annual["year"].astype(str)
    annual = annual.dropna(subset=["log_residential_price", "cum_dc_count"])
    annual = annual[annual["residential_customers"] >= 1000]

    rows = []

    for label, treat in [
        ("annual_cum_dc_count", "cum_dc_count"),
        ("annual_cum_dc_per_100k", "cum_dc_per_100k_customers"),
    ]:
        formula = f"log_residential_price ~ {treat} + C(unit_id) + C(state_year)"
        res = _fit_fe_ols(annual, formula, cluster_var="unit_id", label=label)
        beta = res["params"][treat]
        se = res["se"][treat]
        pval = res["pvalue"][treat]
        _add_result(
            rows,
            spec="1_historical_annual_2010_2024",
            sample="full_polygon_assigned_utility_state_panel",
            outcome="log_residential_price",
            treatment=treat,
            coefficient=beta,
            std_error=se,
            p_value=pval,
            n_obs=res["n_obs"],
            r2_adj=res["r2_adj"],
            percent_effect_approx=beta * 100,
        )

    return pd.DataFrame(rows)


def run_monthly_passthrough(df: pd.DataFrame) -> pd.DataFrame:
    """Spec 2: monthly PJM capacity-price pass-through."""
    work = df.dropna(subset=["log_residential_price", "pjm_capacity_price_mw_day"]).copy()
    work["state_year_month"] = work["state"] + "_" + work["year"].astype(str) + "_" + work["month"].astype(str).str.zfill(2)
    work["year_month"] = work["year"].astype(str) + "_" + work["month"].astype(str).str.zfill(2)

    rows = []

    for label, sample_filter in [
        ("all_utilities_monthly", lambda d: d),
        ("pjm_utilities_only_monthly", lambda d: d[d["in_pjm"] == 1]),
        ("pjm_state_based_monthly", lambda d: d[d["is_pjm_state"] == 1]),
    ]:
        sub = sample_filter(work)
        if sub["unit_id"].nunique() < 5:
            continue

        for treat in ["log_pjm_capacity_price", "pjm_capacity_price_mw_day"]:
            sub2 = sub.dropna(subset=[treat])
            if sub2["unit_id"].nunique() < 5:
                continue
            formula = (
                f"log_residential_price ~ {treat} + C(unit_id) + C(year_month)"
            )
            try:
                res = _fit_fe_ols(sub2, formula, cluster_var="unit_id", label=f"{label}_{treat}")
            except Exception as e:
                print(f"  skip {label}/{treat}: {e}")
                continue
            beta = res["params"][treat]
            se = res["se"][treat]
            pval = res["pvalue"][treat]
            _add_result(
                rows,
                spec="2_monthly_passthrough",
                sample=label,
                outcome="log_residential_price",
                treatment=treat,
                coefficient=beta,
                std_error=se,
                p_value=pval,
                n_obs=res["n_obs"],
                r2_adj=res["r2_adj"],
                percent_effect_approx=beta * 100,
            )

    return pd.DataFrame(rows)


def run_heterogeneity_did(df: pd.DataFrame) -> pd.DataFrame:
    """Spec 3: high-DC x post-shock DiD inside PJM."""
    pjm = df[df["in_pjm"] == 1].dropna(subset=["log_residential_price"]).copy()
    pjm = pjm[pjm["residential_customers"] >= 1000]

    rows = []

    pjm["high_dc_x_post"] = pjm["high_dc_utility"] * pjm["pjm_shock_active"]
    formula = (
        "log_residential_price ~ high_dc_x_post + C(unit_id) + C(year_month)"
    )
    res = _fit_fe_ols(pjm, formula, cluster_var="unit_id", label="pjm_high_dc_did")
    treat = "high_dc_x_post"
    _add_result(
        rows,
        spec="3_high_dc_heterogeneity_did_pjm",
        sample="pjm_utilities_only",
        outcome="log_residential_price",
        treatment=treat,
        coefficient=res["params"][treat],
        std_error=res["se"][treat],
        p_value=res["pvalue"][treat],
        n_obs=res["n_obs"],
        r2_adj=res["r2_adj"],
        percent_effect_approx=res["params"][treat] * 100,
    )

    pjm["some_or_high_dc"] = ((pjm["high_dc_utility"] == 1) | (pjm["some_dc_utility"] == 1)).astype(int)
    pjm["any_dc_x_post"] = pjm["some_or_high_dc"] * pjm["pjm_shock_active"]
    formula2 = (
        "log_residential_price ~ any_dc_x_post + C(unit_id) + C(year_month)"
    )
    res2 = _fit_fe_ols(pjm, formula2, cluster_var="unit_id", label="pjm_any_dc_did")
    treat = "any_dc_x_post"
    _add_result(
        rows,
        spec="3_any_dc_heterogeneity_did_pjm",
        sample="pjm_utilities_only",
        outcome="log_residential_price",
        treatment=treat,
        coefficient=res2["params"][treat],
        std_error=res2["se"][treat],
        p_value=res2["pvalue"][treat],
        n_obs=res2["n_obs"],
        r2_adj=res2["r2_adj"],
        percent_effect_approx=res2["params"][treat] * 100,
    )

    return pd.DataFrame(rows)


def run_single_state_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Spec 4: triple-DiD on single-state utility areas."""
    work = df[df["single_state_service_area"] == 1].copy()
    work = work.dropna(subset=["log_residential_price"])

    rows = []

    if work["unit_id"].nunique() < 5:
        return pd.DataFrame(rows)

    work["in_pjm_x_post"] = work["in_pjm"] * work["pjm_shock_active"]
    work["high_dc_x_post"] = work["high_dc_utility"] * work["pjm_shock_active"]
    work["in_pjm_x_high_dc_x_post"] = (
        work["in_pjm"] * work["high_dc_utility"] * work["pjm_shock_active"]
    )

    formula = (
        "log_residential_price ~ in_pjm_x_post + high_dc_x_post + "
        "in_pjm_x_high_dc_x_post + C(unit_id) + C(year_month)"
    )
    try:
        res = _fit_fe_ols(work, formula, cluster_var="unit_id", label="single_state_triple_did")
        for treat in ["in_pjm_x_post", "high_dc_x_post", "in_pjm_x_high_dc_x_post"]:
            if treat not in res["params"]:
                continue
            _add_result(
                rows,
                spec="4_single_state_triple_did",
                sample="single_state_service_areas_only",
                outcome="log_residential_price",
                treatment=treat,
                coefficient=res["params"][treat],
                std_error=res["se"][treat],
                p_value=res["pvalue"][treat],
                n_obs=res["n_obs"],
                r2_adj=res["r2_adj"],
                percent_effect_approx=res["params"][treat] * 100,
            )
    except Exception as e:
        print(f"  spec 4 single-state triple-DiD skipped: {e}")

    pjm_ss = work[work["in_pjm"] == 1].copy()
    if pjm_ss["unit_id"].nunique() >= 5:
        formula = "log_residential_price ~ high_dc_x_post + C(unit_id) + C(year_month)"
        try:
            res = _fit_fe_ols(pjm_ss, formula, cluster_var="unit_id", label="single_state_pjm_high_dc")
            treat = "high_dc_x_post"
            _add_result(
                rows,
                spec="4_single_state_high_dc_did_pjm",
                sample="single_state_pjm_utilities_only",
                outcome="log_residential_price",
                treatment=treat,
                coefficient=res["params"][treat],
                std_error=res["se"][treat],
                p_value=res["pvalue"][treat],
                n_obs=res["n_obs"],
                r2_adj=res["r2_adj"],
                percent_effect_approx=res["params"][treat] * 100,
            )
        except Exception as e:
            print(f"  spec 4 single-state high-DC DiD skipped: {e}")

    return pd.DataFrame(rows)


def run_triple_diff_did_with_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Spec 5: triple-DiD with HDD/CDD controls."""
    work = df.dropna(subset=["log_residential_price", "hdd", "cdd"]).copy()
    work = work[work["residential_customers"] >= 1000]
    work["in_pjm_x_post"] = work["in_pjm"] * work["pjm_shock_active"]
    work["high_dc_x_post"] = work["high_dc_utility"] * work["pjm_shock_active"]
    work["in_pjm_x_high_dc_x_post"] = work["in_pjm"] * work["high_dc_utility"] * work["pjm_shock_active"]
    work["hdd_k"] = work["hdd"] / 1000.0
    work["cdd_k"] = work["cdd"] / 1000.0

    rows = []
    formula = (
        "log_residential_price ~ in_pjm_x_post + high_dc_x_post + in_pjm_x_high_dc_x_post "
        "+ hdd_k + cdd_k + C(unit_id) + C(year_month)"
    )
    try:
        res = _fit_fe_ols(work, formula, cluster_var="unit_id", label="triple_did_with_controls")
        for t in ["in_pjm_x_post", "high_dc_x_post", "in_pjm_x_high_dc_x_post", "hdd_k", "cdd_k"]:
            if t not in res["params"]:
                continue
            _add_result(
                rows,
                spec="5_triple_did_with_weather_controls",
                sample="states_with_hdd_cdd_coverage",
                outcome="log_residential_price",
                treatment=t,
                coefficient=res["params"][t],
                std_error=res["se"][t],
                p_value=res["pvalue"][t],
                n_obs=res["n_obs"],
                r2_adj=res["r2_adj"],
                percent_effect_approx=res["params"][t] * 100,
            )
    except Exception as e:
        print(f"  triple-DiD with controls skipped: {e}")

    return pd.DataFrame(rows)


def run_event_study(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate monthly PJM effects relative to May 2025."""
    work = df.dropna(subset=["log_residential_price"]).copy()
    work["unit_id"] = work["unit_id"].astype(str)
    work["pjm_x_year_month"] = work["in_pjm"].astype(str) + "_" + work["year_month"]
    work["year_month_dt"] = pd.to_datetime(
        work["year"].astype(str) + "-" + work["month"].astype(str).str.zfill(2) + "-01"
    )

    work["pjm_relative_month"] = (
        (work["year"] - 2025) * 12 + (work["month"] - 6)
    )
    work["pjm_x_relmonth"] = work["in_pjm"] * work["pjm_relative_month"]

    work = work[(work["year_month_dt"] >= "2023-01-01") & (work["year_month_dt"] <= "2026-03-01")].copy()

    rows = []
    months = sorted(work["year_month"].unique())
    ref_month = "2025_05"
    if ref_month not in months:
        ref_month = months[len(months) // 2]

    for ym in months:
        if ym == ref_month:
            _add_result(
                rows,
                year_month=ym,
                coefficient=0.0,
                std_error=0.0,
                p_value=1.0,
                n_pjm=int((work["in_pjm"][work["year_month"] == ym] == 1).sum()),
                n_nonpjm=int((work["in_pjm"][work["year_month"] == ym] == 0).sum()),
                is_reference=True,
            )
            continue
        sub = work[work["year_month"].isin([ref_month, ym])].copy()
        sub["is_focal_month"] = (sub["year_month"] == ym).astype(int)
        sub["pjm_x_focal"] = sub["in_pjm"] * sub["is_focal_month"]
        try:
            formula = "log_residential_price ~ pjm_x_focal + C(unit_id) + C(year_month)"
            res = _fit_fe_ols(sub, formula, cluster_var="unit_id", label=f"event_{ym}")
            _add_result(
                rows,
                year_month=ym,
                coefficient=res["params"]["pjm_x_focal"],
                std_error=res["se"]["pjm_x_focal"],
                p_value=res["pvalue"]["pjm_x_focal"],
                n_pjm=int((sub["in_pjm"] == 1).sum()),
                n_nonpjm=int((sub["in_pjm"] == 0).sum()),
                is_reference=False,
            )
        except Exception as e:
            print(f"  event {ym} skipped: {e}")
            continue

    return pd.DataFrame(rows)


def run_high_low_dc_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare Q1 price changes by DC group and PJM membership."""
    work = df.copy()
    work["dc_group"] = np.where(
        work["high_dc_utility"] == 1, "high_dc_10plus",
        np.where(work["some_dc_utility"] == 1, "some_dc_1to9", "zero_dc"),
    )

    def _utility_q1_avg(year):
        sub = work[
            (work["year"] == year)
            & (work["month"].isin([1, 2, 3]))
            & work["residential_price_cents_kwh"].notna()
            & (work["residential_customers"] >= 1000)
        ]
        return (
            sub.groupby(["unit_id", "in_pjm", "dc_group"], as_index=False)
            .agg(
                price=("residential_price_cents_kwh", "mean"),
                customers=("residential_customers", "mean"),
            )
            .rename(columns={"price": f"price_q1_{year}", "customers": f"customers_q1_{year}"})
        )

    q1_24 = _utility_q1_avg(2024)
    q1_25 = _utility_q1_avg(2025)
    q1_26 = _utility_q1_avg(2026)

    rows = []

    def _yoy(num_df, num_label, den_df, den_label):
        merged = num_df.merge(den_df, on=["unit_id", "in_pjm", "dc_group"], how="inner")
        merged["change_cents"] = merged[f"price_q1_{num_label}"] - merged[f"price_q1_{den_label}"]
        merged["change_pct"] = 100 * merged["change_cents"] / merged[f"price_q1_{den_label}"]
        out = (
            merged.groupby(["in_pjm", "dc_group"], as_index=False)
            .agg(
                n_utilities=("unit_id", "nunique"),
                mean_price_change_cents=("change_cents", "mean"),
                median_price_change_cents=("change_cents", "median"),
                mean_price_change_pct=("change_pct", "mean"),
                median_price_change_pct=("change_pct", "median"),
                customer_weighted_change_pct=("change_pct", lambda x: np.average(x, weights=merged.loc[x.index, f"customers_q1_{den_label}"])),
            )
        )
        out["comparison"] = f"Q1_{num_label}_vs_Q1_{den_label}"
        return out

    rows.append(_yoy(q1_26, "2026", q1_25, "2025"))
    rows.append(_yoy(q1_25, "2025", q1_24, "2024"))
    rows.append(_yoy(q1_26, "2026", q1_24, "2024"))

    return pd.concat(rows, ignore_index=True)


def run_triple_diff_did(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate the PJM x high-DC x post-shock triple-DiD."""
    work = df.dropna(subset=["log_residential_price"]).copy()
    work = work[work["residential_customers"] >= 1000]

    work["in_pjm_x_post"] = work["in_pjm"] * work["pjm_shock_active"]
    work["high_dc_x_post"] = work["high_dc_utility"] * work["pjm_shock_active"]
    work["in_pjm_x_high_dc_x_post"] = work["in_pjm"] * work["high_dc_utility"] * work["pjm_shock_active"]
    work["some_or_high_dc"] = ((work["high_dc_utility"] == 1) | (work["some_dc_utility"] == 1)).astype(int)
    work["any_dc_x_post"] = work["some_or_high_dc"] * work["pjm_shock_active"]
    work["in_pjm_x_any_dc_x_post"] = work["in_pjm"] * work["some_or_high_dc"] * work["pjm_shock_active"]

    rows = []
    for label, treats in [
        (
            "triple_did_high_dc",
            ["in_pjm_x_post", "high_dc_x_post", "in_pjm_x_high_dc_x_post"],
        ),
        (
            "triple_did_any_dc",
            ["in_pjm_x_post", "any_dc_x_post", "in_pjm_x_any_dc_x_post"],
        ),
    ]:
        formula = (
            "log_residential_price ~ "
            + " + ".join(treats)
            + " + C(unit_id) + C(year_month)"
        )
        try:
            res = _fit_fe_ols(work, formula, cluster_var="unit_id", label=label)
            for t in treats:
                _add_result(
                    rows,
                    spec="3b_triple_did",
                    sample=label,
                    outcome="log_residential_price",
                    treatment=t,
                    coefficient=res["params"][t],
                    std_error=res["se"][t],
                    p_value=res["pvalue"][t],
                    n_obs=res["n_obs"],
                    r2_adj=res["r2_adj"],
                    percent_effect_approx=res["params"][t] * 100,
                )
        except Exception as e:
            print(f"  {label} triple-DiD skipped: {e}")

    return pd.DataFrame(rows)


def run_monthly_pjm_index(df: pd.DataFrame) -> pd.DataFrame:
    """Build customer-weighted monthly price indexes."""
    work = df.copy()
    work = work[work["residential_price_cents_kwh"].notna() & work["residential_customers"].notna()]
    work = work[work["residential_customers"] >= 1000]
    work["dc_group"] = np.where(
        work["high_dc_utility"] == 1, "high_dc_10plus",
        np.where(work["some_dc_utility"] == 1, "some_dc_1to9", "zero_dc"),
    )

    work["weighted_price"] = work["residential_price_cents_kwh"] * work["residential_customers"]

    rows = []

    def _agg(group_cols, label):
        out = (
            work.groupby(group_cols + ["year", "month"], as_index=False)
            .agg(wp=("weighted_price", "sum"), cust=("residential_customers", "sum"))
        )
        out["weighted_price_cents_kwh"] = out["wp"] / out["cust"]
        out["year_month"] = (
            out["year"].astype(int).astype(str)
            + "-"
            + out["month"].astype(int).astype(str).str.zfill(2)
        )
        out["group_label"] = label
        return out

    def _rebase(sub_df, group_keys):
        base_rows = []
        for _, sub in sub_df.groupby(group_keys, dropna=False):
            sub = sub.sort_values(["year", "month"]).copy()
            base = sub[(sub["year"] == 2023) & (sub["month"] == 1)]["weighted_price_cents_kwh"]
            if base.empty:
                base_val = sub.iloc[0]["weighted_price_cents_kwh"]
            else:
                base_val = base.values[0]
            sub["index_jan_2023_100"] = 100 * sub["weighted_price_cents_kwh"] / base_val
            base_rows.append(sub)
        return pd.concat(base_rows, ignore_index=True) if base_rows else sub_df

    pjm_only = _rebase(_agg(["in_pjm"], "by_in_pjm"), ["in_pjm"])
    pjm_dc = _rebase(_agg(["in_pjm", "dc_group"], "by_in_pjm_x_dc"), ["in_pjm", "dc_group"])

    out = pd.concat([pjm_only, pjm_dc], ignore_index=True, sort=False)
    keep = [
        "group_label", "in_pjm", "dc_group", "year", "month", "year_month",
        "weighted_price_cents_kwh", "cust", "index_jan_2023_100",
    ]
    return out[[c for c in keep if c in out.columns]]


def run_state_zone_price_change(df: pd.DataFrame) -> pd.DataFrame:
    """Customer-weighted mean residential price by state and PJM zone for Q1 2025, Q1 2026."""
    work = df.copy()
    work = work[work["residential_price_cents_kwh"].notna() & work["residential_customers"].notna()]
    work = work[work["residential_customers"] >= 1000]

    rows = []
    for period_label, mask in [
        ("Q1_2025", (work["year"] == 2025) & (work["month"].isin([1, 2, 3]))),
        ("Q1_2026", (work["year"] == 2026) & (work["month"].isin([1, 2, 3]))),
    ]:
        sub = work[mask].copy()
        utility_avg = (
            sub.groupby(["unit_id", "state", "pjm_zone", "in_pjm"], as_index=False)
            .agg(
                price=("residential_price_cents_kwh", "mean"),
                customers=("residential_customers", "mean"),
            )
        )
        for grp_label, grp_cols in [
            ("by_state", ["state", "in_pjm"]),
            ("by_pjm_zone", ["pjm_zone"]),
        ]:
            grouped_data = utility_avg.dropna(subset=grp_cols).copy()
            grouped_data["weighted_price"] = grouped_data["price"] * grouped_data["customers"]
            grouped = (
                grouped_data.groupby(grp_cols, as_index=False)
                .agg(
                    sum_wp=("weighted_price", "sum"),
                    sum_customers=("customers", "sum"),
                    n_utilities=("unit_id", "nunique"),
                )
            )
            grouped["weighted_price_cents_kwh"] = grouped["sum_wp"] / grouped["sum_customers"]
            grouped["period"] = period_label
            grouped["group_kind"] = grp_label
            rows.append(grouped[grp_cols + ["weighted_price_cents_kwh", "n_utilities", "sum_customers", "period", "group_kind"]])

    out = pd.concat(rows, ignore_index=True)
    pivot_rows = []
    for grp_kind, sub in out.groupby("group_kind"):
        keys = [c for c in ["state", "in_pjm", "pjm_zone"] if c in sub.columns and sub[c].notna().any()]
        wide = sub.pivot_table(index=keys, columns="period", values="weighted_price_cents_kwh", aggfunc="first").reset_index()
        cust_wide = sub.pivot_table(index=keys, columns="period", values="sum_customers", aggfunc="first").reset_index()
        n_wide = sub.pivot_table(index=keys, columns="period", values="n_utilities", aggfunc="first").reset_index()
        merged = wide.merge(cust_wide, on=keys, suffixes=("_price", "_customers"))
        merged = merged.merge(n_wide, on=keys, suffixes=("", "_nutil"))
        if "Q1_2025_price" in merged.columns and "Q1_2026_price" in merged.columns:
            merged["price_change_cents"] = merged["Q1_2026_price"] - merged["Q1_2025_price"]
            merged["price_change_pct"] = 100 * merged["price_change_cents"] / merged["Q1_2025_price"]
        merged["group_kind"] = grp_kind
        pivot_rows.append(merged)
    if pivot_rows:
        return pd.concat(pivot_rows, ignore_index=True)
    return pd.DataFrame()


def run_threshold_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity of the triple-DiD coefficient to the high-DC cutoff."""
    base = df.dropna(subset=["log_residential_price", "cum_dc_count"]).copy()
    base = base[base["residential_customers"] >= 1000]

    rows = []
    for threshold in [5, 10, 20, 50]:
        work = base.copy()
        high_col = f"high_dc_ge_{threshold}"
        high_post = f"{high_col}_x_post"
        triple = f"in_pjm_x_{high_col}_x_post"
        work[high_col] = (work["cum_dc_count"] >= threshold).astype(int)
        work["in_pjm_x_post"] = work["in_pjm"] * work["pjm_shock_active"]
        work[high_post] = work[high_col] * work["pjm_shock_active"]
        work[triple] = work["in_pjm"] * work[high_col] * work["pjm_shock_active"]
        formula = (
            f"log_residential_price ~ in_pjm_x_post + {high_post} + {triple} "
            "+ C(unit_id) + C(year_month)"
        )
        try:
            res = _fit_fe_ols(work, formula, cluster_var="unit_id", label=f"threshold_{threshold}")
        except Exception as e:
            print(f"  threshold {threshold} skipped: {e}")
            continue
        latest = work.sort_values(["year", "month"]).drop_duplicates("unit_id", keep="last")
        rows.append({
            "threshold": threshold,
            "n_high_dc_units_total": int(latest[high_col].sum()),
            "n_high_dc_pjm_units": int((latest[high_col] * latest["in_pjm"]).sum()),
            "beta_triple_pp": res["params"].get(triple, np.nan) * 100,
            "se_triple_pp": res["se"].get(triple, np.nan) * 100,
            "p_triple": res["pvalue"].get(triple, np.nan),
            "beta_pjm_post_pp": res["params"].get("in_pjm_x_post", np.nan) * 100,
            "se_pjm_post_pp": res["se"].get("in_pjm_x_post", np.nan) * 100,
            "p_pjm_post": res["pvalue"].get("in_pjm_x_post", np.nan),
            "N": res["n_obs"],
        })
    return pd.DataFrame(rows)


def run_continuous_exposure_spec(df: pd.DataFrame) -> dict:
    """Continuous exposure version of the triple-DiD using DCs per 100k customers."""
    work = df.dropna(subset=[
        "log_residential_price", "cum_dc_per_100k_customers",
        "in_pjm", "pjm_shock_active",
    ]).copy()
    work = work[work["residential_customers"] >= 1000]
    work["dc_per_100k_x_post"] = work["cum_dc_per_100k_customers"] * work["pjm_shock_active"]
    work["in_pjm_x_post"] = work["in_pjm"] * work["pjm_shock_active"]
    work["in_pjm_x_dc_per_100k_x_post"] = (
        work["in_pjm"] * work["cum_dc_per_100k_customers"] * work["pjm_shock_active"]
    )
    formula = (
        "log_residential_price ~ in_pjm_x_post + dc_per_100k_x_post "
        "+ in_pjm_x_dc_per_100k_x_post + C(unit_id) + C(year_month)"
    )
    res = _fit_fe_ols(work, formula, cluster_var="unit_id", label="continuous_exposure")
    beta = res["params"].get("in_pjm_x_dc_per_100k_x_post", np.nan) * 100
    se = res["se"].get("in_pjm_x_dc_per_100k_x_post", np.nan) * 100

    latest = (
        work[(work["in_pjm"] == 1)]
        .sort_values(["year", "month"])
        .drop_duplicates("unit_id", keep="last")
        .sort_values("cum_dc_count", ascending=False)
        .head(3)
    )
    implied = []
    for _, r in latest.iterrows():
        exposure = float(r["cum_dc_per_100k_customers"])
        implied.append({
            "utility_name": r["utility_name"],
            "state": r["state"],
            "dc_per_100k": exposure,
            "implied_effect_pp": beta * exposure,
            "implied_se_pp": se * exposure if pd.notna(se) else np.nan,
        })

    return {
        "beta_pjm_post_pp": res["params"].get("in_pjm_x_post", np.nan) * 100,
        "se_pjm_post_pp": res["se"].get("in_pjm_x_post", np.nan) * 100,
        "p_pjm_post": res["pvalue"].get("in_pjm_x_post", np.nan),
        "beta_triple_per_dc_per_100k_pp": beta,
        "se_triple_per_dc_per_100k_pp": se,
        "p_triple": res["pvalue"].get("in_pjm_x_dc_per_100k_x_post", np.nan),
        "N": res["n_obs"],
        "implied_effects": implied,
    }


def main():
    print("Loading utility-area monthly panel...")
    df_monthly = pd.read_csv(FINAL / "utility_area_month_panel_2023_2026.csv")
    df_monthly["year"] = df_monthly["year"].astype(int)
    df_monthly["month"] = df_monthly["month"].astype(int)
    df_monthly["unit_id"] = df_monthly["unit_id"].astype(str)
    df_monthly["state"] = df_monthly["state"].astype(str)
    df_monthly["year_month"] = df_monthly["year"].astype(str) + "_" + df_monthly["month"].astype(str).str.zfill(2)

    print(f"Monthly panel: {len(df_monthly):,} rows, {df_monthly['unit_id'].nunique():,} utility-states")
    print()

    print("Spec 1: historical annual 2010-2024...")
    s1 = run_annual_historical()
    print(s1.to_string(index=False))
    print()

    print("Spec 2: monthly pass-through...")
    s2 = run_monthly_passthrough(df_monthly)
    print(s2.to_string(index=False))
    print()

    print("Spec 3: high-DC heterogeneity DiD...")
    s3 = run_heterogeneity_did(df_monthly)
    print(s3.to_string(index=False))
    print()

    print("Spec 4: single-state robustness...")
    s4 = run_single_state_robustness(df_monthly)
    print(s4.to_string(index=False))
    print()

    print("Spec 3b: triple-DiD (PJM x high-DC x post)...")
    s3b = run_triple_diff_did(df_monthly)
    print(s3b.to_string(index=False))
    print()

    print("Spec 5: triple-DiD with weather + gas controls (robustness)...")
    s5 = run_triple_diff_did_with_controls(df_monthly)
    print(s5.to_string(index=False))
    print()

    all_main = pd.concat([s1, s2, s3, s3b, s4, s5], ignore_index=True, sort=False)
    out_main = FINAL / "main_regression_results.csv"
    all_main.to_csv(out_main, index=False)
    print(f"  -> {out_main}")
    print()

    print("Event study (PJM vs non-PJM, monthly)...")
    ev = run_event_study(df_monthly)
    out_ev = FINAL / "event_study_monthly_coefficients.csv"
    ev.to_csv(out_ev, index=False)
    print(f"  -> {out_ev}")
    print(ev.tail(15).to_string(index=False))
    print()

    print("High-DC vs low-DC group comparison (Q1 2024/25/26)...")
    hl = run_high_low_dc_comparison(df_monthly)
    out_hl = FINAL / "high_low_dc_comparison.csv"
    hl.to_csv(out_hl, index=False)
    print(f"  -> {out_hl}")
    print(hl.to_string(index=False))
    print()

    print("State / PJM-zone price change summary (Q1 2025 -> Q1 2026)...")
    sz = run_state_zone_price_change(df_monthly)
    out_sz = FINAL / "state_zone_price_change_summary.csv"
    sz.to_csv(out_sz, index=False)
    print(f"  -> {out_sz}")
    print(sz.to_string(index=False))
    print()

    print("Monthly PJM-vs-non-PJM customer-weighted price index (Jan 2023 = 100)...")
    idx = run_monthly_pjm_index(df_monthly)
    out_idx = FINAL / "monthly_pjm_price_index.csv"
    idx.to_csv(out_idx, index=False)
    print(f"  -> {out_idx}")
    print(idx[idx["group_label"] == "by_in_pjm"].tail(20).to_string(index=False))

    print()
    print("Exposure-threshold and continuous-exposure robustness...")
    ts = run_threshold_sensitivity(df_monthly)
    out_ts = FINAL / "threshold_sensitivity.csv"
    ts.to_csv(out_ts, index=False)
    print(f"  -> {out_ts}")
    continuous = run_continuous_exposure_spec(df_monthly)
    out_cont = FINAL / "continuous_exposure_spec.json"
    out_cont.write_text(json.dumps(continuous, indent=2))
    print(f"  -> {out_cont}")


if __name__ == "__main__":
    main()
