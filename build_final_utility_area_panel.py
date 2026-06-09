"""Build the final utility-service-area monthly panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
RAW = DATA / "raw"
OUT_DIR = PROCESSED / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_monthly_panel() -> pd.DataFrame:
    path = PROCESSED / "utility_service_territory" / "eia861m_monthly_service_territory_panel_2023_2026.csv"
    df = pd.read_csv(path)
    df["utility_number"] = df["utility_number"].astype(int)
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    df["unit_id"] = df["state"] + "_" + df["utility_number"].astype(str)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["avg_monthly_bill_dollars"] = (
            df["residential_revenue_thousand_dollars"] * 1000.0
            / df["residential_customers"].replace({0: np.nan})
        )
    return df


def _load_zone_crosswalk() -> pd.DataFrame:
    path = PROCESSED / "pjm" / "utility_to_pjm_zone_crosswalk.csv"
    cw = pd.read_csv(path)
    cw["utility_number"] = cw["utility_number"].astype(int)
    return cw[["utility_number", "state", "pjm_zone", "in_pjm", "mapping_confidence"]]


def _load_pjm_capacity_prices() -> pd.DataFrame:
    """PJM BRA prices by LDA expanded to monthly via June-May delivery year."""
    bra = pd.read_csv(PROCESSED / "pjm" / "pjm_bra_clearing_prices_by_lda.csv")
    rows = []
    for _, r in bra.iterrows():
        dy = r["delivery_year"]
        if "/" not in str(dy):
            continue
        start_year = int(str(dy).split("/")[0])
        for offset in range(12):
            calc_year = start_year if offset < 7 else start_year + 1
            month = 6 + offset if offset < 7 else offset - 6
            rows.append({
                "delivery_year": dy,
                "lda": r["lda"],
                "clearing_price_mw_day": r["clearing_price_mw_day"],
                "rto_price_mw_day": r["rto_price_mw_day"],
                "year": calc_year,
                "month": month,
            })
    monthly_cap = pd.DataFrame(rows)
    monthly_cap = monthly_cap.rename(columns={
        "lda": "pjm_zone",
        "clearing_price_mw_day": "pjm_capacity_price_mw_day",
        "rto_price_mw_day": "pjm_rto_capacity_price_mw_day",
        "delivery_year": "pjm_bra_delivery_year",
    })
    return monthly_cap


def _load_hdd_cdd() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "degree_days_state_month.csv")[
        ["state", "year", "month", "hdd", "cdd"]
    ]


def _load_henry_hub() -> pd.DataFrame:
    df = pd.read_excel(RAW / "henry_hub_monthly.xls", sheet_name="Data 1", header=2)
    df.columns = ["date", "henry_hub_price"]
    df = df.dropna()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["henry_hub_price"] = pd.to_numeric(df["henry_hub_price"], errors="coerce")
    return df[["year", "month", "henry_hub_price"]]


def build_final_panel() -> pd.DataFrame:
    monthly = _load_monthly_panel()
    crosswalk = _load_zone_crosswalk()
    cap_prices = _load_pjm_capacity_prices()
    hdd_cdd = _load_hdd_cdd()
    hh = _load_henry_hub()

    df = monthly.merge(
        crosswalk,
        how="left",
        on=["utility_number", "state"],
        validate="m:1",
    )

    df["pjm_zone_for_join"] = df["pjm_zone"]
    df = df.merge(
        cap_prices,
        how="left",
        left_on=["pjm_zone_for_join", "year", "month"],
        right_on=["pjm_zone", "year", "month"],
        suffixes=("", "_caprow"),
    ).drop(columns=["pjm_zone_caprow", "pjm_zone_for_join"], errors="ignore")

    df = df.merge(hdd_cdd, how="left", on=["state", "year", "month"])

    df = df.merge(hh, how="left", on=["year", "month"])

    df["in_pjm"] = df["in_pjm"].fillna(0).astype(int)

    pjm_states_existing = {"VA", "MD", "NJ", "OH", "PA", "WV", "IL", "NC", "DE", "DC"}
    df["is_pjm_state"] = df["state"].isin(pjm_states_existing).astype(int)

    df["log_residential_price"] = np.log(df["residential_price_cents_kwh"].replace({0: np.nan}))
    df["log_avg_monthly_bill"] = np.log(df["avg_monthly_bill_dollars"].replace({0: np.nan}))

    df["high_dc_utility"] = ((df["cum_dc_count"].fillna(0) >= 10)).astype(int)
    df["some_dc_utility"] = ((df["cum_dc_count"].fillna(0) >= 1) & (df["cum_dc_count"].fillna(0) < 10)).astype(int)
    df["zero_dc_utility"] = ((df["cum_dc_count"].fillna(0) == 0) & df["cum_dc_count"].notna()).astype(int)

    df["pjm_shock_active"] = ((df["year"] > 2025) | ((df["year"] == 2025) & (df["month"] >= 6))).astype(int)

    df["log_pjm_capacity_price"] = np.log(df["pjm_capacity_price_mw_day"].clip(lower=1))
    df["log_henry_hub"] = np.log(df["henry_hub_price"])

    df["state_year_month"] = df["state"] + "_" + df["year"].astype(str) + "_" + df["month"].astype(str).str.zfill(2)
    df["year_month"] = df["year"].astype(str) + "_" + df["month"].astype(str).str.zfill(2)

    keep_cols = [
        "year", "month", "date", "state",
        "utility_number", "utility_name", "unit_id",
        "ownership", "data_status",
        "pjm_zone", "in_pjm", "is_pjm_state", "mapping_confidence",
        "residential_price_cents_kwh", "log_residential_price",
        "avg_monthly_bill_dollars", "log_avg_monthly_bill",
        "residential_sales_mwh", "residential_customers",
        "commercial_price_cents_kwh", "industrial_price_cents_kwh",
        "cum_dc_count", "cum_dc_per_100k_customers",
        "high_dc_utility", "some_dc_utility", "zero_dc_utility",
        "single_state_service_area",
        "pjm_bra_delivery_year", "pjm_capacity_price_mw_day", "pjm_rto_capacity_price_mw_day",
        "log_pjm_capacity_price", "pjm_shock_active",
        "hdd", "cdd", "henry_hub_price", "log_henry_hub",
        "state_year_month", "year_month",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]
    return df


def main():
    df = build_final_panel()
    out_path = OUT_DIR / "utility_area_month_panel_2023_2026.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(df):,} rows, {df['unit_id'].nunique():,} utility-states")
    print()
    print("Coverage by year and PJM membership:")
    print(df.groupby(["year", "in_pjm"])["unit_id"].nunique().unstack(fill_value=0))
    print()
    print("PJM zone counts (latest year):")
    latest = df[df["year"] == df["year"].max()]
    print(latest[latest["in_pjm"] == 1].groupby("pjm_zone")["unit_id"].nunique())
    print()
    print("Capacity price merge sanity (PJM only, by delivery year):")
    pjm_only = df[df["in_pjm"] == 1].dropna(subset=["pjm_capacity_price_mw_day"])
    print(
        pjm_only.groupby(["pjm_bra_delivery_year"])
        .agg(min_price=("pjm_capacity_price_mw_day", "min"),
             max_price=("pjm_capacity_price_mw_day", "max"),
             n_rows=("unit_id", "size"))
    )


if __name__ == "__main__":
    main()
