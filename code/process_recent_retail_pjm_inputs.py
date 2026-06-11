#!/usr/bin/env python3
"""Normalize recent EIA-861M and PJM BRA inputs."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import fitz
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from utility_config import RAW, ROOT


PROCESSED = ROOT / "data" / "processed"
OUT_861M = PROCESSED / "utility_service_territory"
OUT_PJM = PROCESSED / "pjm"

EIA861M_COLS = [
    "year", "month", "utility_number", "utility_name", "state", "ownership", "data_status",
    "residential_revenue_thousand_dollars", "residential_sales_mwh", "residential_customers",
    "commercial_revenue_thousand_dollars", "commercial_sales_mwh", "commercial_customers",
    "industrial_revenue_thousand_dollars", "industrial_sales_mwh", "industrial_customers",
    "transportation_revenue_thousand_dollars", "transportation_sales_mwh", "transportation_customers",
    "total_revenue_thousand_dollars", "total_sales_mwh", "total_customers",
]

PJM_ZONE_ROWS = [
    ("AE", "Atlantic City Electric Co.", "NJ"),
    ("AEP", "American Electric Power", "OH,VA,WV"),
    ("AP", "Allegheny Power Systems", "MD,PA,WV,VA"),
    ("ATSI", "American Transmission Systems, Inc.", "OH,PA"),
    ("BGE", "Baltimore Gas and Electric Co.", "MD"),
    ("COMED", "Commonwealth Edison Company", "IL"),
    ("DAY", "Dayton Power & Light Co.", "OH"),
    ("DEOK", "Duke Energy Ohio and Kentucky Corp.", "OH,KY"),
    ("DLCO", "Duquesne Light Co.", "PA"),
    ("DOM", "Dominion", "VA,NC"),
    ("DP&L", "Delmarva Power & Light Co.", "DE,MD,VA"),
    ("EKPC", "East Kentucky Power Cooperative", "KY"),
    ("JCPL", "Jersey Central Power & Light", "NJ"),
    ("METED", "Met-Ed", "PA"),
    ("OVEC", "Ohio Valley Electric Corp.", "OH,WV"),
    ("PECO", "PECO Energy Co.", "PA"),
    ("PENELEC", "Pennsylvania Electric Co.", "PA"),
    ("PEPCO", "Potomac Electric Power Co.", "DC,MD"),
    ("PPL", "PPL Electric Utilities", "PA"),
    ("PSEG", "PSEG", "NJ"),
    ("RECO", "Rockland Electric Co.", "NJ"),
]

BRA_REPORTS = {
    "2020/2021": RAW / "pjm" / "2020-2021-base-residual-auction-report.pdf",
    "2021/2022": RAW / "pjm" / "2021-2022-base-residual-auction-report.pdf",
    "2022/2023": RAW / "pjm" / "2022-2023-base-residual-auction-report.pdf",
    "2023/2024": RAW / "pjm" / "2023-2024-base-residual-auction-report.pdf",
    "2024/2025": RAW / "pjm" / "2024-2025-base-residual-auction-report.pdf",
    "2025/2026": RAW / "pjm" / "2025-2026-base-residual-auction-report.pdf",
    "2026/2027": RAW / "pjm" / "2026-2027-bra-report.pdf",
}

# Keyed official BRA values keep the output deterministic.
BRA_LDA_PRICES = {
    "2020/2021": {
        "RTO": 76.53, "MAAC": 86.04, "EMAAC": 187.87, "COMED": 188.12, "DEOK": 130.00,
    },
    "2021/2022": {
        "RTO": 140.00, "EMAAC": 165.73, "PSEG": 204.29, "BGE": 200.30,
        "ATSI": 171.33, "COMED": 195.55,
    },
    "2022/2023": {
        "RTO": 50.00, "MAAC": 95.79, "EMAAC": 97.86, "BGE": 126.50,
        "COMED": 68.96, "DEOK": 71.69,
    },
    "2023/2024": {
        "RTO": 34.13, "MAAC": 49.49, "DPL-SOUTH": 69.95, "BGE": 69.95,
    },
    "2024/2025": {
        "RTO": 28.92, "MAAC": 49.49, "SWMAAC": 49.49, "PEPCO": 49.49,
        "BGE": 73.00, "EMAAC": 53.60, "DPL-SOUTH": 426.17, "PSEG": 53.60,
        "PS-NORTH": 53.60, "ATSI": 28.92, "ATSI-CLEVELAND": 28.92,
        "PPL": 49.49, "COMED": 28.92, "DAY": 28.92, "DEOK": 96.24,
    },
    "2025/2026": {
        "RTO": 269.92, "ATSI": 269.92, "ATSI-CLEVELAND": 269.92, "COMED": 269.92,
        "DAY": 269.92, "DEOK": 269.92, "DOM": 444.26, "MAAC": 269.92,
        "PPL": 269.92, "EMAAC": 269.92, "DPL-SOUTH": 269.92, "PSEG": 269.92,
        "PS-NORTH": 269.92, "SWMAAC": 269.92, "BGE": 466.35, "PEPCO": 269.92,
    },
    "2026/2027": {
        "RTO": 329.17, "ATSI": 329.17, "ATSI-CLEVELAND": 329.17, "COMED": 329.17,
        "DAY": 329.17, "DEOK": 329.17, "DOM": 329.17, "MAAC": 329.17,
        "PPL": 329.17, "EMAAC": 329.17, "DPL-SOUTH": 329.17, "PSEG": 329.17,
        "PS-NORTH": 329.17, "JCPL": 329.17, "SWMAAC": 329.17, "BGE": 329.17,
        "PEPCO": 329.17,
    },
}

BRA_RTO_SUMMARY = {
    "2020/2021": {"rto_price": 76.53, "cleared_ucap_mw": 165109.2, "total_reserve_margin_pct": 23.3},
    "2021/2022": {"rto_price": 140.00, "cleared_ucap_mw": 163627.3, "total_reserve_margin_pct": 21.5},
    "2022/2023": {"rto_price": 50.00, "cleared_ucap_mw": 144477.3, "total_reserve_margin_pct": 19.9},
    "2023/2024": {"rto_price": 34.13, "cleared_ucap_mw": 144870.6, "total_reserve_margin_pct": 20.3},
    "2024/2025": {"rto_price": 28.92, "cleared_ucap_mw": 147478.9, "total_reserve_margin_pct": 20.4, "total_cost_to_load_billion": 2.2},
    "2025/2026": {"rto_price": 269.92, "cleared_ucap_mw": 135684.0, "total_reserve_margin_pct": 18.5, "total_cost_to_load_billion": 14.7},
    "2026/2027": {"rto_price": 329.17, "cleared_ucap_mw": 134205.3, "total_reserve_margin_pct": 18.9, "total_cost_to_load_billion": 16.1},
}


def clean_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({".": np.nan, "": np.nan}), errors="coerce")


def read_eia861m(path: Path, sheet: str = "Sales Ultimate Cust. -States") -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=3)
    raw = raw.iloc[:, :len(EIA861M_COLS)]
    raw.columns = EIA861M_COLS
    raw = raw[pd.to_numeric(raw["year"], errors="coerce").notna()].copy()

    text_cols = {"utility_name", "state", "ownership", "data_status"}
    for col in EIA861M_COLS:
        if col not in text_cols:
            raw[col] = clean_num(raw[col])

    raw["year"] = raw["year"].astype(int)
    raw["month"] = raw["month"].astype(int)
    raw["utility_number"] = raw["utility_number"].astype("Int64")
    raw["utility_name"] = raw["utility_name"].astype(str).str.strip()
    raw["state"] = raw["state"].astype(str).str.strip()
    raw["ownership"] = raw["ownership"].astype(str).replace("nan", np.nan).str.strip()
    raw["data_status"] = raw["data_status"].astype(str).str.strip()
    raw["source_file"] = path.name
    raw["source_sheet"] = sheet
    raw["is_state_adjustment"] = raw["utility_number"].eq(0).astype(int)
    raw["is_state_total"] = raw["utility_number"].eq(88888).astype(int)
    raw["period"] = raw["year"].astype(str) + "M" + raw["month"].astype(str).str.zfill(2)

    raw["residential_price_cents_kwh"] = (
        raw["residential_revenue_thousand_dollars"] * 1000
        / raw["residential_sales_mwh"].replace(0, np.nan) / 10
    )
    raw["commercial_price_cents_kwh"] = (
        raw["commercial_revenue_thousand_dollars"] * 1000
        / raw["commercial_sales_mwh"].replace(0, np.nan) / 10
    )
    raw["industrial_price_cents_kwh"] = (
        raw["industrial_revenue_thousand_dollars"] * 1000
        / raw["industrial_sales_mwh"].replace(0, np.nan) / 10
    )
    return raw


def aggregate_eia861m(monthly: pd.DataFrame, group_cols: list[str], period_name: str) -> pd.DataFrame:
    sum_cols = [
        "residential_revenue_thousand_dollars", "residential_sales_mwh",
        "commercial_revenue_thousand_dollars", "commercial_sales_mwh",
        "industrial_revenue_thousand_dollars", "industrial_sales_mwh",
        "transportation_revenue_thousand_dollars", "transportation_sales_mwh",
        "total_revenue_thousand_dollars", "total_sales_mwh",
    ]
    mean_cols = [
        "residential_customers", "commercial_customers", "industrial_customers",
        "transportation_customers", "total_customers",
    ]
    out = (
        monthly.groupby(group_cols, as_index=False)
        .agg(
            utility_name=("utility_name", "first"),
            ownership=("ownership", "first"),
            data_status=("data_status", lambda x: ",".join(sorted(set(x.dropna().astype(str))))),
            months_observed=("month", "nunique"),
            first_month=("month", "min"),
            last_month=("month", "max"),
            **{col: (col, "sum") for col in sum_cols},
            **{col: (col, "mean") for col in mean_cols},
        )
    )
    out["period_name"] = period_name
    out["is_full_year"] = out["months_observed"].eq(12).astype(int)
    out["residential_price_cents_kwh"] = (
        out["residential_revenue_thousand_dollars"] * 1000
        / out["residential_sales_mwh"].replace(0, np.nan) / 10
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
        / out["residential_customers"].replace(0, np.nan)
        / out["months_observed"].replace(0, np.nan)
    )
    return out


def add_service_territory_exposure(df: pd.DataFrame) -> pd.DataFrame:
    panel_path = OUT_861M / "utility_state_year_panel.csv"
    if not panel_path.exists():
        return df
    panel = pd.read_csv(panel_path)
    latest = panel[panel["year"].eq(panel["year"].max())]
    exposure_cols = [
        "state", "utility_number", "cum_dc_count", "cum_dc_per_100k_customers",
        "single_state_service_area", "n_service_area_features", "service_area_states_sampled",
    ]
    exposure = latest[[c for c in exposure_cols if c in latest.columns]].drop_duplicates(["state", "utility_number"])
    return df.merge(exposure, on=["state", "utility_number"], how="left")


def process_eia861m() -> dict[str, int]:
    OUT_861M.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW.glob("sales_ult_cust_20*.xlsx"))
    frames = [read_eia861m(path) for path in files]
    monthly_all = pd.concat(frames, ignore_index=True)
    monthly_all = monthly_all.sort_values(["year", "month", "state", "utility_number"]).reset_index(drop=True)
    monthly_clean = monthly_all[
        monthly_all["utility_number"].notna()
        & ~monthly_all["utility_number"].isin([0, 88888, 99999])
        & monthly_all["residential_revenue_thousand_dollars"].fillna(0).gt(0)
        & monthly_all["residential_sales_mwh"].fillna(0).gt(0)
    ].copy()

    annual_all = aggregate_eia861m(monthly_all, ["year", "state", "utility_number"], "calendar_year")
    annual_clean = aggregate_eia861m(monthly_clean, ["year", "state", "utility_number"], "calendar_year")

    q1_clean = aggregate_eia861m(
        monthly_clean[monthly_clean["month"].isin([1, 2, 3])],
        ["year", "state", "utility_number"],
        "q1",
    )

    monthly_clean_enriched = add_service_territory_exposure(monthly_clean)
    annual_clean_enriched = add_service_territory_exposure(annual_clean)
    q1_clean_enriched = add_service_territory_exposure(q1_clean)

    monthly_all.to_csv(OUT_861M / "eia861m_monthly_sales_revenue_all_2023_2026.csv", index=False)
    monthly_clean.to_csv(OUT_861M / "eia861m_monthly_sales_revenue_utility_clean_2023_2026.csv", index=False)
    monthly_clean_enriched.to_csv(OUT_861M / "eia861m_monthly_service_territory_panel_2023_2026.csv", index=False)
    annual_all.to_csv(OUT_861M / "eia861m_annual_sales_revenue_all_2023_2026.csv", index=False)
    annual_clean.to_csv(OUT_861M / "eia861m_annual_sales_revenue_utility_clean_2023_2026.csv", index=False)
    annual_clean_enriched.to_csv(OUT_861M / "eia861m_annual_service_territory_panel_2023_2026.csv", index=False)
    q1_clean.to_csv(OUT_861M / "eia861m_q1_sales_revenue_utility_clean_2023_2026.csv", index=False)
    q1_clean_enriched.to_csv(OUT_861M / "eia861m_q1_service_territory_panel_2023_2026.csv", index=False)

    coverage = (
        monthly_all.groupby(["year", "data_status"], as_index=False)
        .agg(months=("month", lambda x: ",".join(map(str, sorted(set(x))))), rows=("year", "size"))
    )
    coverage.to_csv(OUT_861M / "eia861m_input_coverage.csv", index=False)
    return {
        "eia861m_monthly_all_rows": len(monthly_all),
        "eia861m_monthly_clean_rows": len(monthly_clean),
        "eia861m_annual_clean_rows": len(annual_clean),
    }


def pdf_text(path: Path) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def normalize_delivery_year(short_year: str) -> str:
    left, right = short_year.split("/")
    start = 2000 + int(left)
    end = 2000 + int(right)
    return f"{start}/{end}"


def make_bra_outputs() -> dict[str, int]:
    OUT_PJM.mkdir(parents=True, exist_ok=True)
    text_rows = []
    for delivery_year, path in BRA_REPORTS.items():
        text = pdf_text(path)
        expected = delivery_year.replace("20", "", 1).replace("20", "", 1)
        text_rows.append({
            "delivery_year": delivery_year,
            "source_file": str(path.relative_to(ROOT)),
            "text_chars": len(text),
            "contains_delivery_year": int(delivery_year in text or expected in text),
        })

    price_rows = []
    for delivery_year, prices in BRA_LDA_PRICES.items():
        source = str(BRA_REPORTS[delivery_year].relative_to(ROOT))
        rto_price = BRA_LDA_PRICES[delivery_year].get("RTO")
        for lda, price in prices.items():
            price_rows.append({
                "delivery_year": delivery_year,
                "auction_type": "Base Residual Auction",
                "lda": lda,
                "clearing_price_mw_day": price,
                "rto_price_mw_day": rto_price,
                "price_adder_vs_rto": price - rto_price if rto_price is not None else np.nan,
                "source_file": source,
            })

    summary_rows = []
    for delivery_year, vals in BRA_RTO_SUMMARY.items():
        row = {
            "delivery_year": delivery_year,
            "auction_type": "Base Residual Auction",
            "source_file": str(BRA_REPORTS[delivery_year].relative_to(ROOT)),
        }
        row.update(vals)
        summary_rows.append(row)

    zones = pd.DataFrame(PJM_ZONE_ROWS, columns=["zone_code", "zone_name", "state_hint"])
    zones["source_file"] = "data/raw/pjm-zones.pdf"
    zones["source_note"] = "Text legend extracted from PJM zones map PDF; use state_hint as broad reference, not a precise service polygon."
    zones.to_csv(OUT_PJM / "pjm_zone_reference.csv", index=False)
    pd.DataFrame([
        {
            "raw_file": "data/raw/pjm-zones.pdf",
            "processed_use": "zone_reference_legend",
            "machine_readable_text": 1,
            "processed_file": "data/processed/pjm/pjm_zone_reference.csv",
        },
        {
            "raw_file": "data/raw/PJM Locational Marginal Pricing Map.png",
            "processed_use": "visual_reference_only",
            "machine_readable_text": 0,
            "processed_file": "",
        },
    ]).to_csv(OUT_PJM / "pjm_map_inventory.csv", index=False)
    pd.DataFrame(price_rows).to_csv(OUT_PJM / "pjm_bra_clearing_prices_by_lda.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_PJM / "pjm_bra_rto_summary.csv", index=False)
    pd.DataFrame(text_rows).to_csv(OUT_PJM / "pjm_bra_pdf_readability_inventory.csv", index=False)

    # Backward-compatible copy for older scripts that expected this path/name.
    legacy = pd.DataFrame(price_rows).rename(columns={"lda": "zone"})
    legacy["auction_month"] = np.nan
    legacy["source_note"] = "official_pjm_bra_report_pdf_processed"
    legacy[["delivery_year", "auction_month", "zone", "clearing_price_mw_day", "source_note"]].to_csv(
        PROCESSED / "pjm_auction_clearing_prices.csv", index=False
    )

    return {
        "pjm_bra_price_rows": len(price_rows),
        "pjm_zone_reference_rows": len(zones),
    }


def make_inventory(metrics: dict[str, int]) -> None:
    rows = []
    for path in [
        OUT_861M / "eia861m_monthly_sales_revenue_all_2023_2026.csv",
        OUT_861M / "eia861m_monthly_sales_revenue_utility_clean_2023_2026.csv",
        OUT_861M / "eia861m_monthly_service_territory_panel_2023_2026.csv",
        OUT_861M / "eia861m_annual_sales_revenue_utility_clean_2023_2026.csv",
        OUT_861M / "eia861m_annual_service_territory_panel_2023_2026.csv",
        OUT_861M / "eia861m_q1_service_territory_panel_2023_2026.csv",
        OUT_PJM / "pjm_bra_clearing_prices_by_lda.csv",
        OUT_PJM / "pjm_bra_rto_summary.csv",
        OUT_PJM / "pjm_zone_reference.csv",
        OUT_PJM / "pjm_map_inventory.csv",
    ]:
        rows.append({
            "processed_file": str(path.relative_to(ROOT)),
            "exists": int(path.exists()),
            "bytes": path.stat().st_size if path.exists() else 0,
        })
    inv = pd.DataFrame(rows)
    for key, value in metrics.items():
        inv.loc[len(inv)] = {"processed_file": key, "exists": 1, "bytes": value}
    inv.to_csv(PROCESSED / "recent_retail_pjm_processing_inventory.csv", index=False)


def main() -> int:
    metrics = {}
    metrics.update(process_eia861m())
    metrics.update(make_bra_outputs())
    make_inventory(metrics)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Wrote inventory: {PROCESSED / 'recent_retail_pjm_processing_inventory.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
