"""Generate final figures and LaTeX tables."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FINAL = PROCESSED / "final"
FIG_DIR = ROOT / "figures" / "final"
TAB_DIR = ROOT / "figures" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (8.0, 4.8),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.family": "sans-serif",
})


def fig1_pjm_capacity_prices():
    bra = pd.read_csv(PROCESSED / "pjm" / "pjm_bra_clearing_prices_by_lda.csv")
    bra["delivery_year_start"] = bra["delivery_year"].str.split("/").str[0].astype(int)

    selected = ["RTO", "DOM", "BGE", "COMED", "PSEG", "PEPCO"]
    sub = bra[bra["lda"].isin(selected)].copy()
    sub = sub.sort_values(["lda", "delivery_year_start"])

    fig, ax = plt.subplots()
    colors = {
        "RTO": "#222222",
        "DOM": "#d62728",
        "BGE": "#ff7f0e",
        "COMED": "#1f77b4",
        "PSEG": "#2ca02c",
        "PEPCO": "#9467bd",
    }
    for lda, grp in sub.groupby("lda"):
        ax.plot(
            grp["delivery_year_start"],
            grp["clearing_price_mw_day"],
            marker="o",
            label=lda,
            color=colors.get(lda, None),
            linewidth=2 if lda in ("RTO", "DOM") else 1.5,
        )
    ax.set_xlabel("Delivery year (auction cleared previous summer)")
    ax.set_ylabel("Capacity clearing price ($ per MW day)")
    ax.set_title("PJM Base Residual Auction clearing prices, selected zones")
    ax.axvline(2025, color="grey", linestyle="-", linewidth=1, alpha=0.6)
    ax.text(2025.05, ax.get_ylim()[1] * 0.9, "2025/2026 DY", fontsize=8, color="grey")
    ax.legend(ncol=3, loc="upper left", frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "fig1_pjm_bra_capacity_prices.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  -> {out}")


def fig2_pjm_vs_nonpjm_index():
    idx = pd.read_csv(FINAL / "monthly_pjm_price_index.csv")
    sub = idx[idx["group_label"] == "by_in_pjm"].copy()
    sub["date"] = pd.to_datetime(sub["year_month"], format="%Y-%m")

    fig, ax = plt.subplots()
    for in_pjm, label, color in [(1, "PJM utilities", "#1f77b4"), (0, "Non PJM utilities", "#d62728")]:
        s = sub[sub["in_pjm"] == in_pjm].sort_values("date")
        ax.plot(s["date"], s["index_jan_2023_100"], label=label, linewidth=2, color=color)
    shock = pd.Timestamp("2025-06-01")
    ax.axvline(shock, color="grey", linestyle="-", linewidth=1, alpha=0.7)
    ax.text(shock + pd.Timedelta(days=10), ax.get_ylim()[1] - 2,
            "PJM 2025/2026\ndelivery year begins",
            fontsize=8, color="grey", verticalalignment="top")
    ax.set_ylabel("Residential price index (Jan 2023 = 100)")
    ax.set_xlabel("Month")
    ax.set_title("Customer weighted residential price, PJM and non PJM utilities")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "fig2_pjm_vs_nonpjm_price_index.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  -> {out}")


def fig3_pjm_dc_group_index():
    hl = pd.read_csv(FINAL / "high_low_dc_comparison.csv")
    sub = hl[hl["comparison"] == "Q1_2026_vs_Q1_2025"].copy()

    order = [
        ("PJM, high DC (10 plus)", 1, "high_dc_10plus", "#b91c1c"),
        ("PJM, some DC (1 to 9)", 1, "some_dc_1to9", "#dc2626"),
        ("PJM, zero DC", 1, "zero_dc", "#f87171"),
        ("Non PJM, high DC (10 plus)", 0, "high_dc_10plus", "#1d4ed8"),
        ("Non PJM, some DC (1 to 9)", 0, "some_dc_1to9", "#2563eb"),
        ("Non PJM, zero DC", 0, "zero_dc", "#60a5fa"),
    ]
    rows = []
    for label, pjm, dc, color in order:
        row = sub[(sub["in_pjm"] == pjm) & (sub["dc_group"] == dc)]
        if row.empty:
            continue
        rows.append({
            "label": label, "color": color,
            "cw": row["customer_weighted_change_pct"].iloc[0],
            "mean": row["mean_price_change_pct"].iloc[0],
            "n": int(row["n_utilities"].iloc[0]),
        })

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ys = list(range(len(rows)))
    cws = [r["cw"] for r in rows]
    colors = [r["color"] for r in rows]
    labels = [r["label"] for r in rows]
    bars = ax.barh(ys, cws, color=colors)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(cws) * 1.22)
    ax.set_xlabel("Customer weighted percent change, Q1 2026 versus Q1 2025")
    ax.set_title("Residential price change by PJM membership and data center exposure")
    for bar, val, r in zip(bars, cws, rows):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%  (n={r['n']})", va="center", fontsize=9)
    ax.axvline(0, color="black", linewidth=0.6)
    fig.tight_layout()
    out = FIG_DIR / "fig3_pjm_dc_exposure_change.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  -> {out}")


def fig3b_pjm_dc_group_monthly_index():
    """Plot the monthly PJM price index by DC-exposure group."""
    panel = pd.read_csv(FINAL / "utility_area_month_panel_2023_2026.csv")
    panel = panel[(panel["in_pjm"] == 1) & panel["residential_price_cents_kwh"].notna()].copy()
    panel["date"] = pd.to_datetime(panel["date"])

    def assign_group(r):
        if r["high_dc_utility"] == 1:
            return "PJM, high DC (10 plus)"
        if r["some_dc_utility"] == 1:
            return "PJM, some DC (1 to 9)"
        return "PJM, zero DC"

    panel["group"] = panel.apply(assign_group, axis=1)

    panel["weighted_price"] = panel["residential_price_cents_kwh"] * panel["residential_customers"].clip(lower=1)
    monthly = (
        panel.groupby(["group", "date"], as_index=False)
        .agg(weighted_price=("weighted_price", "sum"), customers=("residential_customers", "sum"))
    )
    monthly["cw_price"] = monthly["weighted_price"] / monthly["customers"]
    base = monthly[monthly["date"] == pd.Timestamp("2023-01-01")].set_index("group")["cw_price"]
    monthly["index"] = monthly.apply(
        lambda r: 100.0 * r["cw_price"] / base.loc[r["group"]], axis=1
    )

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    colors = {
        "PJM, high DC (10 plus)": "#b91c1c",
        "PJM, some DC (1 to 9)": "#dc2626",
        "PJM, zero DC": "#f87171",
    }
    for grp in ["PJM, high DC (10 plus)", "PJM, some DC (1 to 9)", "PJM, zero DC"]:
        sub = monthly[monthly["group"] == grp].sort_values("date")
        ax.plot(sub["date"], sub["index"], label=grp, color=colors[grp], linewidth=2)
    ax.axvline(pd.Timestamp("2025-06-01"), color="grey", linestyle="-", linewidth=1)
    ax.text(pd.Timestamp("2025-06-10"), ax.get_ylim()[1] * 0.97,
            "PJM 2025/2026\ndelivery year begins",
            fontsize=8, color="grey", va="top")
    ax.set_xlabel("Month")
    ax.set_ylabel("Residential price index (Jan 2023 = 100)")
    ax.set_title("Customer weighted residential price index, PJM utilities by data center exposure")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / "fig3b_pjm_dc_group_monthly_index.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  -> {out}")


def fig4_dc_counts_by_utility():
    panel = pd.read_csv(FINAL / "utility_area_month_panel_2023_2026.csv")
    latest = panel[(panel["year"] == 2024) & (panel["month"] == 12)].copy()
    latest = latest[latest["in_pjm"] == 1]
    latest = latest.sort_values("cum_dc_count", ascending=False).drop_duplicates("unit_id")
    top = latest.head(8)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    labels = top["utility_name"].str.slice(0, 32) + " (" + top["state"] + ")"
    ax.barh(labels[::-1], top["cum_dc_count"][::-1], color="#1f77b4")
    ax.set_xlabel("Cumulative data centers in service area")
    ax.set_title("Data center counts by PJM utility service area (top 8)")
    for i, (val, name) in enumerate(zip(top["cum_dc_count"][::-1], labels[::-1])):
        ax.text(val + 1, i, str(int(val)), va="center", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "fig4_dc_counts_by_pjm_utility.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  -> {out}")


def fig5_pjm_zone_price_change():
    sz = pd.read_csv(FINAL / "state_zone_price_change_summary.csv")
    zone_rows = sz[sz["group_kind"] == "by_pjm_zone"].copy()
    if "price_change_pct" not in zone_rows.columns:
        return
    zone_rows = zone_rows.dropna(subset=["pjm_zone", "price_change_pct"]).copy()
    zone_rows = zone_rows.sort_values("price_change_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#d62728" if z == "DOM" else "#1f77b4" for z in zone_rows["pjm_zone"]]
    bars = ax.barh(zone_rows["pjm_zone"], zone_rows["price_change_pct"], color=colors)
    ax.set_xlabel("Q1 2026 vs Q1 2025 residential price change (%)")
    ax.set_title("PJM zone customer weighted residential price change")
    for bar, val in zip(bars, zone_rows["price_change_pct"]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.6)
    fig.tight_layout()
    out = FIG_DIR / "fig5_pjm_zone_price_change.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  -> {out}")


def _esc(s):
    if pd.isna(s):
        return ""
    return str(s).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def table1_data_coverage():
    panel = pd.read_csv(FINAL / "utility_area_month_panel_2023_2026.csv")
    by_year = panel.groupby("year").agg(
        utilities=("unit_id", "nunique"),
        states=("state", "nunique"),
        rows=("unit_id", "size"),
    ).reset_index()
    by_year["status"] = ["Final", "Final", "Preliminary", "Preliminary"]
    by_year["months"] = ["Jan-Dec", "Jan-Dec", "Jan-Dec", "Jan-Mar"]

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Coverage of the EIA-861M monthly utility-service-area panel, 2023--2026.}",
        r"\label{tab:coverage}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Year & Status & Months & Utility-states & Observations \\",
        r"\midrule",
    ]
    for _, r in by_year.iterrows():
        lines.append(
            f"{int(r['year'])} & {r['status']} & {r['months']} & {int(r['utilities'])} & {int(r['rows']):,} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out = TAB_DIR / "tab1_data_coverage.tex"
    out.write_text("\n".join(lines))
    print(f"  -> {out}")


def table2_descriptives():
    panel = pd.read_csv(FINAL / "utility_area_month_panel_2023_2026.csv")
    last = panel[(panel["year"] == 2024) & (panel["month"] == 12)].copy()
    last["dc_group"] = np.where(
        last["high_dc_utility"] == 1, "high_dc_10plus",
        np.where(last["some_dc_utility"] == 1, "some_dc_1to9", "zero_dc"),
    )

    rows = []
    for in_pjm in [1, 0]:
        for dc_group in ["high_dc_10plus", "some_dc_1to9", "zero_dc"]:
            sub = last[(last["in_pjm"] == in_pjm) & (last["dc_group"] == dc_group)]
            if sub.empty:
                continue
            rows.append({
                "in_pjm": in_pjm,
                "dc_group": dc_group,
                "n_utilities": int(sub["unit_id"].nunique()),
                "mean_price": float(sub["residential_price_cents_kwh"].mean()),
                "mean_customers": float(sub["residential_customers"].mean()),
                "mean_dcs": float(sub["cum_dc_count"].mean()),
            })

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Descriptive statistics by PJM membership and data-center exposure (December 2024 snapshot).}",
        r"\label{tab:descriptives}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"PJM & DC exposure & $N$ & Mean price (¢/kWh) & Mean customers & Mean DCs \\",
        r"\midrule",
    ]
    label_map = {
        "high_dc_10plus": "High (10+ DCs)",
        "some_dc_1to9": "Some (1--9)",
        "zero_dc": "Zero",
    }
    for r in rows:
        pjm = "PJM" if r["in_pjm"] == 1 else "Non-PJM"
        lines.append(
            f"{pjm} & {label_map[r['dc_group']]} & {r['n_utilities']} & "
            f"{r['mean_price']:.2f} & {r['mean_customers']:,.0f} & {r['mean_dcs']:.1f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out = TAB_DIR / "tab2_descriptives.tex"
    out.write_text("\n".join(lines))
    print(f"  -> {out}")


def _fmt_pct(x):
    return f"{x*100:+.2f}\\%" if pd.notna(x) else ""


def _fmt_se(x):
    return f"({x*100:.2f})" if pd.notna(x) else ""


def _fmt_p(x):
    if pd.isna(x):
        return ""
    if x < 0.001:
        return r"$p<0.001$"
    return f"$p={x:.3f}$"


def table3_main_regs():
    results = pd.read_csv(FINAL / "main_regression_results.csv")

    historical = results[results["spec"].str.startswith("1_")]
    monthly_pass = results[results["spec"].str.startswith("2_")]
    het_did = results[results["spec"].str.startswith("3_")]
    triple_did = results[results["spec"] == "3b_triple_did"]

    rows = []

    if not historical.empty:
        r1 = historical[historical["treatment"] == "cum_dc_per_100k_customers"].iloc[0]
        rows.append(("Historical annual: DCs/100k cust.",
                     "2010--2024 utility-state",
                     r1["coefficient"], r1.get("std_error", np.nan), r1["p_value"], int(r1["n_obs"])))

    if not monthly_pass.empty:
        r2 = monthly_pass[(monthly_pass["sample"] == "pjm_utilities_only_monthly") &
                           (monthly_pass["treatment"] == "log_pjm_capacity_price")].iloc[0]
        rows.append(("Monthly pass-through: log capacity",
                     "PJM utilities 2023--2026",
                     r2["coefficient"], r2["std_error"], r2["p_value"], int(r2["n_obs"])))

    if not het_did.empty:
        r3a = het_did[het_did["spec"] == "3_high_dc_heterogeneity_did_pjm"].iloc[0]
        rows.append(("Within-PJM: High-DC \\(\\times\\) post",
                     "PJM utilities 2023--2026",
                     r3a["coefficient"], r3a["std_error"], r3a["p_value"], int(r3a["n_obs"])))
        r3b = het_did[het_did["spec"] == "3_any_dc_heterogeneity_did_pjm"].iloc[0]
        rows.append(("Within-PJM: Any-DC \\(\\times\\) post",
                     "PJM utilities 2023--2026",
                     r3b["coefficient"], r3b["std_error"], r3b["p_value"], int(r3b["n_obs"])))

    if not triple_did.empty:
        trip = triple_did[triple_did["sample"] == "triple_did_high_dc"]
        for treat, label in [
            ("in_pjm_x_post", "Broad uplift: PJM \\(\\times\\) post"),
            ("in_pjm_x_high_dc_x_post", "Triple-DiD: PJM \\(\\times\\) hi-DC \\(\\times\\) post"),
        ]:
            r = trip[trip["treatment"] == treat]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append((label,
                         "Full panel 2023--2026",
                         r["coefficient"], r["std_error"], r["p_value"], int(r["n_obs"])))

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Main regression results. Outcome is log residential price. Standard errors clustered at utility-state; coefficients reported in percentage points with SE in parentheses. All monthly specifications include utility-state and year-month fixed effects; the historical specification uses utility-state and state-year fixed effects.}",
        r"\label{tab:mainregs}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{6.1cm} >{\raggedright\arraybackslash}p{4.2cm} c c c}",
        r"\toprule",
        r"Specification & Sample & Estimate & $p$-value & $N$ \\",
        r"\midrule",
    ]
    for label, sample, coef, se, p, n in rows:
        if pd.notna(se):
            coef_str = f"{coef*100:+.2f}\\,({se*100:.2f})"
        else:
            coef_str = f"{coef*100:+.2f}\\,(---)"
        p_str = _fmt_p(p) if pd.notna(p) else "---"
        lines.append(f"{label} & {sample} & {coef_str} & {p_str} & {n:,} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out = TAB_DIR / "tab3_main_regressions.tex"
    out.write_text("\n".join(lines))
    print(f"  -> {out}")


def table4_robustness_with_controls():
    results = pd.read_csv(FINAL / "main_regression_results.csv")
    rows_to_keep = results[results["spec"].isin([
        "5_triple_did_with_weather_controls",
    ])].copy()
    if rows_to_keep.empty:
        print("  table4 skipped: no robustness rows")
        return

    label_map = {
        "in_pjm_x_post": "PJM \\(\\times\\) post-shock",
        "high_dc_x_post": "High-DC \\(\\times\\) post-shock",
        "in_pjm_x_high_dc_x_post": "PJM \\(\\times\\) high-DC \\(\\times\\) post-shock",
        "hdd_k": "HDD (thousands)",
        "cdd_k": "CDD (thousands)",
    }

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Triple-difference design with weather controls. Outcome is log residential price. Sample is utility-states with NOAA HDD/CDD coverage. Unit and year-month fixed effects; SEs clustered by utility-state. Coefficients in percentage points (HDD/CDD reported per 1{,}000 degree-days).}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{p{7cm} c c c c}",
        r"\toprule",
        r"Treatment & Estimate & SE & $p$-value & $N$ \\",
        r"\midrule",
    ]
    for _, r in rows_to_keep.iterrows():
        treat = r["treatment"]
        if treat not in label_map:
            continue
        coef = r["coefficient"] * 100
        se = r["std_error"] * 100 if pd.notna(r["std_error"]) else np.nan
        p = r["p_value"]
        lines.append(
            f"{label_map[treat]} & {coef:+.2f}\\% & {se:.2f} & {_fmt_p(p)} & {int(r['n_obs']):,} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out = TAB_DIR / "tab4_robustness_with_controls.tex"
    out.write_text("\n".join(lines))
    print(f"  -> {out}")


def fig5_event_study():
    """Plot monthly PJM-vs-non-PJM coefficients relative to May 2025."""
    ev = pd.read_csv(FINAL / "event_study_monthly_coefficients.csv")
    ev["dt"] = pd.to_datetime(
        ev["year_month"].str[:4] + "-" + ev["year_month"].str[5:] + "-01"
    )
    ev = ev.sort_values("dt").reset_index(drop=True)
    ev["coef_pp"] = ev["coefficient"] * 100
    ev["ci_low"] = (ev["coefficient"] - 1.96 * ev["std_error"]) * 100
    ev["ci_high"] = (ev["coefficient"] + 1.96 * ev["std_error"]) * 100

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.errorbar(
        ev["dt"], ev["coef_pp"],
        yerr=[ev["coef_pp"] - ev["ci_low"], ev["ci_high"] - ev["coef_pp"]],
        fmt="o", color="#1f77b4", ecolor="lightgray",
        elinewidth=1.5, capsize=2, markersize=4, label="PJM versus non PJM"
    )
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(pd.Timestamp("2025-06-01"), color="gray", linestyle="-", lw=1.0)
    _ymin, _ymax = ax.get_ylim()
    ax.text(pd.Timestamp("2025-06-15"), _ymin + (_ymax - _ymin) * 0.06,
            "PJM 2025/26\ndelivery year begins", fontsize=8, color="gray",
            va="bottom", ha="left")
    ax.set_ylabel("PJM versus non PJM log price differential (pp, relative to May 2025)")
    ax.set_xlabel("Month")
    ax.set_title("Month by month DiD: PJM versus non PJM residential log price (rel. to May 2025)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "fig5_event_study.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


def fig6_rf_importance():
    """Random forest feature importance horizontal bar chart."""
    path = FINAL / "ml_rf_importance.csv"
    if not path.exists():
        print(f"  fig6 skipped: {path} not found")
        return
    rf = pd.read_csv(path).sort_values("importance", ascending=True)
    pretty = {
        "residential_customers": "Utility size (residential customers)",
        "log_pjm_capacity_price": "PJM capacity price (log)",
        "hdd": "Heating degree days",
        "cdd": "Cooling degree days",
        "cum_dc_per_100k_customers": "Data centers / 100k customers",
        "cum_dc_count": "Data centers (count)",
        "log_henry_hub": "Henry Hub gas price (log)",
        "log_residential_sales": "Residential sales (log)",
        "high_dc_utility": "High DC indicator",
        "in_pjm": "PJM member indicator",
        "is_pjm_state": "PJM state indicator",
        "pjm_shock_active": "Post shock indicator",
    }
    labels = [pretty.get(f, f) for f in rf["feature"]]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    bars = ax.barh(labels, rf["importance"], color="#1f77b4")
    for bar, v in zip(bars, rf["importance"]):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Random forest impurity based importance")
    ax.set_title("Variable importance for residual monthly residential price\n(500 trees, "
                 "outcome and features residualized for utility and month FE)")
    ax.set_xlim(0, max(rf["importance"]) * 1.18)
    plt.tight_layout()
    out = FIG_DIR / "fig6_rf_importance.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


def fig7_parallel_trends_within_pjm():
    """Plot parallel trends for the three PJM/DC groups."""
    panel = pd.read_csv(FINAL / "utility_area_month_panel_2023_2026.csv")
    panel = panel[
        panel["residential_price_cents_kwh"].notna()
        & panel["residential_customers"].notna()
        & (panel["residential_customers"] >= 1000)
    ].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["group"] = np.select(
        [
            (panel["in_pjm"] == 1) & (panel["high_dc_utility"] == 1),
            (panel["in_pjm"] == 1) & (panel["high_dc_utility"] == 0),
        ],
        ["PJM, high DC (10 plus DCs)", "PJM, low or zero DC"],
        default="Non PJM",
    )
    panel["weighted_price"] = panel["residential_price_cents_kwh"] * panel["residential_customers"].clip(lower=1)
    monthly = (
        panel.groupby(["group", "date"], as_index=False)
        .agg(weighted_price=("weighted_price", "sum"), customers=("residential_customers", "sum"))
    )
    monthly["price_cw"] = monthly["weighted_price"] / monthly["customers"]
    base = monthly[monthly["date"] == pd.Timestamp("2025-05-01")].set_index("group")["price_cw"]
    monthly["idx"] = monthly.apply(lambda r: 100.0 * r["price_cw"] / base.loc[r["group"]], axis=1)
    monthly.to_csv(FINAL / "parallel_trends_three_groups.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    styles = {
        "Non PJM": {"color": "#2ca02c", "marker": "^", "linestyle": "-"},
        "PJM, high DC (10 plus DCs)": {"color": "#c00000", "marker": "o", "linestyle": "-"},
        "PJM, low or zero DC": {"color": "#1f77b4", "marker": "s", "linestyle": "-"},
    }
    for group in ["Non PJM", "PJM, high DC (10 plus DCs)", "PJM, low or zero DC"]:
        sub = monthly[monthly["group"] == group].sort_values("date")
        ax.plot(sub["date"], sub["idx"], label=group, linewidth=2, markersize=4, **styles[group])
    ax.axhline(100, color="gray", linewidth=0.8, alpha=0.6)
    ax.axvline(pd.Timestamp("2025-06-01"), color="black", linestyle="-", linewidth=1.3)
    ax.text(pd.Timestamp("2025-06-05"), 88, "June 2025\nPJM 2025/26 DY", fontsize=9)
    ax.set_title("Customer weighted residential price: high DC PJM, low or zero DC PJM, and non PJM")
    ax.set_xlabel("Month")
    ax.set_ylabel("Residential price index (May 2025 = 100)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = FIG_DIR / "fig7_parallel_trends_within_pjm.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  -> {out}")


def table7_single_state_robustness():
    """Tabulate the single-state utility-area robustness check."""
    mr = pd.read_csv(FINAL / "main_regression_results.csv")
    ss = mr[mr["spec"] == "4_single_state_triple_did"].copy()
    if ss.empty:
        print("  table7 skipped: no single-state rows in main_regression_results.csv")
        return

    label_map = {
        "in_pjm_x_post": "PJM \\(\\times\\) post (broad uplift)",
        "high_dc_x_post": "High-DC \\(\\times\\) post (outside-PJM)",
        "in_pjm_x_high_dc_x_post": "PJM \\(\\times\\) high-DC \\(\\times\\) post (triple-DiD)",
    }

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Robustness: triple-difference restricted to single-state utility service areas. Sample is utility-states whose HIFLD service-territory polygon does not cross state lines (48 utility-states, 2 in PJM, 0 high-DC PJM). Unit and year-month fixed effects; SEs clustered by utility-state. The triple-DiD interaction is not identified on this subsample because no high-DC utility (Dominion VA, ComEd IL, PSE\&G NJ) has a single-state service-territory polygon; we report it for completeness.}",
        r"\label{tab:single_state}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{l c c c r}",
        r"\toprule",
        r"Treatment & Estimate (pp) & SE (pp) & $p$-value & $N$ \\",
        r"\midrule",
    ]
    for _, r in ss.iterrows():
        label = label_map.get(r["treatment"], r["treatment"])
        coef = r["coefficient"] * 100
        se = r["std_error"] * 100 if pd.notna(r["std_error"]) else np.nan
        p_str = f"$p={r['p_value']:.3f}$" if pd.notna(r["p_value"]) else "---"
        se_str = f"{se:.2f}" if pd.notna(se) else "---"
        lines.append(
            f"{label} & {coef:+.2f} & {se_str} & {p_str} & {int(r['n_obs']):,} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out = TAB_DIR / "tab7_single_state_robustness.tex"
    out.write_text("\n".join(lines))
    print(f"  -> {out}")


def table6_ml_robustness():
    """Single table combining Lasso post-selection and wild cluster bootstrap results."""
    lasso_p = FINAL / "ml_lasso_selection.csv"
    wcb_p = FINAL / "ml_wild_bootstrap.csv"
    if not (lasso_p.exists() and wcb_p.exists()):
        print("  table6 skipped: ML outputs missing")
        return
    lasso = pd.read_csv(lasso_p)
    wcb = pd.read_csv(wcb_p)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Machine-learning robustness specifications. Panel A: Lasso post-selection on the historical 2010--2024 utility-state annual panel; Belloni-Chernozhukov-Hansen (2014) double-selection with cluster-robust standard errors at the utility-state level. Panel B: wild cluster bootstrap of the triple-difference coefficients with $999$ Rademacher replications at the utility-state cluster (Cameron-Gelbach-Miller, 2008). Bootstrap $p$-values reported alongside the analytical cluster-robust $p$-values.}",
        r"\label{tab:ml_robustness}",
        r"\small",
        r"",
        r"\textit{Panel A: Lasso post-selection on historical 2010--2024 panel}",
        r"\vspace{2pt}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{l c c c r c}",
        r"\toprule",
        r"Specification & Estimate (pp) & SE (pp) & $p$ & $N$ & Selected / Cand. \\",
        r"\midrule",
    ]
    r = lasso.iloc[0]
    lines.append(
        f"Double-selection Lasso & {r['coefficient']*100:+.3f} & {r['std_error']*100:.3f} & "
        f"$p={r['p_value']:.3f}$ & {int(r['n_obs']):,} & "
        f"{int(r['n_selected_controls'])} / {int(r['n_candidate_controls'])} \\\\"
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{10pt}",
        r"\textit{Panel B: Wild cluster bootstrap on triple-difference (999 Rademacher replications)}",
        r"\vspace{2pt}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l c c c c r}",
        r"\toprule",
        r"Coefficient & Estimate (pp) & SE (pp) & Analytical $p$ & Bootstrap $p$ & $N$ \\",
        r"\midrule",
    ]
    label_map = {
        "PJM x post (broad uplift)": "PJM $\\times$ post (broad uplift)",
        "High-DC x post (outside-PJM component)": "High-DC $\\times$ post (outside-PJM)",
        "PJM x high-DC x post (triple-DiD)": "PJM $\\times$ high-DC $\\times$ post (triple-DiD)",
    }
    for _, r in wcb.iterrows():
        label = label_map.get(r["label"], r["label"])
        lines.append(
            f"{label} & {r['coefficient']*100:+.2f} & {r['std_error_analytical']*100:.2f} & "
            f"{r['p_value_analytical']:.3f} & {r['p_value_bootstrap']:.3f} & "
            f"{int(r['n_obs']):,} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out = TAB_DIR / "tab6_ml_robustness.tex"
    out.write_text("\n".join(lines))
    print(f"  -> {out}")


def table8_exposure_robustness():
    threshold_path = FINAL / "threshold_sensitivity.csv"
    continuous_path = FINAL / "continuous_exposure_spec.json"
    if not (threshold_path.exists() and continuous_path.exists()):
        print("  table8 skipped: exposure robustness outputs missing")
        return
    threshold = pd.read_csv(threshold_path)
    continuous = json.loads(continuous_path.read_text())

    def p_fmt(p):
        if pd.isna(p):
            return "---"
        if p < 0.001:
            return r"$<$0.001"
        return f"{p:.3f}"

    n_obs = int(threshold["N"].dropna().iloc[0]) if "N" in threshold and threshold["N"].notna().any() else int(continuous.get("N", 0))
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Triple-difference robustness to the high-DC threshold and to continuous exposure. Outcome is log residential price. Specification follows Equation~\ref{{eq:tripledid}} with utility-state and year-month fixed effects; SEs clustered by utility-state. ``Triple-diff'' is the PJM~$\times$~HighDC~$\times$~post coefficient (binary specifications) or the PJM~$\times$~DC-per-100k~$\times$~post coefficient (continuous). Sample matches the headline specification where available, $N={n_obs:,}$ utility-state-month observations.}}",
        r"\label{tab:exposure_robustness}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{p{5.2cm} c c c c c}",
        r"\toprule",
        r"Specification & High-DC PJM units & Triple-diff & SE & $p$-value & PJM$\times$post \\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Panel A: Binary high-DC threshold sensitivity}} \\",
        r"\midrule",
    ]
    for _, r in threshold.iterrows():
        label = f"Threshold $\\geq {int(r['threshold'])}$ DCs"
        if int(r["threshold"]) == 10:
            label += " (headline)"
        lines.append(
            f"{label} & {int(r['n_high_dc_pjm_units'])} & "
            f"{r['beta_triple_pp']:+.2f}\\% & {r['se_triple_pp']:.2f} & "
            f"{p_fmt(r['p_triple'])} & {r['beta_pjm_post_pp']:+.2f}\\% \\\\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Panel B: Continuous exposure (DCs per 100k residential customers)}} \\",
        r"\midrule",
        (
            f"Per additional DC per 100k cust. & --- & "
            f"{continuous.get('beta_triple_per_dc_per_100k_pp', np.nan):+.2f} pp & "
            f"{continuous.get('se_triple_per_dc_per_100k_pp', np.nan):.2f} & "
            f"{p_fmt(continuous.get('p_triple', np.nan))} & "
            f"{continuous.get('beta_pjm_post_pp', np.nan):+.2f}\\% \\\\"
        ),
    ]
    for item in continuous.get("implied_effects", []):
        raw_name = str(item.get("utility_name", "Utility"))
        pretty = {
            "Virginia Electric & Power Co": "Dominion VA",
            "Commonwealth Edison Co": "ComEd IL",
            "Public Service Elec & Gas Co": "PSE\\&G NJ",
        }
        name = pretty.get(raw_name, _esc(raw_name))
        exposure = item.get("dc_per_100k", np.nan)
        effect = item.get("implied_effect_pp", np.nan)
        se = item.get("implied_se_pp", np.nan)
        lines.append(
            f"\\quad Implied: {name} ($\\approx${exposure:.1f} DC/100k) & --- & "
            f"{effect:+.2f} pp & {se:.2f} & --- & --- \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out = TAB_DIR / "tab8_exposure_robustness.tex"
    out.write_text("\n".join(lines))
    print(f"  -> {out}")


def table5_pjm_zone_price_change():
    sz = pd.read_csv(FINAL / "state_zone_price_change_summary.csv")
    zone_rows = sz[(sz["group_kind"] == "by_pjm_zone") & sz["pjm_zone"].notna()].copy()
    if "price_change_pct" not in zone_rows.columns:
        return
    zone_rows = zone_rows.sort_values("price_change_pct", ascending=False)
    panel = pd.read_csv(FINAL / "utility_area_month_panel_2023_2026.csv")
    dc_by_zone = panel.dropna(subset=["pjm_zone"]).groupby("pjm_zone").agg(
        cum_dc=("cum_dc_count", "max"),
    ).reset_index()
    zone_rows = zone_rows.merge(dc_by_zone, on="pjm_zone", how="left")

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Customer-weighted residential price change by PJM zone, Q1 2026 vs Q1 2025. 2025 and 2026 data are preliminary.}",
        r"\label{tab:pjm_zone_change}",
        r"\begin{tabular}{l c c c c c}",
        r"\toprule",
        r"Zone & Utilities & Customers (mn) & Q1 2025 ¢/kWh & Q1 2026 ¢/kWh & \% change \\",
        r"\midrule",
    ]
    for _, r in zone_rows.iterrows():
        cust_mn = (r.get("Q1_2025_customers", np.nan) or 0) / 1e6
        zone_label = str(r['pjm_zone']).replace("&", r"\&")
        lines.append(
            f"{zone_label} & {int(r['Q1_2025'])} & {cust_mn:.2f} & "
            f"{r['Q1_2025_price']:.2f} & {r['Q1_2026_price']:.2f} & "
            f"{r['price_change_pct']:+.1f}\\% \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out = TAB_DIR / "tab5_pjm_zone_price_change.tex"
    out.write_text("\n".join(lines))
    print(f"  -> {out}")


def main():
    print("Generating figures...")
    fig1_pjm_capacity_prices()
    fig2_pjm_vs_nonpjm_index()
    fig3_pjm_dc_group_index()
    fig4_dc_counts_by_utility()
    fig5_event_study()
    fig6_rf_importance()
    fig7_parallel_trends_within_pjm()
    print()
    print("Generating LaTeX tables...")
    table1_data_coverage()
    table2_descriptives()
    table3_main_regs()
    table4_robustness_with_controls()
    table5_pjm_zone_price_change()
    table6_ml_robustness()
    table7_single_state_robustness()
    table8_exposure_robustness()
    print("Done.")


if __name__ == "__main__":
    main()
