"""Run ML robustness specifications."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FINAL = PROCESSED / "final"
FINAL.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)


def demean_two_way(
    df: pd.DataFrame, col: str, entity: str = "unit_id", time: str = "year"
) -> pd.Series:
    overall = df[col].mean()
    ent = df.groupby(entity)[col].transform("mean")
    tim = df.groupby(time)[col].transform("mean")
    return df[col] - ent - tim + overall


def fit_clustered_ols(y: pd.Series, x: pd.DataFrame, clusters: pd.Series):
    x = sm.add_constant(x, has_constant="add")
    model = sm.OLS(y, x, missing="drop")
    return model.fit(cov_type="cluster", cov_kwds={"groups": clusters})


def run_lasso_post_selection() -> pd.DataFrame:
    """Run double-selection Lasso on the annual panel."""
    annual = pd.read_csv(
        PROCESSED / "utility_service_territory" / "utility_state_year_panel.csv"
    )
    annual["unit_id"] = annual["state"] + "_" + annual["utility_number"].astype(str)
    work = annual.dropna(
        subset=["log_residential_price", "cum_dc_per_100k_customers"]
    ).copy()
    work = work[work["residential_customers"] >= 1000]

    control_candidates = [
        "log_residential_sales",
        "commercial_customers",
        "industrial_sales_mwh",
        "residential_customers",
    ]
    controls = [c for c in control_candidates if c in work.columns]
    work = work.dropna(subset=controls)

    work["log_industrial_sales_mwh"] = np.log1p(work["industrial_sales_mwh"].clip(lower=0))
    work["log_commercial_customers"] = np.log1p(work["commercial_customers"].clip(lower=0))
    work["log_residential_customers"] = np.log1p(
        work["residential_customers"].clip(lower=0)
    )
    extended = controls + [
        "log_industrial_sales_mwh",
        "log_commercial_customers",
        "log_residential_customers",
    ]
    extended = [c for c in extended if c in work.columns]

    y_dm = demean_two_way(work, "log_residential_price").to_numpy()
    d_dm = demean_two_way(work, "cum_dc_per_100k_customers").to_numpy()
    X_dm = np.column_stack([demean_two_way(work, c).to_numpy() for c in extended])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dm)

    lasso_y = LassoCV(cv=5, random_state=42, max_iter=50_000)
    lasso_y.fit(X_scaled, y_dm)
    sel_y = np.where(np.abs(lasso_y.coef_) > 1e-6)[0]

    lasso_d = LassoCV(cv=5, random_state=42, max_iter=50_000)
    lasso_d.fit(X_scaled, d_dm)
    sel_d = np.where(np.abs(lasso_d.coef_) > 1e-6)[0]

    selected_idx = sorted(set(sel_y.tolist()) | set(sel_d.tolist()))
    selected = [extended[i] for i in selected_idx]

    work["dm_treat"] = demean_two_way(work, "cum_dc_per_100k_customers")
    for c in selected:
        work[f"dm_{c}"] = demean_two_way(work, c)
    x_final = work[["dm_treat"] + [f"dm_{c}" for c in selected]].rename(
        columns={"dm_treat": "cum_dc_per_100k_customers",
                 **{f"dm_{c}": c for c in selected}}
    )
    y_series = pd.Series(y_dm, index=work.index)
    res = fit_clustered_ols(y_series, x_final, work["unit_id"])

    beta = res.params.get("cum_dc_per_100k_customers", np.nan)
    se = res.bse.get("cum_dc_per_100k_customers", np.nan)
    pval = res.pvalues.get("cum_dc_per_100k_customers", np.nan)

    rows = [{
        "spec": "lasso_double_selection_BCH2014",
        "sample": "historical_annual_2010_2024_polygon_assigned",
        "outcome": "log_residential_price",
        "treatment": "cum_dc_per_100k_customers",
        "coefficient": float(beta),
        "std_error": float(se),
        "p_value": float(pval),
        "n_obs": int(len(work)),
        "lasso_alpha_outcome": float(lasso_y.alpha_),
        "lasso_alpha_treatment": float(lasso_d.alpha_),
        "n_candidate_controls": len(extended),
        "n_selected_controls": len(selected),
        "selected_controls": "; ".join(selected) if selected else "(none)",
    }]
    out = pd.DataFrame(rows)
    out.to_csv(FINAL / "ml_lasso_selection.csv", index=False)
    print(f"  Lasso post-selection: β={beta:.4f} ({beta*100:.2f} pp), p={pval:.3f}")
    print(f"    Selected {len(selected)} of {len(extended)} candidates")
    return out


def run_rf_variable_importance() -> pd.DataFrame:
    """Rank monthly covariates with a residualized random forest."""
    month = pd.read_csv(FINAL / "utility_area_month_panel_2023_2026.csv")
    month["year_month"] = (
        month["year"].astype(int).astype(str) + "_" +
        month["month"].astype(int).astype(str).str.zfill(2)
    )

    work = month.dropna(
        subset=["log_residential_price", "unit_id", "year_month"]
    ).copy()
    work = work[work["residential_customers"] >= 1000]

    feature_candidates = [
        "cum_dc_count",
        "cum_dc_per_100k_customers",
        "high_dc_utility",
        "in_pjm",
        "is_pjm_state",
        "pjm_shock_active",
        "log_pjm_capacity_price",
        "hdd",
        "cdd",
        "log_henry_hub",
        "log_residential_sales",
        "residential_customers",
    ]
    features = [c for c in feature_candidates if c in work.columns]
    work = work.dropna(subset=features)

    y_dm = work.groupby("unit_id")["log_residential_price"].transform("mean")
    y_tm = work.groupby("year_month")["log_residential_price"].transform("mean")
    y_resid = (work["log_residential_price"]
               - y_dm - y_tm + work["log_residential_price"].mean())

    X_resid = pd.DataFrame(index=work.index)
    for c in features:
        e = work.groupby("unit_id")[c].transform("mean")
        t = work.groupby("year_month")[c].transform("mean")
        X_resid[c] = work[c] - e - t + work[c].mean()

    rf = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_resid, y_resid)

    importance_df = pd.DataFrame({
        "feature": features,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1
    importance_df["n_obs"] = len(work)
    importance_df["r2_oob"] = float(rf.score(X_resid, y_resid))

    importance_df.to_csv(FINAL / "ml_rf_importance.csv", index=False)
    print(f"  Random forest variable importance (top 5):")
    for _, r in importance_df.head(5).iterrows():
        print(f"    {int(r['rank']):2d}. {r['feature']:<30s} importance={r['importance']:.4f}")
    return importance_df


def _within_two_way(df: pd.DataFrame, col: str, e: str, t: str) -> np.ndarray:
    """Two-way within transformation: subtract entity and time means, add grand mean."""
    overall = df[col].mean()
    ent = df.groupby(e)[col].transform("mean")
    tim = df.groupby(t)[col].transform("mean")
    return (df[col] - ent - tim + overall).to_numpy()


def _wild_cluster_bootstrap(
    df: pd.DataFrame,
    outcome: str,
    treatments: list[str],
    entity_col: str,
    time_col: str,
    cluster_col: str,
    n_boot: int = 999,
) -> dict:
    """Run a within-transformed wild cluster bootstrap."""
    y = _within_two_way(df, outcome, entity_col, time_col)
    X = np.column_stack([
        _within_two_way(df, t, entity_col, time_col) for t in treatments
    ])

    clusters = df[cluster_col].astype(str).to_numpy()
    unique_clusters, cluster_pos = np.unique(clusters, return_inverse=True)

    XtX_inv = np.linalg.inv(X.T @ X)
    beta_obs = XtX_inv @ (X.T @ y)
    resid = y - X @ beta_obs
    fitted = X @ beta_obs

    n, k = X.shape
    se_cluster = np.zeros(k)
    sum_xu = np.zeros((len(unique_clusters), k))
    for i in range(n):
        sum_xu[cluster_pos[i]] += X[i] * resid[i]
    meat = sum_xu.T @ sum_xu
    G = len(unique_clusters)
    dof_adj = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    cov_cluster = dof_adj * XtX_inv @ meat @ XtX_inv
    se_cluster = np.sqrt(np.diag(cov_cluster))

    from scipy.stats import t as student_t
    p_analytical = 2 * (1 - student_t.cdf(np.abs(beta_obs / se_cluster), df=G - 1))

    boot_betas = np.zeros((n_boot, k))
    for b in range(n_boot):
        w = RNG.choice([-1.0, 1.0], size=len(unique_clusters))
        w_obs = w[cluster_pos]
        y_star = fitted + resid * w_obs
        boot_betas[b] = XtX_inv @ (X.T @ y_star)

    summary = {}
    for j, t_name in enumerate(treatments):
        centered = boot_betas[:, j] - boot_betas[:, j].mean()
        p_boot = float(np.mean(np.abs(centered) >= np.abs(beta_obs[j])))
        summary[t_name] = {
            "beta": float(beta_obs[j]),
            "se_analytical": float(se_cluster[j]),
            "p_analytical": float(p_analytical[j]),
            "boot_p": p_boot,
            "n_boot": n_boot,
        }
    return summary


def run_wild_cluster_bootstrap() -> pd.DataFrame:
    """Bootstrap the triple-DiD coefficients."""
    month = pd.read_csv(FINAL / "utility_area_month_panel_2023_2026.csv")
    work = month.dropna(
        subset=["log_residential_price", "in_pjm", "high_dc_utility",
                "pjm_shock_active", "unit_id"]
    ).copy()
    work = work[work["residential_customers"] >= 1000]

    work["in_pjm"] = work["in_pjm"].astype(int)
    work["high_dc"] = work["high_dc_utility"].astype(int)
    work["post"] = work["pjm_shock_active"].astype(int)
    work["pjm_post"] = work["in_pjm"] * work["post"]
    work["highdc_post"] = work["high_dc"] * work["post"]
    work["pjm_highdc_post"] = work["in_pjm"] * work["high_dc"] * work["post"]
    work["year_month"] = (
        work["year"].astype(int).astype(str) + "_" +
        work["month"].astype(int).astype(str).str.zfill(2)
    )

    print(f"  Wild cluster bootstrap on {work['unit_id'].nunique()} utility-state clusters,")
    print(f"    {work['unit_id'][work['high_dc'] == 1].nunique()} of which are high-DC")
    print(f"  Running 999 Rademacher replications (within-transformed)...")

    coefs = ["pjm_post", "highdc_post", "pjm_highdc_post"]
    summary = _wild_cluster_bootstrap(
        work,
        outcome="log_residential_price",
        treatments=coefs,
        entity_col="unit_id",
        time_col="year_month",
        cluster_col="unit_id",
        n_boot=999,
    )

    rows = []
    label_map = {
        "pjm_post": "PJM x post (broad uplift)",
        "highdc_post": "High-DC x post (outside-PJM component)",
        "pjm_highdc_post": "PJM x high-DC x post (triple-DiD)",
    }
    for c, info in summary.items():
        rows.append({
            "spec": "wild_cluster_bootstrap_CGM2008",
            "sample": "monthly_panel_2023_2026",
            "label": label_map.get(c, c),
            "coefficient_name": c,
            "coefficient": info["beta"],
            "std_error_analytical": info.get("se_analytical", np.nan),
            "p_value_analytical": info.get("p_analytical", np.nan),
            "p_value_bootstrap": info["boot_p"],
            "n_bootstrap_replications": info["n_boot"],
            "n_obs": int(len(work)),
            "n_clusters": int(work["unit_id"].nunique()),
        })
        print(f"    {label_map.get(c, c)}: β={info['beta']*100:+.2f} pp, "
              f"analytical p={info.get('p_analytical', np.nan):.3f}, "
              f"bootstrap p={info['boot_p']:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(FINAL / "ml_wild_bootstrap.csv", index=False)
    return out


def main():
    print("=" * 70)
    print("ML ROBUSTNESS SPECIFICATIONS")
    print("=" * 70)
    print("\n[1/3] Lasso post-selection on historical 2010-2024 panel...")
    run_lasso_post_selection()
    print("\n[2/3] Random forest variable importance on monthly panel...")
    run_rf_variable_importance()
    print("\n[3/3] Wild cluster bootstrap on triple-DiD coefficients...")
    run_wild_cluster_bootstrap()
    print("\nAll ML robustness outputs written to data/processed/final/")


if __name__ == "__main__":
    main()
