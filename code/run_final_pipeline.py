#!/usr/bin/env python3
"""Run the final output pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
FINAL_DIR = CODE_DIR.parent / "data" / "processed" / "final"


def run(script: str) -> None:
    path = CODE_DIR / script
    print(f"\n{'=' * 72}\n>>> {path.name}\n{'=' * 72}")
    subprocess.run([sys.executable, str(path)], check=True)


def sklearn_available() -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", "import sklearn"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return probe.returncode == 0


def ml_outputs_exist() -> bool:
    return all(
        (FINAL_DIR / name).exists()
        for name in ["ml_lasso_selection.csv", "ml_rf_importance.csv", "ml_wild_bootstrap.csv"]
    )


def main() -> int:
    run("run_service_territory_did.py")
    run("process_recent_retail_pjm_inputs.py")
    run("process_degree_days.py")
    run("build_final_utility_area_panel.py")
    run("run_final_utility_area_analysis.py")
    if sklearn_available():
        run("run_ml_robustness.py")
    elif ml_outputs_exist():
        print("\nSkipping run_ml_robustness.py: scikit-learn is not installed for this Python,")
        print("but existing ML robustness CSVs are present in data/processed/final.")
    else:
        raise RuntimeError(
            "scikit-learn is required to regenerate ML robustness outputs, "
            "and the existing ML CSVs are missing."
        )
    run("make_final_figures.py")
    print("\nFinal pipeline complete. Outputs are in figures/final and figures/tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
