#!/usr/bin/env python3
"""Normalize NOAA monthly degree-day files to state-month CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from utility_config import ANALYSIS_STATES, MONTH_LOOKUP, RAW, ROOT, STATE_NAME_TO_ABBR


PROCESSED = ROOT / "data" / "processed"
OUT_PATH = PROCESSED / "degree_days_state_month.csv"

DEFAULT_WEATHER_STATES = (
    "GA", "IL", "MD", "ME", "NC", "NJ", "OH", "PA", "SC", "TN", "TX", "VA", "WV",
)

STATE_NAME_ALIASES = {
    "DISTRCT COLUMBIA": "DC",
}


def parse_month_year_from_name(path: Path) -> tuple[int, int] | None:
    """Extract year and month from NOAA archive filenames."""
    stem = path.stem.lower().replace(",", " ")
    tokens = re.findall(r"[a-z]+|\d{2,4}", stem)
    month = year = None
    for token in tokens:
        if token in MONTH_LOOKUP:
            month = MONTH_LOOKUP[token]
        elif token.isdigit():
            value = int(token)
            if value < 100:
                year = 1900 + value if value >= 70 else 2000 + value
            else:
                year = value
    return (year, month) if year and month else None


def state_abbr_from_name(name: str) -> str | None:
    normalized = re.sub(r"\s+", " ", name.strip())
    return STATE_NAME_TO_ABBR.get(normalized) or STATE_NAME_ALIASES.get(normalized)


def parse_noaa_file(path: Path, source: str, states: set[str] | None) -> list[dict]:
    parsed = parse_month_year_from_name(path)
    if not parsed:
        return []
    year, month = parsed
    try:
        text = path.read_text(encoding="latin-1")
    except OSError:
        return []

    rows: list[dict] = []
    for line in text.splitlines():
        match = re.match(r"^\s*([A-Z][A-Z ]*[A-Z])\s+(-?\d+)\s+", line)
        if not match:
            continue
        state = state_abbr_from_name(match.group(1))
        if not state or (states is not None and state not in states):
            continue
        value = int(match.group(2))
        if value < 0:
            continue
        rows.append({
            "state": state,
            "year": year,
            "month": month,
            source: float(value),
        })
    return rows


def build_degree_days(
    states: set[str] | None = None,
    start_year: int = 2015,
    end_year: int | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for source in ("cdd", "hdd"):
        base = RAW / "noaa" / source
        if not base.exists():
            continue
        for path in sorted(base.glob("*/*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            rows.extend(parse_noaa_file(path, source, states))

    if not rows:
        return pd.DataFrame(columns=["state", "year", "month", "date", "cdd", "hdd"])

    raw = pd.DataFrame(rows)
    raw = raw[raw["year"].ge(start_year)].copy()
    if end_year is not None:
        raw = raw[raw["year"].le(end_year)].copy()

    out = (
        raw.groupby(["state", "year", "month"], as_index=False)
        .agg(cdd=("cdd", "max"), hdd=("hdd", "max"))
        .sort_values(["state", "year", "month"])
        .reset_index(drop=True)
    )
    out["date"] = pd.to_datetime(
        out["year"].astype(str) + "-" + out["month"].astype(str).str.zfill(2) + "-01"
    ).dt.strftime("%Y-%m-%d")
    return out[["state", "year", "month", "date", "cdd", "hdd"]]


def parse_state_option(value: str) -> set[str] | None:
    normalized = value.strip().lower()
    if normalized == "all":
        return None
    if normalized == "analysis":
        return set(ANALYSIS_STATES)
    if normalized in {"default", "final"}:
        return set(DEFAULT_WEATHER_STATES)
    return {part.strip().upper() for part in value.split(",") if part.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--states",
        default="default",
        help="State filter: default/final, analysis, all, or comma-separated abbreviations.",
    )
    parser.add_argument("--all-states", action="store_true", help="Shortcut for --states all.")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    states = None if args.all_states else parse_state_option(args.states)
    df = build_degree_days(states=states, start_year=args.start_year, end_year=args.end_year)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    state_label = "all states" if states is None else f"{len(states)} states"
    print(f"Wrote {args.output} with {len(df):,} state-month rows for {state_label}.")
    if not df.empty:
        print(f"Years: {int(df['year'].min())}-{int(df['year'].max())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
