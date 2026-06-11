"""Shared configuration for the utility-year data-center analysis."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed" / "utility"
STATE_PROCESSED = ROOT / "data" / "processed" / "state_pairs"
FIGURES = ROOT / "figures" / "utility"
STATE_FIGURES = ROOT / "figures" / "state_pairs"
TABLES = ROOT / "tables" / "utility"
STATE_TABLES = ROOT / "tables" / "state_pairs"
OUTPUT = ROOT / "output"

PANEL_START = 2010
PANEL_END = 2025

CHATGPT_LAUNCH_YEAR = 2022
PRE_EXPOSURE_CUTOFF = 2021
POST_YEAR = 2022
EVENT_REF_YEAR = 2021

PJM_STATES = ["VA", "MD", "NJ", "OH", "PA", "WV", "IL", "NC", "DE", "DC"]
SOUTHEAST_STATES = ["VA", "MD", "NC", "SC", "GA", "TN", "FL", "AL", "MS", "KY"]
NEIGHBOR_STATES = ["VA", "MD", "NC", "WV", "KY", "TN"]

ANALYSIS_STATES = [
    "VA", "MD", "NC", "TX", "GA", "IL", "OH", "PA", "NJ", "SC", "TN",
    "IA", "AZ", "OR", "NE", "WV", "ME", "FL", "MN", "CO", "UT", "NV",
]

MIN_RESIDENTIAL_CUSTOMERS = 5_000

STATE_NAME_TO_ABBR = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "DISTRICT COLUMBIA": "DC", "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI",
    "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT",
    "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI", "WYOMING": "WY",
}

MONTH_LOOKUP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def ensure_dirs() -> None:
    for path in (PROCESSED, STATE_PROCESSED, FIGURES, STATE_FIGURES, TABLES, STATE_TABLES, OUTPUT):
        path.mkdir(parents=True, exist_ok=True)
