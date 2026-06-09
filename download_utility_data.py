#!/usr/bin/env python3
"""Download missing EIA Form 861 vintages and Census ACS state controls."""

from __future__ import annotations

import io
import re
import ssl
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

import pandas as pd

from utility_config import PANEL_END, PANEL_START, RAW, ensure_dirs


EIA_ZIP_URLS = {
    2010: "https://www.eia.gov/electricity/data/eia861/archive/zip/861_2010.zip",
    2011: "https://www.eia.gov/electricity/data/eia861/archive/zip/861_2011.zip",
    2012: "https://www.eia.gov/electricity/data/eia861/archive/zip/f8612012.zip",
    2013: "https://www.eia.gov/electricity/data/eia861/archive/zip/f8612013.zip",
    2014: "https://www.eia.gov/electricity/data/eia861/archive/zip/f8612014.zip",
}

# ACS 5-year subject table: population, median household income, median home value
ACS_VARS = {
    "NAME": "name",
    "S0101_C01_001E": "population",
    "S1901_C01_001E": "median_household_income",
    "S2503_C03_001E": "median_home_value",
    "S2301_C04_001E": "pct_bachelors_plus",
}

STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT",
    "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL",
    "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD",
    "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE",
    "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV",
    "55": "WI", "56": "WY",
}

# Map panel year to ACS 5-year vintage (end year of estimate)
ACS_VINTAGE_BY_PANEL_YEAR = {
    y: min(2023, max(2012, y)) for y in range(PANEL_START, PANEL_END + 1)
}


def download_bytes(url: str, timeout: int = 120) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (research project)"})
    try:
        ctx = ssl.create_default_context()
        with urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read()
    except ssl.SSLError:
        # macOS Python installs often lack system certs; fall back to curl.
        proc = subprocess.run(
            ["curl", "-fsSL", url],
            check=True,
            capture_output=True,
            timeout=timeout,
        )
        return proc.stdout


def extract_eia_year(year: int) -> bool:
    dest_dir = RAW / "f-861" / f"f861{year}"
    if dest_dir.exists() and any(dest_dir.glob(f"**/*Sales*{year}*.xls*")):
        print(f"EIA {year}: already present")
        return True

    url = EIA_ZIP_URLS.get(year)
    if not url:
        print(f"EIA {year}: no URL configured")
        return False

    print(f"EIA {year}: downloading {url}")
    try:
        payload = download_bytes(url)
    except Exception as exc:
        print(f"EIA {year}: download failed ({exc})")
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(dest_dir)
    print(f"EIA {year}: extracted to {dest_dir}")
    return any(dest_dir.glob(f"**/*Sales*{year}*.xls*"))


def fetch_acs_vintage(vintage: int) -> pd.DataFrame:
    var_list = ",".join(ACS_VARS)
    url = (
        f"https://api.census.gov/data/{vintage}/acs/acs5/subject"
        f"?get={var_list}&for=state:*"
    )
    print(f"ACS vintage {vintage}: {url}")
    payload = download_bytes(url)
    raw = pd.read_csv(io.BytesIO(payload))
    raw = raw.rename(columns=raw.iloc[0]).drop(0)
    raw = raw.rename(columns=ACS_VARS)
    raw["state_fips"] = raw["state"].astype(str).str.zfill(2)
    raw["state"] = raw["state_fips"].map(STATE_FIPS)
    for col in ACS_VARS.values():
        if col != "name":
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["acs_vintage"] = vintage
    return raw.dropna(subset=["state"])


def build_acs_panel() -> pd.DataFrame:
    frames = []
    vintages = sorted(set(ACS_VINTAGE_BY_PANEL_YEAR.values()))
    cache: dict[int, pd.DataFrame] = {}
    for year in range(PANEL_START, PANEL_END + 1):
        vintage = ACS_VINTAGE_BY_PANEL_YEAR[year]
        if vintage not in cache:
            try:
                cache[vintage] = fetch_acs_vintage(vintage)
            except Exception as exc:
                print(f"ACS vintage {vintage} failed: {exc}")
                continue
        slice_df = cache[vintage].copy()
        slice_df["year"] = year
        frames.append(slice_df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out_path = RAW / "acs_state_controls.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out):,} rows)")
    return out


def main() -> int:
    ensure_dirs()
    (RAW / "f-861").mkdir(parents=True, exist_ok=True)
    for year in range(2010, 2015):
        extract_eia_year(year)
    build_acs_panel()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
