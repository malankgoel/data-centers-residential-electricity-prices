#!/usr/bin/env python3
"""Download NOAA monthly state-level degree-day files."""

import argparse
import re
import sys
import time
import urllib.parse
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


ARCHIVES = ("https://ftp.cpc.ncep.noaa.gov/htdocs/products/"
            "analysis_monitoring/cdus/degree_days/archives")

# NOAA uses different paths for CDD and HDD files.
CANDIDATES = {
    "cdd": [
        f"{ARCHIVES}/Cooling%20Degree%20Days/monthly%20cooling%20degree%20days%20state/",
    ],
    "hdd": [
        f"{ARCHIVES}/Heating%20degree%20Days/monthly%20states/",
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = PROJECT_ROOT / "data" / "raw" / "noaa"
UA = "Mozilla/5.0 (research project; contact: goelmalank10@gmail.com)"
SLEEP = 0.15  # seconds between requests

YEAR_RE = re.compile(r"^\s*(\d{4})/?\s*$")


def list_links(url: str) -> list[tuple[str, str]]:
    """Return [(display_text, absolute_url)] from an Apache directory listing."""
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        text = a.get_text(strip=True)
        if not href:
            continue
        # Skip directory navigation links.
        if href.startswith("?") or href.startswith("/"):
            continue
        if href.startswith("..") or text.lower().startswith("parent"):
            continue
        out.append((text, urljoin(url, href)))
    return out


def discover_base(candidates: list[str]) -> str | None:
    """Return first candidate URL that returns 200; else None."""
    for c in candidates:
        try:
            r = requests.get(c, headers={"User-Agent": UA}, timeout=20)
            if r.status_code == 200 and "Index of" in r.text:
                return c
        except requests.RequestException:
            continue
    return None


def year_folders(base_url: str) -> list[tuple[str, str]]:
    """List of (year_str, year_url) for every 4-digit-named entry."""
    out = []
    for text, href in list_links(base_url):
        m = YEAR_RE.match(text)
        if m:
            out.append((m.group(1), href))
            continue
        # Fallback: check the URL last segment
        last = href.rstrip("/").rsplit("/", 1)[-1]
        if last.isdigit() and len(last) == 4:
            out.append((last, href))
    return sorted(set(out))


def month_files(year_url: str) -> list[tuple[str, str]]:
    """List of (decoded_filename, file_url) inside a year folder."""
    out = []
    for _, href in list_links(year_url):
        if href.endswith("/"):
            continue  # skip nested directories
        fname = urllib.parse.unquote(href.rsplit("/", 1)[-1]).strip()
        # Reject placeholder/empty filenames.
        if not fname or fname in (".txt",) or fname.startswith("."):
            continue
        out.append((fname, href))
    return out


def safe_filename(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_").strip()
    return name


def download_one(url: str, dest: Path) -> str:
    """Return 'new', 'cached', or 'fail'."""
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    time.sleep(SLEEP)
    return "new"


def cleanup_stale(out_dir: Path):
    """Remove any %20.txt / ' .txt' placeholder from a previous run."""
    for stale_name in ("%20.txt", " .txt", ".txt"):
        p = out_dir / stale_name
        if p.exists():
            print(f"  Removing stale placeholder: {p.name}")
            p.unlink()


def run(label: str, candidates: list[str]):
    print(f"\n=== {label.upper()} ===")
    base = discover_base(candidates)
    if not base:
        print(f"  None of the candidate URLs worked. Tried:")
        for c in candidates:
            print(f"    {c}")
        print(f"  Open {ARCHIVES}/ in a browser to find the correct path,")
        print(f"  then add it to CANDIDATES['{label}'] and rerun.")
        return

    print(f"  Base: {base}")
    out_dir = OUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale(out_dir)

    years = year_folders(base)
    if not years:
        print("  No year folders found at this URL. Check structure manually.")
        return
    print(f"  Years found: {years[0][0]} to {years[-1][0]} ({len(years)} total)")

    counts = {"new": 0, "cached": 0, "fail": 0}
    for year, year_url in years:
        year_dir = out_dir / year
        year_dir.mkdir(exist_ok=True)
        try:
            files = month_files(year_url)
        except Exception as e:
            print(f"  [{year}] index failed: {e}")
            counts["fail"] += 1
            continue
        if not files:
            print(f"  [{year}] no files found")
            continue
        for fname, furl in files:
            dest = year_dir / safe_filename(fname)
            try:
                status = download_one(furl, dest)
                counts[status] += 1
            except Exception as e:
                counts["fail"] += 1
                print(f"  [{year}/{fname}] fail: {e}")
        sys.stdout.write(f"  [{year}] {len(files)} files done\r")
        sys.stdout.flush()
    print()  # newline after the \r loop

    print(f"  Summary: {counts['new']} new, "
          f"{counts['cached']} cached, {counts['fail']} failed")
    print(f"  Saved under: {out_dir}/")

    # Quick preview of one recent file
    pick = next(iter(sorted((out_dir / years[-1][0]).glob("*"))), None)
    if pick:
        print(f"\n  Preview of {years[-1][0]}/{pick.name}:")
        print("  " + "-" * 60)
        content = pick.read_text(errors="replace").splitlines()[:18]
        for line in content:
            print("  " + line)
        print("  " + "-" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["cdd", "hdd", "both"],
                        default="both",
                        help="Which source to download (default: both)")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    labels = [args.source] if args.source != "both" else list(CANDIDATES.keys())
    for label in labels:
        run(label, CANDIDATES[label])
    print("\nDone. Each (state, year, month) file is fixed-width text.")
    print("Parse with pd.read_fwf() once you've eyeballed one to confirm column widths.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
