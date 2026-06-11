#!/usr/bin/env python3
"""Fetch US data-center locations from OpenStreetMap."""

import json
import sys
from pathlib import Path

import pandas as pd
import requests


# Overpass mirrors provide fallback capacity.
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "data_centers"
OUT_CSV = OUT_DIR / "osm_data_centers.csv"
OUT_RAW = OUT_DIR / "osm_raw.json"

# Identify the project to Overpass servers.
UA = "metrics-project/1.0 (research; contact: goelmalank10@gmail.com)"

HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json",
}

# Query common data-center tags and polygon centroids.
QUERY = """
[out:json][timeout:240];
area["ISO3166-1"="US"][admin_level=2]->.us;
(
  nwr["telecom"="data_center"](area.us);
  nwr["man_made"="data_center"](area.us);
  nwr["building"="data_center"](area.us);
  nwr["telecom"="data_centre"](area.us);
);
out center tags;
"""


def fetch() -> dict:
    """Try each Overpass endpoint until one succeeds."""
    last_err = None
    for endpoint in OVERPASS_ENDPOINTS:
        print(f"Querying {endpoint} (typically 30-90s)...")
        try:
            # Raw POST bodies avoid some 406 responses.
            r = requests.post(endpoint, data=QUERY.encode("utf-8"),
                              headers=HEADERS, timeout=300)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            print(f"  -> HTTP {status}: trying next mirror")
            last_err = e
        except requests.RequestException as e:
            print(f"  -> {type(e).__name__}: trying next mirror")
            last_err = e
    raise RuntimeError(f"All Overpass endpoints failed. Last error: {last_err}")


def normalize(elements: list) -> list[dict]:
    rows = []
    for el in elements:
        tags = el.get("tags", {})

        # Ways and relations use the returned centroid.
        if "lat" in el and "lon" in el:
            lat, lon = el["lat"], el["lon"]
        else:
            c = el.get("center", {})
            lat, lon = c.get("lat"), c.get("lon")

        street = " ".join(filter(None, [
            tags.get("addr:housenumber"),
            tags.get("addr:street"),
        ])).strip() or None

        rows.append({
            "osm_id":     el.get("id"),
            "osm_type":   el.get("type"),
            "name":       tags.get("name"),
            "operator":   tags.get("operator"),
            "street":     street,
            "city":       tags.get("addr:city"),
            "state":      tags.get("addr:state"),
            "zip":        tags.get("addr:postcode"),
            "country":    tags.get("addr:country"),
            "lat":        lat,
            "lon":        lon,
            "building":   tags.get("building"),
            "telecom":    tags.get("telecom"),
            "man_made":   tags.get("man_made"),
            "start_date": tags.get("start_date"),
            "power":      tags.get("power") or tags.get("generator:output:electricity"),
            "tags_raw":   json.dumps(tags, ensure_ascii=False),
        })
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        data = fetch()
    except Exception as e:
        print(f"\nAll Overpass mirrors failed: {e}")
        print("Wait 60s and re-run, or try again later -- Overpass servers")
        print("rate-limit heavy queries. The query may also be timing out;")
        print("if so, narrow the geographic scope (e.g. one state at a time).")
        return 1

    OUT_RAW.write_text(json.dumps(data, indent=2))
    print(f"Raw response saved: {OUT_RAW}")

    elements = data.get("elements", [])
    print(f"Found {len(elements)} OSM features tagged as data centers in the US")

    if not elements:
        print("No results. The Overpass API may be down -- try mirror:")
        print("  https://overpass.kumi.systems/api/interpreter")
        print("(edit OVERPASS at top of script)")
        return 1

    rows = normalize(elements)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV}  ({len(df)} rows)")

    # Coverage diagnostics.
    print("\n--- State breakdown (top 15) ---")
    print(df["state"].value_counts().head(15).to_string())

    with_addr = df.dropna(subset=["street", "city", "state"]).shape[0]
    with_coord = df.dropna(subset=["lat", "lon"]).shape[0]
    with_name = df["name"].notna().sum()
    print(f"\nFacilities with full street+city+state:  {with_addr}")
    print(f"Facilities with lat/lon coordinates:    {with_coord}")
    print(f"Facilities with a 'name' tag:           {with_name}")

    print("\nNext steps:")
    print(" 1. Open the CSV and spot-check 10 facilities.")
    print(" 2. Compare top-15 states to the proposal's expected top-10:")
    print("    VA, TX, CA, AZ, IL, OR, IA, OH, PA, GA.")
    print(" 3. If Virginia or Texas look thin, supplement manually for")
    print("    the major hyperscalers (Amazon, Google, Microsoft, Meta).")
    print(" 4. Spatial-join lat/lon to the utility territory shapefile")
    print("    to assign each facility to a utility.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
