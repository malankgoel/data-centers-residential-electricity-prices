#!/usr/bin/env python3
"""Enrich OSM data centers with opening years."""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "data_centers"
OUT = ROOT / "data" / "processed" / "data_centers"
OUT.mkdir(parents=True, exist_ok=True)
CACHE_PATH = RAW / "opening_year_cache.json"

USER_AGENT = (
    "MetricsProjectBot/1.0 (academic research; contact: student@example.edu) "
    "Python-requests/" + requests.__version__
)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})

YEAR_RE = re.compile(r"\b(19[89]\d|20[0-2]\d)\b")
GOOD_CONTEXT_RE = re.compile(
    r"(open(?:ed|ing)?|onlin(?:e)?|commission(?:ed|ing)?|built|complete[d]?|"
    r"launch(?:ed|ing)?|inaugurate[d]?|construct(?:ed|ion completed)|"
    r"began operations|began operating|came online|operational|"
    r"unveil(?:ed)?|broke ground|broke-ground|going live|went live|first phase|"
    r"establish(?:ed)?|founded|opened in|opened by|opened on)",
    re.IGNORECASE,
)
BAD_CONTEXT_RE = re.compile(
    r"(quarterly|annual report|fiscal year|q[1-4] 20|by 20[2-9]\d|"
    r"plans to|will open|to be completed|expected|projected|forecast|"
    r"acquired|merger|merged|sold|bought|funding round)",
    re.IGNORECASE,
)

CACHE: dict[str, dict[str, Any]] = {}
WD_INDEX: dict[str, int] = {}  # normalized_facility_name -> inception_year
HYPERSCALER_INDEX: dict[str, dict[str, int]] = {}  # operator -> {normalized_city_or_county -> year}


def load_cache() -> None:
    global CACHE
    if CACHE_PATH.exists():
        try:
            CACHE = json.loads(CACHE_PATH.read_text())
        except Exception:
            CACHE = {}
    else:
        CACHE = {}


def save_cache() -> None:
    CACHE_PATH.write_text(json.dumps(CACHE, indent=2))


def safe_get(url: str, *, timeout: float = 8.0, **kw) -> requests.Response | None:
    try:
        r = SESSION.get(url, timeout=timeout, **kw)
        if r.status_code == 200:
            return r
    except Exception:
        return None
    return None


# Pass A: parse OSM tags

def parse_year_from_string(s: str | None) -> int | None:
    if not s or not isinstance(s, str):
        return None
    m = YEAR_RE.search(s)
    if m:
        y = int(m.group(1))
        if 1985 <= y <= 2025:
            return y
    return None


def pass_a_osm_tags(row: pd.Series) -> tuple[int | None, str]:
    for col in ("start_date", "opening_date", "construction_start_date"):
        if col in row and pd.notna(row.get(col)):
            y = parse_year_from_string(str(row[col]))
            if y is not None:
                return y, f"osm:{col}"
    tags_raw = row.get("tags_raw")
    if isinstance(tags_raw, str) and tags_raw.strip().startswith("{"):
        try:
            tags = json.loads(tags_raw)
        except Exception:
            tags = {}
        for k in ("start_date", "opening_date", "construction:start_date", "open_date", "year_built"):
            if k in tags:
                y = parse_year_from_string(tags[k])
                if y is not None:
                    return y, f"osm_tag:{k}"
    return None, ""


def get_wikidata_qcode(row: pd.Series) -> str | None:
    tags_raw = row.get("tags_raw")
    if not isinstance(tags_raw, str):
        return None
    try:
        tags = json.loads(tags_raw)
    except Exception:
        return None
    q = tags.get("wikidata")
    if isinstance(q, str) and q.startswith("Q"):
        return q
    return None


# Pass B0: Wikidata index

def normalize_name(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\b(data ?center|datacentre|dc|facility|building|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_wikidata_index() -> None:
    """Build a Wikidata name-to-year index."""
    global WD_INDEX
    cache_key = "wd_sparql_index_v2"
    if cache_key in CACHE:
        WD_INDEX = CACHE[cache_key]
        return
    # Include subclasses of data center.
    query = """
    SELECT ?dcLabel ?inception ?operatorLabel WHERE {
      ?dc wdt:P31/wdt:P279* wd:Q671224.
      ?dc wdt:P571 ?inception.
      OPTIONAL { ?dc wdt:P137 ?operator. }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """
    url = "https://query.wikidata.org/sparql"
    try:
        r = SESSION.get(
            url,
            params={"query": query, "format": "json"},
            headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT},
            timeout=30,
        )
        if r.status_code != 200:
            CACHE[cache_key] = {}
            return
        data = r.json()
    except Exception:
        CACHE[cache_key] = {}
        return
    idx: dict[str, int] = {}
    for binding in data.get("results", {}).get("bindings", []):
        label = binding.get("dcLabel", {}).get("value", "")
        inception = binding.get("inception", {}).get("value", "")
        op_label = binding.get("operatorLabel", {}).get("value", "")
        if not label or not inception:
            continue
        try:
            year = int(inception[:4])
        except Exception:
            continue
        if not (1985 <= year <= 2025):
            continue
        keys = [normalize_name(label)]
        if op_label:
            keys.append(normalize_name(f"{op_label} {label}"))
            keys.append(normalize_name(label.replace(op_label, "").strip()))
        for k in keys:
            if k and (k not in idx or year < idx[k]):
                idx[k] = year
    CACHE[cache_key] = idx
    WD_INDEX = idx
    print(f"  wikidata SPARQL index: {len(idx)} entries")


def pass_b0_wikidata_index(name: str, operator: str | None) -> tuple[int | None, str]:
    if not name and not operator:
        return None, ""
    candidates = []
    if name:
        candidates.append(normalize_name(name))
    if operator and name:
        candidates.append(normalize_name(f"{operator} {name}"))
        candidates.append(normalize_name(name.replace(operator, "")))
    for k in candidates:
        if k and k in WD_INDEX:
            return WD_INDEX[k], "wikidata:sparql_index"
    return None, ""


# Pass B: Wikidata inception (P571) lookup

DATA_CENTER_QCODES = {"Q671224"}  # data center


def _entity_is_data_center(claims: dict) -> bool:
    for claim in claims.get("P31", []):
        try:
            qid = claim["mainsnak"]["datavalue"]["value"]["id"]
        except Exception:
            continue
        if qid in DATA_CENTER_QCODES:
            return True
    return False


def pass_b_wikidata(qcode: str) -> tuple[int | None, str]:
    if not qcode:
        return None, ""
    cache_key = f"wd:{qcode}"
    if cache_key in CACHE:
        c = CACHE[cache_key]
        return c.get("year"), c.get("method", "")
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qcode}.json"
    r = safe_get(url, timeout=10)
    if r is None:
        CACHE[cache_key] = {"year": None, "method": ""}
        return None, ""
    try:
        ent = r.json().get("entities", {}).get(qcode, {})
        claims = ent.get("claims", {})
        # Reject parent-company inception years.
        if not _entity_is_data_center(claims):
            CACHE[cache_key] = {"year": None, "method": ""}
            return None, ""
        for claim in claims.get("P571", []):
            try:
                dv = claim["mainsnak"]["datavalue"]["value"]["time"]
                y = int(dv.lstrip("+").split("-")[0])
                if 1985 <= y <= 2025:
                    CACHE[cache_key] = {"year": y, "method": "wikidata:P571"}
                    return y, "wikidata:P571"
            except Exception:
                continue
    except Exception:
        pass
    CACHE[cache_key] = {"year": None, "method": ""}
    return None, ""


# Pass B1: Wikipedia operator tables

HYPERSCALER_PAGES: list[tuple[str, list[str]]] = [
    ("Google", ["Google data centers"]),
    ("Amazon Web Services", ["Amazon Web Services", "Timeline of Amazon Web Services"]),
    ("Microsoft", ["Microsoft Azure"]),
    ("Meta", ["Meta Platforms"]),
    ("Apple Inc.", ["Apple Park", "List of Apple Inc. facilities"]),
    ("Equinix", ["Equinix"]),
    ("Digital Realty", ["Digital Realty"]),
    ("CoreSite", ["CoreSite"]),
    ("CyrusOne", ["CyrusOne"]),
    ("Switch", ["Switch (company)"]),
    ("Quality Technology Services", ["Quality Technology Services"]),
    ("Flexential", ["Flexential"]),
    ("NTT", ["NTT Global Data Centers"]),
]

# Prefer opening years over announcement years.
PREFERRED_YEAR_VERBS = ("launched", "online", "operational", "completed",
                        "opened", "open", "commissioned", "live")
DEPRIORITIZED_VERBS = ("announced", "broke ground", "plan", "expansion", "expanded")


def parse_year_cell(cell_text: str) -> int | None:
    """Extract the preferred opening year from a table cell."""
    if not cell_text:
        return None
    text = re.sub(r"\[[^\]]*\]", "", cell_text)
    text_lc = text.lower()
    matches = [(int(m.group(1)), m.start(), m.end()) for m in YEAR_RE.finditer(text)]
    if not matches:
        return None
    candidates: list[tuple[int, int]] = []  # (priority, year)
    for i, (y, start, end) in enumerate(matches):
        if not (1990 <= y <= 2025):
            continue
        # Match each year with its following phrase.
        next_start = matches[i + 1][1] if i + 1 < len(matches) else min(len(text), end + 40)
        window = text_lc[end:next_start]
        prio = 3
        if any(v in window for v in PREFERRED_YEAR_VERBS):
            prio = 1
        elif any(v in window for v in DEPRIORITIZED_VERBS):
            prio = 4
        else:
            prio = 2
        candidates.append((prio, y))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def fetch_wikipedia_html(title: str) -> str:
    cache_key = f"wp_html:{title}"
    if cache_key in CACHE:
        return CACHE[cache_key]
    try:
        r = SESSION.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "parse", "page": title, "format": "json", "prop": "text"},
            timeout=15,
        )
        html = r.json().get("parse", {}).get("text", {}).get("*", "") if r.status_code == 200 else ""
    except Exception:
        html = ""
    CACHE[cache_key] = html
    return html


def location_keys(text: str) -> list[str]:
    """Normalize a Wikipedia location cell into lookup keys."""
    if not text:
        return []
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.split(r"[°\d]", text)[0]  # drop coordinate text
    parts = [p.strip() for p in re.split(r"[,/]", text) if p.strip()]
    keys = set()
    for p in parts:
        n = re.sub(r"\([^)]*\)", "", p).strip()
        n = re.sub(r"[^a-zA-Z0-9 ]+", " ", n).strip().lower()
        n = re.sub(r"\s+", " ", n)
        if n and 2 <= len(n) <= 50:
            keys.add(n)
            # Add shorter location keys.
            tokens = n.split()
            if tokens:
                keys.add(tokens[0])
                if len(tokens) >= 2:
                    keys.add(" ".join(tokens[:2]))
    return sorted(keys)


def load_hyperscaler_index() -> None:
    """Build operator -> {location_key -> year} from Wikipedia HTML tables."""
    global HYPERSCALER_INDEX
    cache_key = "hyperscaler_index_v1"
    if cache_key in CACHE:
        HYPERSCALER_INDEX = {op: {k: int(v) for k, v in d.items()} for op, d in CACHE[cache_key].items()}
        return
    index: dict[str, dict[str, int]] = {}
    for operator, pages in HYPERSCALER_PAGES:
        sub: dict[str, int] = {}
        for title in pages:
            html = fetch_wikipedia_html(title)
            if not html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            for table in soup.find_all("table"):
                ths = [h.get_text(" ", strip=True).lower() for h in table.find_all("th")]
                has_year = any(any(k in h for k in ("timeline", "year", "opened", "launched")) for h in ths)
                has_loc = any(any(k in h for k in ("location", "city", "site", "region", "facility")) for h in ths)
                if not (has_year and has_loc):
                    continue
                header_cells = table.find("tr")
                if header_cells is None:
                    continue
                col_names = [c.get_text(" ", strip=True).lower()
                             for c in header_cells.find_all(["th", "td"])]
                if not col_names:
                    continue
                year_col = next((i for i, n in enumerate(col_names)
                                 if any(k in n for k in ("timeline", "year", "opened", "launched"))), None)
                loc_col = next((i for i, n in enumerate(col_names)
                                if any(k in n for k in ("location", "city", "site", "region", "facility"))), None)
                if year_col is None or loc_col is None:
                    continue
                for tr in table.find_all("tr")[1:]:
                    cells = tr.find_all(["td", "th"])
                    if len(cells) <= max(year_col, loc_col):
                        continue
                    loc_text = cells[loc_col].get_text(" ", strip=True)
                    year_text = cells[year_col].get_text(" ", strip=True)
                    y = parse_year_cell(year_text)
                    if y is None:
                        continue
                    for key in location_keys(loc_text):
                        if key not in sub or y < sub[key]:
                            sub[key] = y
        if sub:
            index[operator] = sub
            print(f"  hyperscaler table [{operator}]: {len(sub)} location keys")
    CACHE[cache_key] = index
    HYPERSCALER_INDEX = index


def pass_b1_hyperscaler_table(name: str, operator: str | None, city: str | None) -> tuple[int | None, str]:
    """Match a facility against Wikipedia operator tables."""
    if not operator:
        return None, ""
    # Allow operator-name variants.
    op_match = None
    op_lc = operator.lower()
    for key in HYPERSCALER_INDEX:
        if key.lower() in op_lc or op_lc in key.lower():
            op_match = key
            break
    if op_match is None:
        return None, ""
    sub = HYPERSCALER_INDEX[op_match]

    keys: list[str] = []
    if city:
        ck = re.sub(r"[^a-zA-Z0-9 ]+", " ", city).strip().lower()
        ck = re.sub(r"\s+", " ", ck)
        if ck:
            keys.append(ck)
            tokens = ck.split()
            if tokens:
                keys.append(tokens[0])
    if name:
        for t in re.split(r"[\s\-/,()]+", name):
            t = t.strip().lower()
            if t and len(t) >= 4 and t not in GENERIC_WORDS:
                keys.append(t)

    for k in keys:
        if k in sub:
            return sub[k], f"wikipedia_table:{op_match}"
    return None, ""


# Pass C: Wikipedia search + page text scan

def wikipedia_search_titles(query: str, limit: int = 4) -> list[str]:
    cache_key = f"wp_search:{query}"
    if cache_key in CACHE:
        return CACHE[cache_key]
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "format": "json",
        "srlimit": limit,
        "srsearch": query,
    }
    try:
        r = SESSION.get(url, params=params, timeout=8)
        if r.status_code != 200:
            CACHE[cache_key] = []
            return []
        data = r.json()
        titles = [item["title"] for item in data.get("query", {}).get("search", [])]
    except Exception:
        titles = []
    CACHE[cache_key] = titles
    return titles


def wikipedia_page_text(title: str) -> str:
    cache_key = f"wp_page:{title}"
    if cache_key in CACHE:
        return CACHE[cache_key]
    url = (
        "https://en.wikipedia.org/w/api.php?"
        "action=query&prop=extracts&explaintext=1&format=json&"
        f"titles={quote_plus(title)}"
    )
    r = safe_get(url, timeout=10)
    txt = ""
    if r is not None:
        try:
            pages = r.json().get("query", {}).get("pages", {})
            for _, p in pages.items():
                txt = p.get("extract", "") or ""
                break
        except Exception:
            pass
    CACHE[cache_key] = txt
    return txt


def best_year_from_text(text: str, anchor: str | None = None) -> int | None:
    """Find an opening year near an optional anchor."""
    if not text:
        return None
    sentences = re.split(r"(?<=[\.\?\!])\s+", text)
    anchor_lc = anchor.lower() if anchor else None
    candidate_years: list[tuple[int, int]] = []  # (year, distance_to_anchor)
    anchor_idx: list[int] = []
    if anchor_lc:
        for i, s in enumerate(sentences):
            if anchor_lc in s.lower():
                anchor_idx.append(i)
    for i, s in enumerate(sentences):
        if BAD_CONTEXT_RE.search(s):
            continue
        if not GOOD_CONTEXT_RE.search(s):
            continue
        for m in YEAR_RE.finditer(s):
            y = int(m.group(1))
            if 1990 <= y <= 2025:
                if anchor_idx:
                    dist = min(abs(i - ai) for ai in anchor_idx)
                    if dist > 3:
                        continue
                    candidate_years.append((y, dist))
                else:
                    candidate_years.append((y, 0))
    if not candidate_years:
        return None
    candidate_years.sort(key=lambda t: (t[1], t[0]))
    return candidate_years[0][0]


GENERIC_WORDS = {
    "data", "center", "centre", "datacenter", "datacentre", "server",
    "room", "building", "facility", "the", "of", "and", "computer", "computing",
    "site", "network", "north", "south", "east", "west", "office",
    "tech", "technology", "campus", "complex",
    "communications", "communication", "telecom", "telecommunications",
    "services", "service", "systems", "system", "corporation", "company",
    "incorporated", "international", "global", "limited", "group",
    "solutions", "industries", "industrial", "enterprise", "enterprises",
}


def specific_tokens(name: str) -> list[str]:
    """Return distinctive facility-name tokens, with codes first."""
    raw = [t for t in re.split(r"[\s\-/,()]+", name or "") if t]
    coded = [t for t in raw if re.search(r"\d", t) and re.search(r"[A-Za-z]", t) and len(t) >= 2]
    other = [t for t in raw if t not in coded and len(t) >= 4 and t.lower() not in GENERIC_WORDS]
    return coded + other


def page_is_relevant(text: str, title: str, name: str, operator: str | None) -> bool:
    """Check whether a Wikipedia page matches the facility."""
    if not text:
        return False
    text_lc = text.lower()
    title_lc = (title or "").lower()
    if "data center" not in text_lc and "data centre" not in text_lc:
        return False
    facility_tokens = specific_tokens(name)
    op_in_title = bool(operator) and operator.lower() in title_lc
    facility_token_in_title = any(t.lower() in title_lc for t in facility_tokens)

    # Require an operator or facility token in the title.
    if not (op_in_title or facility_token_in_title):
        return False

    # Without an operator, require the strongest facility token.
    if not operator:
        if not facility_tokens:
            return False
        first = facility_tokens[0].lower()
        if first not in title_lc:
            return False
    return True


def pass_c2_wikipedia_city_anchored(name: str, operator: str | None, city: str | None) -> tuple[int | None, str]:
    """Find city-anchored opening years on operator pages."""
    if not operator or not city:
        return None, ""
    queries = [f"{operator} data centers", f"{operator} data center", operator]
    seen: set[str] = set()
    for q in queries:
        if q in seen:
            continue
        seen.add(q)
        titles = wikipedia_search_titles(q, limit=3)
        for title in titles[:3]:
            if operator.lower() not in title.lower():
                continue
            text = wikipedia_page_text(title)
            if not text:
                continue
            text_lc = text.lower()
            if city.lower() not in text_lc:
                continue
            y = best_year_from_text(text[:25000], anchor=city)
            if y is not None:
                return y, f"wikipedia_city:{title}"
    return None, ""


def pass_c_wikipedia(name: str, operator: str | None) -> tuple[int | None, str]:
    facility_tokens = specific_tokens(name)
    if not facility_tokens and not operator:
        return None, ""

    queries: list[str] = []
    if name and facility_tokens:
        queries.append(f"{name} data center")
        if operator and operator.lower() not in name.lower():
            queries.append(f"{operator} {name}")
    if operator:
        queries.append(f"{operator} data center")
    seen: set[str] = set()
    for q in queries:
        if q in seen:
            continue
        seen.add(q)
        titles = wikipedia_search_titles(q)
        for title in titles[:3]:
            text = wikipedia_page_text(title)
            if not page_is_relevant(text, title, name, operator):
                continue
            # Prefer facility codes to avoid operator-level years.
            anchor = None
            for t in facility_tokens:
                if re.search(r"\d", t) and re.search(r"[A-Za-z]", t) and t.lower() in text.lower():
                    anchor = t
                    break
            if anchor is None:
                # Avoid letter-only anchors on operator pages.
                op_in_title = bool(operator) and operator.lower() in title.lower()
                if op_in_title:
                    continue
                for t in facility_tokens:
                    if t.lower() in text.lower():
                        anchor = t
                        break
            if anchor is None:
                continue
            y = best_year_from_text(text[:25000], anchor=anchor)
            if y is not None:
                return y, f"wikipedia:{title}"
    return None, ""


# Pass D0: Baxtel.com facility lookup

def baxtel_search(query: str) -> list[str]:
    """Return the candidate baxtel.com facility URLs matching a query."""
    cache_key = f"baxtel_search:{query}"
    if cache_key in CACHE:
        return CACHE[cache_key]
    try:
        r = SESSION.get(
            "https://baxtel.com/search",
            params={"q": query},
            timeout=15,
            headers={"User-Agent": BROWSER_UA},
        )
        if r.status_code != 200:
            CACHE[cache_key] = []
            return []
    except Exception:
        CACHE[cache_key] = []
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    urls: list[str] = []
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if h.startswith("/data-center/") and h not in urls:
            urls.append(h)
        if len(urls) >= 5:
            break
    CACHE[cache_key] = urls
    return urls


BAXTEL_PRIORITY_PATTERNS = [
    # Lower priority is preferred.
    (re.compile(r"(?:opened|online|operational|launched|commission(?:ed)?|completed|in service|"
                r"inaugurated|came online|brought online)"
                r"\s+(?:in\s+)?(?:q[1-4]\s+of\s+)?(?:q[1-4]\s+)?\b(19[89]\d|20[012]\d)\b",
                flags=re.IGNORECASE), 1),
    (re.compile(r"\b(19[89]\d|20[012]\d)\b"
                r"\s+[-:–]?\s*(?:open|online|operational|launched|commissioned|completed)",
                flags=re.IGNORECASE), 1),
    # Treat construction years as weaker evidence.
    (re.compile(r"(?:data\s*center(?:\s+\w+){0,3}\s+(?:was\s+)?built|"
                r"purpose[- ]built(?:\s+by[^.]{0,40})?)\s+(?:in\s+)?\b(19[89]\d|20[012]\d)\b",
                flags=re.IGNORECASE), 2),
]


def baxtel_facility_year(path: str, anchor: str | None = None) -> int | None:
    cache_key = f"baxtel_page:{path}"
    if cache_key in CACHE:
        cached = CACHE[cache_key]
        return cached.get("year")
    try:
        r = SESSION.get(
            f"https://baxtel.com{path}",
            timeout=15,
            headers={"User-Agent": BROWSER_UA},
        )
        if r.status_code != 200:
            CACHE[cache_key] = {"year": None}
            return None
    except Exception:
        CACHE[cache_key] = {"year": None}
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(" ", strip=True)
    candidates: list[tuple[int, int]] = []
    for pat, prio in BAXTEL_PRIORITY_PATTERNS:
        for m in pat.finditer(text):
            y = int(m.group(1))
            if 1990 <= y <= 2025:
                candidates.append((prio, y))
    if not candidates:
        y = best_year_from_text(text[:12000], anchor=anchor)
        if y is not None:
            candidates.append((3, y))
    if not candidates:
        CACHE[cache_key] = {"year": None}
        return None
    candidates.sort()
    result_year = candidates[0][1]
    CACHE[cache_key] = {"year": result_year}
    return result_year


def _baxtel_slug_matches(slug: str, operator: str | None, name: str | None, city: str | None) -> bool:
    """Check whether a Baxtel URL slug matches the facility."""
    slug = slug.replace("/data-center/", "").lower()
    slug_tokens = set(re.split(r"[^a-z0-9]+", slug))
    slug_tokens.discard("")

    op_tokens: set[str] = set()
    if operator:
        op_tokens = {t for t in re.split(r"[^a-z0-9]+", operator.lower())
                     if t and t not in GENERIC_WORDS and len(t) >= 3}

    name_tokens: set[str] = set()
    if name:
        for t in re.split(r"[^a-z0-9]+", name.lower()):
            if not t or t in GENERIC_WORDS:
                continue
            if len(t) >= 3 or re.search(r"\d", t):
                name_tokens.add(t)
    name_specific = name_tokens - op_tokens

    city_token = ""
    if city:
        c = re.sub(r"[^a-z0-9]+", "", city.lower())
        if c and len(c) >= 3:
            city_token = c

    # Require an operator-token match when available.
    if op_tokens and not (op_tokens & slug_tokens):
        return False

    # Require a facility-token or city match.
    specific_hit = bool(name_specific & slug_tokens) or (city_token and city_token in slug_tokens)
    if op_tokens:
        return specific_hit
    # Without an operator, require two facility tokens.
    return specific_hit and len(name_specific & slug_tokens) >= 2


def pass_d0_baxtel(name: str, operator: str | None, city: str | None) -> tuple[int | None, str]:
    queries: list[str] = []
    if operator and name:
        queries.append(f"{operator} {name}")
    if name and len(name) >= 6:
        queries.append(name)
    if operator and city:
        queries.append(f"{operator} {city}")
    seen: set[str] = set()
    for q in queries:
        if not q or q in seen:
            continue
        seen.add(q)
        urls = baxtel_search(q)
        for u in urls[:5]:
            if not _baxtel_slug_matches(u, operator, name, city):
                continue
            y = baxtel_facility_year(u, anchor=name or city)
            if y is not None:
                return y, f"baxtel:{u}"
    return None, ""


# Pass D: Bing HTML search snippet scrape (fallback)

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)


def bing_search(query: str) -> list[tuple[str, str]]:
    """Return list of (title, snippet) tuples."""
    cache_key = f"bing:{query}"
    if cache_key in CACHE:
        return [tuple(x) for x in CACHE[cache_key]]
    url = "https://www.bing.com/search?q=" + quote_plus(query) + "&count=10&setlang=en-us"
    try:
        r = SESSION.get(
            url,
            timeout=12,
            headers={"User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.9"},
        )
        if r.status_code != 200:
            CACHE[cache_key] = []
            return []
    except Exception:
        CACHE[cache_key] = []
        return []
    results: list[tuple[str, str]] = []
    try:
        soup = BeautifulSoup(r.text, "html.parser")
        # Bing organic results live in li.b_algo
        for blk in soup.select("li.b_algo")[:8]:
            title_el = blk.select_one("h2")
            snip_el = blk.select_one(".b_caption p, .b_lineclamp1, .b_lineclamp2, .b_lineclamp3, .b_lineclamp4")
            title = title_el.get_text(" ", strip=True) if title_el else ""
            snip = snip_el.get_text(" ", strip=True) if snip_el else ""
            if title or snip:
                results.append((title, snip))
    except Exception:
        pass
    CACHE[cache_key] = results
    return results


def pass_d_web(name: str, operator: str | None, city: str | None, state: str | None) -> tuple[int | None, str]:
    parts = []
    if operator:
        parts.append(operator)
    if name:
        parts.append(name)
    parts.append("data center")
    parts.append("opened")
    if city:
        parts.append(city)
    if state:
        parts.append(state)
    query = " ".join(p for p in parts if p).strip()
    if not query or query.startswith("data center opened"):
        return None, ""
    results = bing_search(query)
    text_blob = " ".join(f"{t}. {s}" for t, s in results)
    anchor = name if name and len(name) > 3 else operator
    y = best_year_from_text(text_blob, anchor=anchor)
    if y is not None:
        return y, "web_search:bing"
    return None, ""


# Orchestration

def enrich_row(row: pd.Series, *, do_web: bool = True) -> dict[str, Any]:
    osm_id = str(row.get("osm_id"))
    cache_key = f"row:{osm_id}"
    if cache_key in CACHE:
        return CACHE[cache_key]

    y, m = pass_a_osm_tags(row)
    if y is not None:
        out = {"opening_year": y, "method": m, "confidence": "high"}
        CACHE[cache_key] = out
        return out

    name = (row.get("name") or "")
    operator = (row.get("operator") or "")
    city = (row.get("city") or "")
    state = (row.get("state") or "")
    name_str = str(name) if pd.notna(name) else ""
    op_str = str(operator) if pd.notna(operator) else ""
    city_str = str(city) if pd.notna(city) else ""
    state_str = str(state) if pd.notna(state) else ""

    y, m = pass_b0_wikidata_index(name_str, op_str or None)
    if y is not None:
        out = {"opening_year": y, "method": m, "confidence": "high"}
        CACHE[cache_key] = out
        return out

    y, m = pass_b1_hyperscaler_table(name_str, op_str or None, city_str or None)
    if y is not None:
        out = {"opening_year": y, "method": m, "confidence": "high"}
        CACHE[cache_key] = out
        return out

    q = get_wikidata_qcode(row)
    if q:
        y, m = pass_b_wikidata(q)
        if y is not None:
            out = {"opening_year": y, "method": m, "confidence": "high"}
            CACHE[cache_key] = out
            return out

    if name_str:
        y, m = pass_c_wikipedia(name_str, op_str or None)
        if y is not None:
            out = {"opening_year": y, "method": m, "confidence": "medium"}
            CACHE[cache_key] = out
            return out

    if op_str and city_str:
        y, m = pass_c2_wikipedia_city_anchored(name_str, op_str, city_str)
        if y is not None:
            out = {"opening_year": y, "method": m, "confidence": "medium"}
            CACHE[cache_key] = out
            return out

    if do_web and (name_str or (op_str and city_str)):
        y, m = pass_d0_baxtel(name_str, op_str or None, city_str or None)
        if y is not None:
            out = {"opening_year": y, "method": m, "confidence": "medium"}
            CACHE[cache_key] = out
            return out

    if do_web and name_str:
        y, m = pass_d_web(name_str, op_str or None, city_str or None, state_str or None)
        if y is not None:
            out = {"opening_year": y, "method": m, "confidence": "low"}
            CACHE[cache_key] = out
            return out

    out = {"opening_year": None, "method": "not_found", "confidence": "none"}
    CACHE[cache_key] = out
    return out


def main(limit: int | None = None, workers: int = 8, do_web: bool = True) -> int:
    df = pd.read_csv(RAW / "osm_data_centers.csv")
    load_cache()
    print(f"Loaded {len(df)} data center rows; cache has {len(CACHE)} entries.")
    load_wikidata_index()
    save_cache()
    print(f"Wikidata SPARQL index ready: {len(WD_INDEX)} known data centers.")
    load_hyperscaler_index()
    save_cache()
    n_keys = sum(len(d) for d in HYPERSCALER_INDEX.values())
    print(f"Hyperscaler table index ready: {len(HYPERSCALER_INDEX)} operators, {n_keys} location keys.")

    rows = df.head(limit) if limit else df
    results: dict[str, dict[str, Any]] = {}

    def worker(rec):
        idx, r = rec
        try:
            return idx, enrich_row(r, do_web=do_web)
        except Exception as e:
            return idx, {"opening_year": None, "method": f"error:{type(e).__name__}", "confidence": "none"}

    save_every = 50
    pending = list(rows.iterrows())
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(worker, rec): rec[0] for rec in pending}
        for fut in as_completed(futures):
            idx, res = fut.result()
            results[idx] = res
            done += 1
            if done % save_every == 0:
                save_cache()
                print(f"  ... {done}/{len(pending)} done, cached {len(CACHE)}")
    save_cache()

    rows = rows.copy()
    rows["opening_year"] = [results.get(i, {}).get("opening_year") for i in rows.index]
    rows["opening_year_method"] = [results.get(i, {}).get("method") for i in rows.index]
    rows["opening_year_confidence"] = [results.get(i, {}).get("confidence") for i in rows.index]

    out_path = OUT / "data_centers_with_year.csv"
    rows.to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} rows -> {out_path}")

    found = rows["opening_year"].notna().sum()
    print(f"\nOPENING-YEAR DISCOVERY SUMMARY:")
    print(f"  Total facilities: {len(rows)}")
    print(f"  Year found:       {found} ({found/len(rows)*100:.1f}%)")
    print(f"  No year:          {len(rows) - found}")
    summary = (
        rows.groupby("opening_year_method", dropna=False)["osm_id"].count()
        .rename("n_facilities").reset_index()
        .sort_values("n_facilities", ascending=False)
    )
    summary.to_csv(OUT / "opening_year_methods_summary.csv", index=False)
    print("\nMethod breakdown:")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    workers = 8
    limit = None
    do_web = True
    args = sys.argv[1:]
    for a in args:
        if a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])
        elif a.startswith("--workers="):
            workers = int(a.split("=", 1)[1])
        elif a == "--no-web":
            do_web = False
    raise SystemExit(main(limit=limit, workers=workers, do_web=do_web))
