"""Canonical normalization for the FusEurope Web of Science corpus.

This module deliberately keeps *raw*, *normalized*, and *analysis* values separate.
No function in this file performs global string replacements on a dataframe.
That design prevents country-name substitutions (e.g. ENGLAND -> UK) from
accidentally changing author surnames such as ENGLAND, A.C.

The two principal tasks are:
1. conservative recovery of old WoS dot-truncated author surnames;
2. affiliation-level country normalization, including the special JET node used
   in the historical co-country analysis.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import re
import unicodedata

import pandas as pd


# ---------------------------------------------------------------------------
# Generic text helpers
# ---------------------------------------------------------------------------

def _ascii_upper(text: str) -> str:
    text = "" if text is None else str(text)
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )
    return text.upper().strip()


def _letters_only(text: str) -> str:
    return re.sub(r"[^A-Z]", "", _ascii_upper(text))


def _surname_letters(text: str) -> str:
    return re.sub(r"[^A-Z-]", "", _ascii_upper(text))


def _initial_letters(text: str) -> str:
    return re.sub(r"[^A-Z]", "", _ascii_upper(text))


# ---------------------------------------------------------------------------
# Author normalization and old WoS dot-truncation recovery
# ---------------------------------------------------------------------------

DOT_TRUNCATED_AUTHOR_RE = re.compile(r"^([^,.]+)\.([A-Za-z]{1,6})$")


def parse_author_form(name: str) -> dict:
    """Parse an AU author form without claiming identity equivalence.

    Old WoS exports in this corpus often contain forms such as ``RUTHERFO.PH``.
    These are marked as ``dot_truncated`` rather than silently expanded.
    """
    raw = "" if name is None else str(name).strip()
    m = DOT_TRUNCATED_AUTHOR_RE.match(raw)
    if m:
        surname = _surname_letters(m.group(1))
        initials = _initial_letters(m.group(2))
        return {
            "raw": raw,
            "surname": surname,
            "initials": initials,
            "form": "dot_truncated",
            "standardized": f"{surname}, {initials}" if initials else surname,
        }

    if "," in raw:
        left, right = raw.split(",", 1)
        surname = _surname_letters(left)
        initials = _initial_letters(right)
    else:
        tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'-]+", raw)
        if not tokens:
            return {"raw": raw, "surname": "", "initials": "", "form": "empty", "standardized": ""}
        surname = _surname_letters(tokens[0])
        initials = "".join(_initial_letters(t) for t in tokens[1:])

    return {
        "raw": raw,
        "surname": surname,
        "initials": initials,
        "form": "regular",
        "standardized": f"{surname}, {initials}" if initials else surname,
    }


def _canonical_author_key(name: str) -> Tuple[str, str]:
    p = parse_author_form(name)
    return p["surname"].replace("-", ""), p["initials"]


def build_truncated_author_resolution(records: Sequence[Mapping]) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Build a conservative map for dot-truncated WoS author forms.

    A truncated form is expanded automatically only when the entire corpus
    contains exactly one *canonical surname* that:
      - begins with the truncated surname fragment,
      - is longer than that fragment, and
      - has identical initials.

    Case variants of the same surname are collapsed before uniqueness is tested.
    Ambiguous and unmatched forms are never guessed; they are returned in the
    audit table for manual review or later context-based disambiguation.
    """
    observed = Counter()
    for r in records:
        for a in r.get("authors_wos", []) or []:
            observed[str(a).strip()] += 1

    regular_by_initials: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    truncated = []

    for name, count in observed.items():
        p = parse_author_form(name)
        if p["form"] == "dot_truncated":
            truncated.append((name, count, p))
        elif p["surname"]:
            regular_by_initials[p["initials"]][p["surname"].replace("-", "")][p["standardized"]] += count

    mapping: Dict[str, str] = {}
    rows = []

    for raw, count, p in sorted(truncated, key=lambda x: x[0]):
        frag = p["surname"].replace("-", "")
        candidate_keys = []
        for surname_key in regular_by_initials.get(p["initials"], {}):
            if surname_key.startswith(frag) and len(surname_key) > len(frag):
                candidate_keys.append(surname_key)
        candidate_keys = sorted(set(candidate_keys))

        candidate_labels = []
        for key in candidate_keys:
            variants = regular_by_initials[p["initials"]][key]
            # Prefer the most frequent standardized spelling; ties are deterministic.
            label = sorted(variants.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            candidate_labels.append(label)

        if len(candidate_keys) == 1:
            status = "auto_unique_prefix"
            resolved = candidate_labels[0]
            mapping[raw] = resolved
        elif len(candidate_keys) == 0:
            status = "unresolved_no_full_form"
            resolved = p["standardized"]
        else:
            status = "review_multiple_candidates"
            resolved = p["standardized"]

        rows.append({
            "raw_truncated_form": raw,
            "occurrences": count,
            "truncated_surname": p["surname"],
            "initials": p["initials"],
            "status": status,
            "resolved_or_standardized": resolved,
            "n_candidate_surnames": len(candidate_keys),
            "candidate_surnames": " | ".join(candidate_keys),
            "candidate_labels": " | ".join(candidate_labels),
        })

    cols = [
        "raw_truncated_form", "occurrences", "truncated_surname", "initials",
        "status", "resolved_or_standardized", "n_candidate_surnames",
        "candidate_surnames", "candidate_labels",
    ]
    return mapping, pd.DataFrame(rows, columns=cols)


def normalize_author_list(authors_wos: Sequence[str], truncation_map: Mapping[str, str] | None = None) -> Tuple[List[str], List[str]]:
    """Normalize one AU list while preserving author count and order.

    Returns ``(normalized_authors, normalization_statuses)``.
    """
    truncation_map = truncation_map or {}
    normalized, statuses = [], []
    for raw in authors_wos or []:
        raw = str(raw).strip()
        p = parse_author_form(raw)
        if raw in truncation_map:
            normalized.append(truncation_map[raw])
            statuses.append("dot_truncation_recovered")
        else:
            normalized.append(p["standardized"])
            if p["form"] == "dot_truncated":
                statuses.append("dot_truncation_unresolved")
            else:
                statuses.append("format_standardized")
    return normalized, statuses


def add_author_normalization(records: Sequence[Mapping]) -> Tuple[List[dict], pd.DataFrame]:
    """Add traceable author-normalization fields to parsed records."""
    trunc_map, audit = build_truncated_author_resolution(records)
    out = []
    for r in records:
        rr = dict(r)
        norm, statuses = normalize_author_list(rr.get("authors_wos", []), trunc_map)
        rr["authors_normalized"] = norm
        rr["author_normalization_status"] = statuses
        rr["n_authors_normalized"] = len(norm)
        out.append(rr)
    return out, audit


# ---------------------------------------------------------------------------
# Affiliation and country normalization
# ---------------------------------------------------------------------------

BRACKET_PREFIX_RE = re.compile(r"^\s*\[[^\]]+\]\s*")
# Historical RP values commonly begin with a person followed by a role marker,
# for example ``Smith, AB (reprint author), JET Joint Undertaking, ...``.  That
# prefix must be removed before institution, country, and JET recognition.
RP_AUTHOR_PREFIX_RE = re.compile(
    r"^\s*.*?\(\s*(?:reprint|corresponding)(?:\s*/\s*(?:reprint|corresponding))?\s+author\s*\)\s*[,;:]?\s*",
    re.I,
)
RP_ROLE_PREFIX_RE = re.compile(
    r"^\s*(?:reprint|corresponding)(?:\s*/\s*(?:reprint|corresponding))?\s+author\s*[:;,]\s*",
    re.I,
)

# Matching is done against the full affiliation string, longest/specific aliases first.
# The normalized labels preserve historically meaningful state changes, including
# USSR/successor-state and CZECHOSLOVAKIA/CZECH REPUBLIC distinctions.  The
# publication corpus has annual (not day-level) dates, so former Soviet republics
# are represented as USSR through 1991 and as successor states from 1992.
# Explicit Soviet-era labels map to USSR through 1991. From 1992, legacy USSR labels are reassigned only when a successor state can be inferred from an explicit country or location token.
SOVIET_SUCCESSOR_START_YEAR = 1992
SOVIET_SUCCESSOR_STATES = {
    "RUSSIA", "UKRAINE", "BELARUS", "MOLDOVA", "GEORGIA", "ARMENIA",
    "AZERBAIJAN", "KAZAKHSTAN", "KYRGYZSTAN", "UZBEKISTAN",
    "TAJIKISTAN", "TURKMENISTAN", "ESTONIA", "LATVIA", "LITHUANIA",
}

COUNTRY_ALIASES: List[Tuple[str, str]] = [
    ("NORTHERN IRELAND", "UK"), ("N IRELAND", "UK"), ("UNITED KINGDOM", "UK"),
    ("GREAT BRITAIN", "UK"), ("ENGLAND", "UK"), ("SCOTLAND", "UK"), ("WALES", "UK"),
    ("UNITED STATES", "USA"), ("U.S.A.", "USA"), ("USA", "USA"),
    ("EAST GERMANY", "EAST GERMANY"), ("GERMAN DEMOCRATIC REPUBLIC", "EAST GERMANY"),
    ("FED REP GER", "GERMANY"), ("FEDERAL REPUBLIC OF GERMANY", "GERMANY"),
    ("WEST GERMANY", "GERMANY"), ("W GERMANY", "GERMANY"), ("GERMANY", "GERMANY"),
    ("CZECHOSLOVAKIA", "CZECHOSLOVAKIA"), ("CZECH REPUBLIC", "CZECH REPUBLIC"),
    ("SOUTH KOREA", "KOREA"), ("REPUBLIC OF KOREA", "KOREA"), ("KOREA", "KOREA"),
    ("PEOPLES R CHINA", "CHINA"), ("PEOPLES REP CHINA", "CHINA"), ("P R CHINA", "CHINA"),
    ("PEOPLE'S REPUBLIC OF CHINA", "CHINA"), ("CHINA", "CHINA"),
    ("RUSSIAN FEDERATION", "RUSSIA"), ("RUSSIA", "RUSSIA"), ("USSR", "USSR"),
    # Historical Soviet-republic labels. These always denote the USSR state layer,
    # irrespective of the modern successor-country name that may also occur in text.
    ("RUSSIAN SFSR", "USSR"), ("RSFSR", "USSR"),
    ("UKRAINIAN SSR", "USSR"), ("UKRAINIAN S.S.R.", "USSR"), ("UKR SSR", "USSR"), ("UKSSR", "USSR"),
    ("BYELORUSSIAN SSR", "USSR"), ("BELORUSSIAN SSR", "USSR"), ("BSSR", "USSR"), ("BESSR", "USSR"),
    ("UZBEK SSR", "USSR"), ("UZSSR", "USSR"),
    ("KAZAKH SSR", "USSR"), ("KAZSSR", "USSR"),
    ("GEORGIAN SSR", "USSR"), ("GSSR", "USSR"), ("GESSR", "USSR"),
    ("AZERBAIJAN SSR", "USSR"), ("AZSSR", "USSR"),
    ("LITHUANIAN SSR", "USSR"), ("LITSSR", "USSR"),
    ("MOLDAVIAN SSR", "USSR"), ("MOLDOVAN SSR", "USSR"), ("MSSR", "USSR"),
    ("LATVIAN SSR", "USSR"), ("LATSSR", "USSR"), ("LASSR", "USSR"),
    ("KIRGHIZ SSR", "USSR"), ("KYRGYZ SSR", "USSR"), ("KIRSSR", "USSR"),
    ("TAJIK SSR", "USSR"), ("TADZHIK SSR", "USSR"), ("TAJSSR", "USSR"),
    ("ARMENIAN SSR", "USSR"), ("ARMSSR", "USSR"), ("ARSSR", "USSR"),
    ("TURKMEN SSR", "USSR"), ("TURKMENIAN SSR", "USSR"), ("TURSSR", "USSR"),
    ("ESTONIAN SSR", "USSR"), ("ESSR", "USSR"),
    ("SOVIET UNION", "USSR"), ("GER DEM REP", "EAST GERMANY"), ("BUNDES REPUBLIK", "GERMANY"),
    ("THE NETHERLANDS", "NETHERLANDS"), ("NETHERLANDS", "NETHERLANDS"), ("NETH.", "NETHERLANDS"),
    ("SWITZERLAND", "SWITZERLAND"), ("AUSTRALIA", "AUSTRALIA"), ("BRAZIL", "BRAZIL"),
    ("NEW ZEALAND", "NEW ZEALAND"), ("SOUTH AFRICA", "SOUTH AFRICA"),
    ("SAUDI ARABIA", "ARABIA"), ("UNITED ARAB EMIRATES", "EMIRATES"),
    ("TAIWAN", "TAIWAN"), ("HONG KONG", "KONG"),
    ("BELORUSSIA", "BELARUS"), ("BELARUS", "BELARUS"), ("UKRAINE", "UKRAINE"),
    ("KAZAKHSTAN", "KAZAKHSTAN"), ("KYRGYZSTAN", "KYRGYZSTAN"),
    ("UZBEKISTAN", "UZBEKISTAN"), ("ARMENIA", "ARMENIA"),
    ("AZERBAIJAN", "AZERBAIJAN"), ("MOLDOVA", "MOLDOVA"), ("MOLDAVIA", "MOLDOVA"),
    ("TAJIKISTAN", "TAJIKISTAN"), ("TURKMENISTAN", "TURKMENISTAN"),
    ("LATVIA", "LATVIA"), ("LITHUANIA", "LITHUANIA"),
    ("ESTONIA", "ESTONIA"), ("SLOVAKIA", "SLOVAKIA"), ("SLOVENIA", "SLOVENIA"),
    ("SERBIA", "SERBIA"), ("MONTENEGRO", "MONTENEGRO"), ("YUGOSLAVIA", "YUGOSLAVIA"),
    ("MACEDONIA", "MACEDONIA"),
    ("FRANCE", "FRANCE"), ("ITALY", "ITALY"), ("JAPAN", "JAPAN"), ("INDIA", "INDIA"),
    ("CANADA", "CANADA"), ("SPAIN", "SPAIN"), ("BELGIUM", "BELGIUM"),
    ("SWEDEN", "SWEDEN"), ("PORTUGAL", "PORTUGAL"), ("FINLAND", "FINLAND"),
    ("ISRAEL", "ISRAEL"), ("AUSTRIA", "AUSTRIA"), ("IRAN", "IRAN"),
    ("POLAND", "POLAND"), ("IRELAND", "IRELAND"), ("PAKISTAN", "PAKISTAN"),
    ("ROMANIA", "ROMANIA"), ("ARGENTINA", "ARGENTINA"), ("GREECE", "GREECE"),
    ("DENMARK", "DENMARK"), ("MEXICO", "MEXICO"), ("HUNGARY", "HUNGARY"),
    ("TURKEY", "TURKEY"), ("UNITED ARAB REP", "EGYPT"), ("U A R", "EGYPT"), ("EGYPT", "EGYPT"), ("BULGARIA", "BULGARIA"),
    ("SINGAPORE", "SINGAPORE"), ("NORWAY", "NORWAY"), ("BANGLADESH", "BANGLADESH"),
    ("ALGERIA", "ALGERIA"), ("CHILE", "CHILE"), ("VENEZUELA", "VENEZUELA"),
    ("MALAYSIA", "MALAYSIA"), ("MOROCCO", "MOROCCO"), ("THAILAND", "THAILAND"),
    ("ZIMBABWE", "ZIMBABWE"), ("TUNISIA", "TUNISIA"), ("JORDAN", "JORDAN"),
    ("LIBYA", "LIBYA"), ("SUDAN", "SUDAN"), ("MYANMAR", "MYANMAR"),
    ("NEPAL", "NEPAL"), ("ICELAND", "ICELAND"), ("CAMEROON", "CAMEROON"),
    ("COLOMBIA", "COLOMBIA"), ("CYPRUS", "CYPRUS"), ("CROATIA", "CROATIA"), ("BOSNIA & HERCEG", "BOSNIA & HERZEGOVINA"), ("SYRIA", "SYRIA"), ("BURKINA FASO", "BURKINA FASO"), ("TOGO", "TOGO"), ("BARBADOS", "BARBADOS"),
    ("TANZANIA", "TANZANIA"), ("NIGERIA", "NIGERIA"), ("ALBANIA", "ALBANIA"),
    ("LIECHTENSTEIN", "LIECHTENSTEIN"), ("PHILIPPINES", "PHILIPPINES"),
    ("SRI LANKA", "LANKA"), ("ETHIOPIA", "ETHIOPIA"), ("BOLIVIA", "BOLIVIA"),
]


def strip_c1_author_prefix(affiliation: str) -> str:
    return BRACKET_PREFIX_RE.sub("", "" if affiliation is None else str(affiliation).strip())


def strip_rp_author_prefix(affiliation: str) -> str:
    """Strip a leading WoS RP corresponding/reprint-author label."""
    original = "" if affiliation is None else str(affiliation).strip()
    cleaned = RP_AUTHOR_PREFIX_RE.sub("", original)
    cleaned = RP_ROLE_PREFIX_RE.sub("", cleaned)
    return cleaned.strip(" ,;:")


def strip_address_prefix(affiliation: str) -> str:
    """Normalize field-specific leading labels without altering the address."""
    return strip_c1_author_prefix(strip_rp_author_prefix(affiliation))


def institution_from_affiliation(affiliation: str) -> str:
    cleaned = strip_address_prefix(affiliation)
    return cleaned.split(",", 1)[0].strip() if cleaned else ""


def normalize_country_from_affiliation(affiliation: str, allow_us_state_heuristic: bool = True, year: int | None = None) -> Tuple[str | None, str]:
    """Return ``(country, method)`` from one raw C1 affiliation.

    The function is deliberately conservative. If no explicit country token is
    identifiable, the country is left missing for later enrichment/overrides.
    """
    text = _ascii_upper(strip_address_prefix(affiliation))
    if not text:
        return None, "missing_affiliation"

    # Search only in the address tail (last ~80 chars) first to avoid matching
    # country words that happen to occur in institution names.
    tail = text[-100:]

    # Some 1992 WoS records still carry a legacy USSR country label although the
    # publication year is post-dissolution. For 1992 onward, use an explicit
    # successor-state token when present; otherwise infer Russia only from
    # unambiguous Russian city tokens found in this legacy-address subset. If a
    # successor cannot be supported, leave the country unresolved rather than
    # assigning the defunct USSR state.
    if year is not None and int(year) >= SOVIET_SUCCESSOR_START_YEAR and re.search(r"(?<![A-Z])(?:USSR|SOVIET UNION|RSFSR|[A-Z]+\s+SSR)(?![A-Z])", text):
        explicit_successors = [
            ("UKRAINE", "UKRAINE"), ("BELARUS", "BELARUS"), ("BELORUSSIA", "BELARUS"),
            ("LITHUANIA", "LITHUANIA"), ("LATVIA", "LATVIA"), ("ESTONIA", "ESTONIA"),
            ("GEORGIA", "GEORGIA"), ("ARMENIA", "ARMENIA"), ("AZERBAIJAN", "AZERBAIJAN"),
            ("KAZAKHSTAN", "KAZAKHSTAN"), ("KYRGYZSTAN", "KYRGYZSTAN"),
            ("UZBEKISTAN", "UZBEKISTAN"), ("TAJIKISTAN", "TAJIKISTAN"),
            ("TURKMENISTAN", "TURKMENISTAN"), ("MOLDOVA", "MOLDOVA"), ("MOLDAVIA", "MOLDOVA"),
            ("RUSSIA", "RUSSIA"),
        ]
        for token, successor in explicit_successors:
            if re.search(r"(?<![A-Z])" + re.escape(token) + r"(?![A-Z])", text):
                return successor, f"post_soviet_legacy:{token}"
        russian_location_tokens = [
            "MOSCOW", "ST PETERSBURG", "LENINGRAD", "NOVOSIBIRSK", "SVERDLOVSK",
            "NIZHNYJ NOVGOROD", "NIZHNY NOVGOROD", "TROISK", "OBNINSK",
        ]
        for token in russian_location_tokens:
            if re.search(r"(?<![A-Z])" + re.escape(token) + r"(?![A-Z])", text):
                return "RUSSIA", f"post_soviet_legacy_city:{token}"
        return None, "post_soviet_legacy_unresolved"
    for alias, canonical in COUNTRY_ALIASES:
        # Require non-letter boundaries. This avoids false matches such as
        # USA inside CHIKUSA or IRAN inside an institution/personal name.
        pat = r"(?<![A-Z])" + re.escape(alias) + r"(?![A-Z])"
        if re.search(pat, tail):
            if canonical in SOVIET_SUCCESSOR_STATES and year is not None and int(year) < SOVIET_SUCCESSOR_START_YEAR:
                return "USSR", f"historical_successor:{alias}->USSR"
            return canonical, f"alias:{alias}"

    # GEORGIA is intentionally not a broad alias: otherwise institution names such
    # as GEORGIA INST TECHNOL in Atlanta are misclassified as the country Georgia.
    # Treat it as a country only when it occurs as an address-tail token.
    if re.search(r",\s*GEORGIA\.?$", text):
        if year is not None and int(year) < SOVIET_SUCCESSOR_START_YEAR:
            return "USSR", "historical_successor:GEORGIA->USSR"
        return "GEORGIA", "tail_country:GEORGIA"

    # One historical WoS form abbreviates Sukhumi, Georgia as `SUKHUMI, GA.`.
    # This must be resolved before the U.S. state heuristic, where GA means Georgia
    # (the U.S. state).  The publication year determines USSR vs successor Georgia.
    if re.search(r"\bSUKHUMI\s*,\s*GA\.?$", text):
        if year is not None and int(year) < SOVIET_SUCCESSOR_START_YEAR:
            return "USSR", "historical_sukhumi_ga:USSR"
        return "GEORGIA", "historical_sukhumi_ga:GEORGIA"

    # Some older U.S. records omit USA and end with a two-letter state + ZIP.
    us_states_zip = r"AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC"
    us_states_no_ga = r"AL|AK|AZ|AR|CA|CO|CT|DE|FL|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC"
    if allow_us_state_heuristic and re.search(r",\s*(?:" + us_states_zip + r")\s+\d{4,10}(?:-\d{4})?\.?$", text):
        return "USA", "us_state_zip"
    # A sizeable block of older WoS records ends with the state abbreviation alone.
    # GA is intentionally excluded here because the corpus contains a Sukhumi, GA
    # address referring to the Georgian SSR rather than the U.S. state.
    if allow_us_state_heuristic and re.search(r",\s*(?:" + us_states_no_ga + r")\.?$", text):
        return "USA", "us_state_only"
    if allow_us_state_heuristic and re.search(r",\s*PR\s+\d{5}\.?$", text):
        return "USA", "us_puerto_rico"

    return None, "unresolved"


# JET is intentionally an analysis node in the same network layer as countries.
# Do not use a broad `JET` substring test because Jet Propulsion Laboratory is
# unrelated to the Joint European Torus.
JET_START_PATTERNS = [
    re.compile(r"^JET(?:\s*,\s*|\s+)JOINT UNDERTAKING\b", re.I),
    re.compile(r"^JET\s*,", re.I),
    re.compile(r"^JET EUROPEAN TOKAMAK JOINT UNDERTAKING\b", re.I),
    re.compile(r"^JET TEAM\b", re.I),
    re.compile(r"^JET DESIGN GRP\b", re.I),
    re.compile(r"^JET CULHAM\b", re.I),
    re.compile(r"^JET FACIL(?:ITY)?\b", re.I),
    re.compile(r"^UKAEA JET FACIL(?:ITY)?\b", re.I),
    re.compile(r"^EFDA JET\b", re.I),
    re.compile(r"^EFDA CSU JET\b", re.I),
    re.compile(r"^EUROPEAN FUS DEV AGREEMENT JET\b", re.I),
    re.compile(r"^\[JET-FEDA CONTRIBUTORS\]\s*JET EFDA\b", re.I),
]


def is_jet_affiliation(affiliation: str) -> bool:
    original = "" if affiliation is None else str(affiliation).strip()
    cleaned = strip_address_prefix(original)
    upper = cleaned.upper()
    if "JET PROP" in upper or "JET PROPULS" in upper or "AEROJET" in upper:
        return False
    # Preserve the special bracketed contributor form before prefix stripping.
    if re.match(r"^\[JET-FEDA CONTRIBUTORS\]\s*JET EFDA\b", original, re.I):
        return True
    if any(p.search(cleaned) for p in JET_START_PATTERNS):
        return True
    # Culham/EFDA support-unit forms where JET occurs after the first institution token.
    if ("CULHAM" in upper or "EFDA" in upper) and re.search(r"\bJET\b", upper):
        return True
    return False


def build_affiliation_table(records: Sequence[Mapping]) -> pd.DataFrame:
    """Build the within-record deduplicated C1 union RP address table."""
    rows = []
    for r in records:
        rid = r.get("record_id")
        candidates = [("C1", aff) for aff in (r.get("affiliations", []) or [])]
        rp = r.get("reprint_address_one_line") or r.get("reprint_address")
        if rp:
            candidates.append(("RP", rp))
        seen: set[str] = set()
        for source, aff in candidates:
            cleaned = strip_address_prefix(aff)
            dedup_key = re.sub(r"\s+", " ", _ascii_upper(cleaned)).strip(" .")
            if not dedup_key or dedup_key in seen:
                continue
            seen.add(dedup_key)
            institution = institution_from_affiliation(cleaned)
            country, method = normalize_country_from_affiliation(cleaned, year=r.get("year"))
            jet = is_jet_affiliation(cleaned)
            network_country = "JET" if jet else country
            rows.append({
                "record_id": rid,
                "year": r.get("year"),
                "affiliation_no": len(seen),
                "address_source": source,
                "affiliation_raw": aff,
                "affiliation_clean": cleaned,
                "rp_prefix_stripped": bool(source == "RP" and cleaned != str(aff).strip()),
                "institution_raw": institution,
                "country_normalized": country,
                "country_normalization_method": method,
                "is_jet_affiliation": jet,
                "network_country": network_country,
            })
    return pd.DataFrame(rows)


def unique_preserve_order(values: Iterable[str | None]) -> List[str]:
    seen = set(); out = []
    for v in values:
        if v is None or v == "":
            continue
        if v not in seen:
            seen.add(v); out.append(v)
    return out


def add_country_lists_to_records(records: Sequence[Mapping], affiliation_df: pd.DataFrame) -> List[dict]:
    grouped = {}
    if not affiliation_df.empty:
        for rid, g in affiliation_df.groupby("record_id", sort=False):
            g = g.sort_values("affiliation_no")
            c1 = g[g["address_source"].eq("C1")]
            grouped[rid] = {
                "institutions_normalized": unique_preserve_order(g["institution_raw"].tolist()),
                # Identity evidence stays on the original C1 representation.
                # RP is added for final geographic reconstruction but must not
                # create new institutional evidence for author merging.
                "identity_institutions_normalized": unique_preserve_order(c1["institution_raw"].tolist()),
                "countries_normalized": unique_preserve_order(g["country_normalized"].tolist()),
                "countries_network": unique_preserve_order(g["network_country"].tolist()),
                "n_affiliations": int(len(g)),
                "n_affiliations_with_country": int(g["country_normalized"].notna().sum()),
                "has_jet_affiliation": bool(g["is_jet_affiliation"].any()),
            }
    out = []
    for r in records:
        rr = dict(r)
        vals = grouped.get(r.get("record_id"), {
            "institutions_normalized": [], "identity_institutions_normalized": [],
            "countries_normalized": [], "countries_network": [], "n_affiliations": 0,
            "n_affiliations_with_country": 0, "has_jet_affiliation": False,
        })
        rr.update(vals)
        rp = rr.get("reprint_address_one_line") or rr.get("reprint_address")
        rp_country, rp_method = normalize_country_from_affiliation(rp or "", year=rr.get("year"))
        rr["reprint_country_normalized"] = rp_country
        rr["reprint_country_normalization_method"] = rp_method
        out.append(rr)
    return out
