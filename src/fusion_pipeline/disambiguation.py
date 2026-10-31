"""Conservative author-variant disambiguation for the FusEurope corpus.

The module is designed for reproducible publication use rather than maximal merging.
It distinguishes three operations:

1. formatting/truncation normalization (phase 2; already applied upstream);
2. conservative automatic merging of author *variants* when name compatibility and
   contextual evidence are strong;
3. explicit manual overrides for a small set of historically important ambiguous cases.

Identical standardized author strings are treated as the same graph node, as in the
historical analyses. This module does not attempt to split homonyms that are genuinely
indistinguishable in the WoS metadata; that limitation should be reported explicitly.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
import ast
import json
import math
import re

import pandas as pd


# ---------------------------------------------------------------------------
# Serialization / parsing helpers
# ---------------------------------------------------------------------------

def parse_list_cell(value) -> List[str]:
    """Parse a list-valued CSV cell without using ``eval``."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x).strip() for x in value if str(x).strip()]
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    # Fallback only for simple delimiter-separated cells.
    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]
    return [text]


def serialize_list(values: Sequence[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def safe_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


# ---------------------------------------------------------------------------
# Name representation and conservative compatibility rules
# ---------------------------------------------------------------------------

def split_standard_author(name: str) -> Tuple[str, str]:
    text = safe_text(name).upper().replace(".", " ")
    if "," in text:
        left, right = text.split(",", 1)
        surname = re.sub(r"[^A-Z-]", "", left)
        initials = re.sub(r"[^A-Z]", "", right)
    else:
        toks = re.findall(r"[A-Z-]+", text)
        surname = re.sub(r"[^A-Z-]", "", toks[0]) if toks else ""
        initials = "".join(re.sub(r"[^A-Z]", "", t) for t in toks[1:]) if len(toks) > 1 else ""
    return surname.strip("-"), initials


def simplify_surname_for_matching(surname: str) -> str:
    """Normalize common transliteration variants without declaring identity.

    These replacements reproduce the historically useful logic from the diagnostic
    scripts, but are used only to *generate/score candidates*. A match on this key is
    never sufficient on its own for automatic merging.
    """
    s = re.sub(r"[^A-Z]", "", safe_text(surname).upper())
    replacements = [
        ("TSCH", "CH"), ("TCH", "CH"), ("SCH", "SH"), ("ZH", "J"),
        ("KH", "H"), ("YA", "IA"), ("YU", "IU"), ("YO", "IO"),
        ("AJ", "AI"), ("J", "I"), ("SKAYA", "SKAIA"),
        ("SKAJA", "SKAIA"), ("SKII", "SKI"), ("SKIY", "SKI"),
        ("SKIJ", "SKI"), ("SKY", "SKI"), ("IY", "I"),
        ("YI", "I"), ("II", "I"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    if s.endswith("Y"):
        s = s[:-1] + "I"
    s = re.sub(r"(.)\1+", r"\1", s)
    return s


def surname_features(a: str, b: str) -> dict:
    ka, kb = simplify_surname_for_matching(a), simplify_surname_for_matching(b)
    if not ka or not kb:
        return {"mode": None, "ratio": 0.0, "key_a": ka, "key_b": kb}
    ratio = SequenceMatcher(None, ka, kb).ratio()
    raw_a = re.sub(r"[^A-Z]", "", safe_text(a).upper())
    raw_b = re.sub(r"[^A-Z]", "", safe_text(b).upper())
    raw_ratio = SequenceMatcher(None, raw_a, raw_b).ratio() if raw_a and raw_b else 0.0
    if a == b:
        mode = "exact"
    # The transliteration key is intentionally not trusted for very short surnames.
    # Global phonetic replacements can otherwise make distinct East-Asian surnames
    # (e.g. YANG/ZHANG) look identical.
    elif ka == kb and min(len(raw_a), len(raw_b)) >= 5 and raw_ratio >= 0.75:
        mode = "normalized_key"
    elif (ka.startswith(kb) or kb.startswith(ka)) and min(len(ka), len(kb)) >= 6:
        mode = "prefix"
    elif ratio >= 0.92 and min(len(ka), len(kb)) >= 6:
        mode = "similar"
    else:
        mode = None
    return {"mode": mode, "ratio": ratio, "raw_ratio": raw_ratio, "key_a": ka, "key_b": kb}


def initials_mode(a: str, b: str) -> str | None:
    if a == b and a:
        return "exact"
    if not a or not b:
        return None
    if a[0] != b[0]:
        return None
    if a.startswith(b) or b.startswith(a):
        return "prefix"
    # Same first initial but incompatible later initials are retained as an explicit
    # conflict category for manual review; never auto-merged.
    return "conflict"


def _institution_context_key(value: str) -> str:
    s = safe_text(value).upper()
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def _records_from_dataframe(df: pd.DataFrame) -> List[dict]:
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        for col in [
            "authors_normalized", "authors_wos", "authors_full_wos", "institutions", "institutions_normalized",
            "identity_institutions_normalized",
            "affiliations", "countries_normalized", "countries_network",
        ]:
            if col in d:
                d[col] = parse_list_cell(d[col])
        rows.append(d)
    return rows


def build_author_profiles(records: Sequence[Mapping], authors_col: str = "authors_normalized") -> Dict[str, dict]:
    profiles: Dict[str, dict] = {}
    for rec in records:
        authors = [safe_text(x).upper() for x in (rec.get(authors_col) or []) if safe_text(x)]
        year = int(rec.get("year")) if rec.get("year") not in (None, "") and not pd.isna(rec.get("year")) else None
        rid = safe_text(rec.get("record_id"))
        institution_values = (
            rec.get("identity_institutions_normalized")
            if "identity_institutions_normalized" in rec
            else rec.get("institutions_normalized") or rec.get("institutions") or []
        )
        institutions = {_institution_context_key(x) for x in institution_values if _institution_context_key(x)}
        full_forms = rec.get("authors_full_wos") or []
        small_team = len(authors) <= 20

        for pos, author in enumerate(authors):
            surname, initials = split_standard_author(author)
            if not surname:
                continue
            if author not in profiles:
                profiles[author] = {
                    "name": author,
                    "surname": surname,
                    "surname_key": simplify_surname_for_matching(surname),
                    "initials": initials,
                    "paper_count": 0,
                    "years": set(),
                    "record_ids": set(),
                    "coauthors": set(),
                    "coauthors_small_team": set(),
                    "institutions": set(),
                    "institutions_small_team": set(),
                    "full_forms": Counter(),
                }
            p = profiles[author]
            p["paper_count"] += 1
            if year is not None:
                p["years"].add(year)
            if rid:
                p["record_ids"].add(rid)
            p["coauthors"].update(a for a in authors if a != author)
            p["institutions"].update(institutions)
            if small_team:
                p["coauthors_small_team"].update(a for a in authors if a != author)
                p["institutions_small_team"].update(institutions)
            if pos < len(full_forms):
                ff = safe_text(full_forms[pos])
                if ff:
                    p["full_forms"][ff] += 1
    return profiles


def _min_year_gap(pa: Mapping, pb: Mapping) -> int | None:
    ya, yb = pa.get("years", set()), pb.get("years", set())
    if not ya or not yb:
        return None
    return min(abs(int(a) - int(b)) for a in ya for b in yb)


def score_candidate(pa: Mapping, pb: Mapping) -> dict | None:
    sf = surname_features(pa["surname"], pb["surname"])
    if sf["mode"] is None:
        return None
    imode = initials_mode(pa["initials"], pb["initials"])
    if imode is None:
        return None

    shared_coauthors = len(pa["coauthors"] & pb["coauthors"])
    shared_coauthors_small = len(pa["coauthors_small_team"] & pb["coauthors_small_team"])
    shared_institutions = len(pa["institutions"] & pb["institutions"])
    shared_institutions_small = len(pa["institutions_small_team"] & pb["institutions_small_team"])
    cooccurrence_papers = len(pa["record_ids"] & pb["record_ids"])
    year_overlap = bool(pa["years"] & pb["years"])
    min_gap = _min_year_gap(pa, pb)

    # A transparent score is retained for ranking, but automatic merging is governed
    # by explicit logical rules below rather than a score threshold alone.
    score = 0
    reasons = []
    surname_points = {"exact": 55, "normalized_key": 50, "prefix": 42, "similar": 36}[sf["mode"]]
    initials_points = {"exact": 30, "prefix": 15, "conflict": 0}[imode]
    score += surname_points + initials_points
    reasons.extend([f"surname_{sf['mode']}", f"initials_{imode}"])
    if shared_coauthors >= 2:
        score += 24; reasons.append("2+_shared_coauthors")
    elif shared_coauthors == 1:
        score += 14; reasons.append("1_shared_coauthor")
    if shared_institutions >= 2:
        score += 18; reasons.append("2+_shared_institutions")
    elif shared_institutions == 1:
        score += 12; reasons.append("1_shared_institution")
    if shared_coauthors_small >= 2:
        reasons.append("2+_shared_coauthors_small_team")
    elif shared_coauthors_small == 1:
        reasons.append("1_shared_coauthor_small_team")
    if shared_institutions_small >= 1:
        reasons.append("shared_institution_small_team")
    if cooccurrence_papers:
        reasons.append("cooccur_same_paper")
    if year_overlap:
        score += 5; reasons.append("year_overlap")
    elif min_gap is not None and min_gap <= 3:
        score += 2; reasons.append("year_gap_le_3")

    # Conservative automatic rules. Large-team papers are deliberately not allowed
    # to provide the decisive coauthor evidence, because a single JET/ITER-style paper
    # can create hundreds of shared coauthors. Paper-level institutions are also treated
    # as supporting rather than sufficient evidence.
    auto = False
    rule = ""
    if cooccurrence_papers > 0:
        plausible_review = False
        rule = "BLOCKED_cooccurrence_same_paper"
    else:
        if imode == "exact":
            if sf["mode"] in {"normalized_key", "prefix", "similar"} and shared_coauthors_small >= 1:
                auto = True; rule = "A_exact_initials_smallteam_coauthor"
            elif (
                sf["mode"] in {"normalized_key", "prefix", "similar"}
                and sf.get("raw_ratio", 0.0) >= 0.94
                and shared_institutions_small >= 1
                and min_gap is not None and min_gap <= 3
            ):
                auto = True; rule = "B_clear_name_variant_smallteam_institution"
        elif imode == "prefix":
            if sf["mode"] in {"exact", "normalized_key"} and shared_coauthors_small >= 2:
                auto = True; rule = "C_initial_prefix_2smallteam_coauthors"
            elif (
                sf["mode"] in {"exact", "normalized_key"}
                and shared_coauthors_small >= 1
                and shared_institutions_small >= 1
            ):
                auto = True; rule = "D_initial_prefix_smallteam_coauthor_institution"
            elif (
                sf["mode"] in {"prefix", "similar"}
                and shared_coauthors_small >= 2
                and shared_institutions_small >= 1
            ):
                auto = True; rule = "E_name_and_initial_variant_very_strong_smallteam_context"
        # imode == conflict is never auto-merged.

        plausible_review = False
        if not auto:
            if imode == "conflict":
                plausible_review = (shared_coauthors_small >= 1 or shared_institutions_small >= 1)
            elif score >= 72:
                plausible_review = True
            elif (shared_coauthors_small >= 1 and sf["mode"] in {"prefix", "similar", "normalized_key"}):
                plausible_review = True

    return {
        "score": score,
        "surname_mode": sf["mode"],
        "surname_similarity": sf["ratio"],
        "surname_raw_similarity": sf.get("raw_ratio", sf["ratio"]),
        "initials_mode": imode,
        "shared_coauthors": shared_coauthors,
        "shared_coauthors_small_team": shared_coauthors_small,
        "shared_institutions": shared_institutions,
        "shared_institutions_small_team": shared_institutions_small,
        "cooccurrence_papers": cooccurrence_papers,
        "year_overlap": int(year_overlap),
        "min_year_gap": min_gap,
        "auto_merge": auto,
        "auto_rule": rule,
        "plausible_review": plausible_review,
        "reasons": "; ".join(reasons),
    }


def generate_candidate_pairs(profiles: Mapping[str, Mapping]) -> pd.DataFrame:
    """Generate plausible variant pairs using surname-prefix buckets and first initial."""
    buckets: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for name, p in profiles.items():
        key = p["surname_key"]
        first = p["initials"][:1]
        if not key or not first:
            continue
        buckets[(key[:4], first)].append(name)

    # Exact-surname ambiguity statistics. Prefix-initial variants such as KIM, J /
    # KIM, JK are especially unsafe when a surname occurs with many initial families.
    surname_initials = defaultdict(set)
    for p in profiles.values():
        if p["surname"] and p["initials"]:
            surname_initials[p["surname"]].add(p["initials"])

    rows = []
    seen = set()
    for names in buckets.values():
        if len(names) < 2:
            continue
        for a, b in combinations(sorted(names), 2):
            pair = (a, b)
            if pair in seen:
                continue
            seen.add(pair)
            if a == b:
                continue
            res = score_candidate(profiles[a], profiles[b])
            if res is None:
                continue
            pa, pb = profiles[a], profiles[b]
            common_surname = (
                len(surname_initials.get(pa["surname"], set())) >= 4
                or len(surname_initials.get(pb["surname"], set())) >= 4
            )
            if res["auto_merge"] and res["initials_mode"] == "prefix" and common_surname:
                res["auto_merge"] = False
                res["plausible_review"] = True
                res["auto_rule"] = "BLOCKED_common_surname_initial_prefix"
                res["reasons"] += "; common_surname_prefix_block"
            if not res["auto_merge"] and not res["plausible_review"]:
                continue
            rows.append({
                "author_a": a,
                "author_b": b,
                "papers_a": pa["paper_count"],
                "papers_b": pb["paper_count"],
                "first_year_a": min(pa["years"]) if pa["years"] else None,
                "last_year_a": max(pa["years"]) if pa["years"] else None,
                "first_year_b": min(pb["years"]) if pb["years"] else None,
                "last_year_b": max(pb["years"]) if pb["years"] else None,
                "common_surname_prefix_risk": int(common_surname and res["initials_mode"] == "prefix"),
                **res,
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["auto_merge", "score", "shared_coauthors", "shared_institutions"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Manual overrides and safe clustering
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {x: x for x in items}
        self.members = {x: {x} for x in items}

    def find(self, x: str) -> str:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra].update(self.members.pop(rb))
        return True

    def cluster(self, x: str) -> set:
        return self.members[self.find(x)]


def load_manual_overrides(path: str | Path | None) -> pd.DataFrame:
    cols = ["author_a", "author_b", "decision", "canonical_name", "reason", "checked_by", "evidence_level"]
    if not path:
        return pd.DataFrame(columns=cols)
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(p, dtype=str).fillna("")
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df["decision"] = df["decision"].str.upper().str.strip()
    return df[cols]


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((safe_text(a).upper(), safe_text(b).upper())))


def _cluster_name_compatible(members_a: Iterable[str], members_b: Iterable[str]) -> bool:
    """Prevent transitive chains from joining plainly incompatible initial families."""
    for a in members_a:
        sa, ia = split_standard_author(a)
        for b in members_b:
            sb, ib = split_standard_author(b)
            sf = surname_features(sa, sb)
            if sf["mode"] is None:
                return False
            im = initials_mode(ia, ib)
            if im is None or im == "conflict":
                return False
    return True


def choose_canonical_name(members: Sequence[str], profiles: Mapping[str, Mapping], manual_preference: str = "") -> str:
    pref = safe_text(manual_preference).upper()
    if pref and pref in members:
        return pref
    def key(name):
        p = profiles[name]
        return (len(p["surname"]), len(p["initials"]), p["paper_count"], name)
    return sorted(members, key=key, reverse=True)[0]


def build_author_mapping(
    profiles: Mapping[str, Mapping],
    candidates: pd.DataFrame,
    manual_overrides: pd.DataFrame | None = None,
) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame]:
    """Apply conservative auto merges plus explicit manual decisions.

    Returns ``(name_map, variant_map_df, merge_log_df)``.
    """
    names = sorted(profiles)
    uf = UnionFind(names)
    manual_overrides = manual_overrides if manual_overrides is not None else pd.DataFrame()

    keep_separate = set()
    forced_merge = []
    canonical_pref = {}
    for _, r in manual_overrides.iterrows():
        a, b = safe_text(r.get("author_a")).upper(), safe_text(r.get("author_b")).upper()
        if not a or not b or a not in profiles or b not in profiles:
            continue
        dec = safe_text(r.get("decision")).upper()
        key = _pair_key(a, b)
        if dec in {"SEPARATE", "KEEP_SEPARATE", "NO"}:
            keep_separate.add(key)
        elif dec in {"MERGE", "YES"}:
            forced_merge.append((a, b, r))
            if safe_text(r.get("canonical_name")):
                canonical_pref[key] = safe_text(r.get("canonical_name")).upper()

    log_rows = []
    if not candidates.empty:
        auto = candidates[candidates["auto_merge"] == True].copy()  # noqa: E712
        auto = auto.sort_values(["score", "shared_coauthors", "shared_institutions"], ascending=False)
        for _, r in auto.iterrows():
            a, b = r["author_a"], r["author_b"]
            if _pair_key(a, b) in keep_separate:
                log_rows.append({"author_a": a, "author_b": b, "action": "blocked_manual_separate", "rule": r["auto_rule"], "score": r["score"]})
                continue
            ca, cb = uf.cluster(a), uf.cluster(b)
            if _cluster_name_compatible(ca, cb):
                changed = uf.union(a, b)
                log_rows.append({"author_a": a, "author_b": b, "action": "auto_merge" if changed else "already_same_cluster", "rule": r["auto_rule"], "score": r["score"]})
            else:
                log_rows.append({"author_a": a, "author_b": b, "action": "blocked_transitive_conflict", "rule": r["auto_rule"], "score": r["score"]})

    # Explicit manual merge decisions are applied after automatic clustering.
    for a, b, r in forced_merge:
        changed = uf.union(a, b)
        log_rows.append({
            "author_a": a, "author_b": b,
            "action": "manual_merge" if changed else "manual_already_same_cluster",
            "rule": safe_text(r.get("reason")), "score": None,
            "checked_by": safe_text(r.get("checked_by")),
            "evidence_level": safe_text(r.get("evidence_level")),
        })

    clusters = defaultdict(list)
    for n in names:
        clusters[uf.find(n)].append(n)

    name_map: Dict[str, str] = {}
    variant_rows = []
    for members in clusters.values():
        pref = ""
        # If any manual pair in this cluster specifies a canonical form, use it.
        for (a, b), cp in canonical_pref.items():
            if a in members and b in members and cp in members:
                pref = cp; break
        canon = choose_canonical_name(members, profiles, pref)
        for m in sorted(members):
            name_map[m] = canon
            p = profiles[m]
            variant_rows.append({
                "variant_name": m,
                "canonical_name": canon,
                "was_merged": int(m != canon),
                "cluster_size": len(members),
                "paper_count": p["paper_count"],
                "first_year": min(p["years"]) if p["years"] else None,
                "last_year": max(p["years"]) if p["years"] else None,
            })

    return (
        name_map,
        pd.DataFrame(variant_rows).sort_values(["canonical_name", "variant_name"]).reset_index(drop=True),
        pd.DataFrame(log_rows),
    )


def apply_author_mapping_to_records(records: Sequence[Mapping], name_map: Mapping[str, str]) -> List[dict]:
    out = []
    for rec in records:
        rr = dict(rec)
        source = [safe_text(x).upper() for x in (rec.get("authors_normalized") or []) if safe_text(x)]
        disamb = [name_map.get(a, a) for a in source]
        network = []
        seen = set()
        for a in disamb:
            if a and a not in seen:
                seen.add(a); network.append(a)
        rr["authors_disambiguated"] = disamb
        rr["authors_network"] = network
        rr["n_authors_disambiguated"] = len(disamb)
        rr["n_authors_network"] = len(network)
        out.append(rr)
    return out


# ---------------------------------------------------------------------------
# Review context utilities
# ---------------------------------------------------------------------------

def author_publications(records: Sequence[Mapping], author: str, authors_col: str = "authors_normalized") -> pd.DataFrame:
    author = safe_text(author).upper()
    rows = []
    for r in records:
        authors = [safe_text(x).upper() for x in (r.get(authors_col) or [])]
        if author in authors:
            rows.append({
                "record_id": r.get("record_id"), "year": r.get("year"),
                "title": r.get("title_one_line") or r.get("title"),
                "authors": "; ".join(authors),
                "institutions": "; ".join((r.get("institutions_normalized") or r.get("institutions") or [])),
                "countries": "; ".join(r.get("countries_normalized") or []),
            })
    return pd.DataFrame(rows).sort_values(["year", "record_id"]) if rows else pd.DataFrame()


def candidate_context(records: Sequence[Mapping], author_a: str, author_b: str) -> pd.DataFrame:
    a = author_publications(records, author_a)
    if not a.empty: a.insert(0, "candidate", "A")
    b = author_publications(records, author_b)
    if not b.empty: b.insert(0, "candidate", "B")
    return pd.concat([a, b], ignore_index=True) if not a.empty or not b.empty else pd.DataFrame()


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def run_disambiguation_from_phase2_dataframe(
    phase2_df: pd.DataFrame,
    manual_overrides_path: str | Path | None = None,
):
    records = _records_from_dataframe(phase2_df)
    profiles = build_author_profiles(records)
    candidates = generate_candidate_pairs(profiles)
    overrides = load_manual_overrides(manual_overrides_path)
    name_map, variant_map, merge_log = build_author_mapping(profiles, candidates, overrides)
    resolved_records = apply_author_mapping_to_records(records, name_map)
    return {
        "records": resolved_records,
        "profiles": profiles,
        "candidates": candidates,
        "name_map": name_map,
        "variant_map": variant_map,
        "merge_log": merge_log,
        "manual_overrides": overrides,
    }
