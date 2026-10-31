#!/usr/bin/env python3
"""Validate curated author-identity overrides against a regenerated phase-3 corpus.

This audit is intentionally deterministic and contains no hidden name-resolution
logic. It checks that explicit MERGE/KEEP_SEPARATE decisions are reflected in
``authors_disambiguated``, that no two distinct listed authors on one publication
have been collapsed into the same canonical identity, and (optionally) that the
22 multiple-candidate dot-truncation families are closed after downstream
identity resolution.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from fusion_pipeline.disambiguation import load_manual_overrides, parse_list_cell, safe_text


def build_name_map(df: pd.DataFrame):
    mapping = {}
    inconsistent = []
    collision_rows = []
    for idx, row in df.iterrows():
        norm = [safe_text(x).upper() for x in parse_list_cell(row.get("authors_normalized", [])) if safe_text(x)]
        dis = [safe_text(x).upper() for x in parse_list_cell(row.get("authors_disambiguated", [])) if safe_text(x)]
        if len(norm) != len(dis):
            inconsistent.append({
                "row_index": int(idx),
                "record_id": row.get("record_id", row.get("accession_id", "")),
                "issue": "normalized/disambiguated length mismatch",
                "n_normalized": len(norm),
                "n_disambiguated": len(dis),
            })
            continue
        for a, c in zip(norm, dis):
            old = mapping.get(a)
            if old is not None and old != c:
                inconsistent.append({
                    "row_index": int(idx),
                    "record_id": row.get("record_id", row.get("accession_id", "")),
                    "issue": "normalized form maps to multiple canonical names",
                    "author": a,
                    "canonical_1": old,
                    "canonical_2": c,
                })
            mapping[a] = c

        by_canon = defaultdict(list)
        for a, c in zip(norm, dis):
            by_canon[c].append(a)
        for canon, source_names in by_canon.items():
            uniq = sorted(set(source_names))
            if len(uniq) > 1:
                collision_rows.append({
                    "row_index": int(idx),
                    "record_id": row.get("record_id", row.get("accession_id", "")),
                    "year": row.get("year", row.get("publication_year", "")),
                    "canonical_name": canon,
                    "source_author_forms": " | ".join(uniq),
                    "title": row.get("title_one_line", row.get("title", "")),
                })
    return mapping, pd.DataFrame(inconsistent), pd.DataFrame(collision_rows)


def check_overrides(overrides: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    rows = []
    for _, r in overrides.iterrows():
        a = safe_text(r.get("author_a")).upper()
        b = safe_text(r.get("author_b")).upper()
        dec = safe_text(r.get("decision")).upper()
        ca, cb = mapping.get(a), mapping.get(b)
        if ca is None or cb is None:
            status = "MISSING_AUTHOR"
            ok = False
        elif dec in {"MERGE", "YES"}:
            ok = ca == cb
            status = "PASS" if ok else "FAIL_MERGE_NOT_UNIFIED"
        elif dec in {"KEEP_SEPARATE", "SEPARATE", "NO"}:
            ok = ca != cb
            status = "PASS" if ok else "FAIL_KEEP_SEPARATE_COLLAPSED"
        else:
            ok = False
            status = "UNKNOWN_DECISION"
        rows.append({
            "author_a": a, "author_b": b, "decision": dec,
            "canonical_a": ca or "", "canonical_b": cb or "",
            "status": status, "ok": int(ok),
            "reason": r.get("reason", ""),
            "checked_by": r.get("checked_by", ""),
            "evidence_level": r.get("evidence_level", ""),
        })
    return pd.DataFrame(rows)


def check_dot_closure(dot_path: Path, mapping: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(dot_path, dtype=str).fillna("")
    df = df[df["status"].eq("review_multiple_candidates")].copy()
    rows = []
    for _, r in df.iterrows():
        source = safe_text(r.get("resolved_or_standardized")).upper()
        candidates = [safe_text(x).upper() for x in safe_text(r.get("candidate_labels")).split("|") if safe_text(x)]
        labels = [x for x in [source] + candidates if x]
        mapped = {lab: mapping.get(lab, "") for lab in labels}
        present = {lab: can for lab, can in mapped.items() if can}
        canonicals = sorted(set(present.values()))
        closed = bool(present) and len(canonicals) == 1
        rows.append({
            "raw_truncated_form": r.get("raw_truncated_form", ""),
            "occurrences": r.get("occurrences", ""),
            "reviewed_labels": " | ".join(labels),
            "mapped_canonical_names": " | ".join(canonicals),
            "n_labels_present": len(present),
            "closed_to_single_canonical_cluster": int(closed),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase3_csv", help="Regenerated preserved_identity_corpus_phase3.csv")
    ap.add_argument("--overrides", default=str(ROOT / "corrections" / "author_identity_overrides.csv"))
    ap.add_argument("--dot-truncation-audit", default=None, help="Optional phase-2 author_dot_truncation_audit.csv")
    ap.add_argument("--output-dir", default="outputs_phase3_identity_review")
    ap.add_argument("--allow-existing-same-paper-collisions", action="store_true",
                    help="Report but do not fail on same-paper identity collisions")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.phase3_csv, low_memory=False)
    mapping, inconsistent, collisions = build_name_map(df)
    overrides = load_manual_overrides(args.overrides)
    override_check = check_overrides(overrides, mapping)

    override_check.to_csv(out / "author_override_validation.csv", index=False)
    inconsistent.to_csv(out / "author_mapping_inconsistencies.csv", index=False)
    collisions.to_csv(out / "same_paper_identity_collisions.csv", index=False)

    dot = pd.DataFrame()
    if args.dot_truncation_audit:
        dot = check_dot_closure(Path(args.dot_truncation_audit), mapping)
        dot.to_csv(out / "dot_truncation_review_closure.csv", index=False)

    canonical_names = set(mapping.values())
    summary = {
        "records": int(len(df)),
        "normalized_author_forms": int(len(mapping)),
        "canonical_author_nodes": int(len(canonical_names)),
        "variant_forms_merged": int(len(mapping) - len(canonical_names)),
        "override_rows": int(len(override_check)),
        "override_merge_rows": int((override_check["decision"] == "MERGE").sum()) if not override_check.empty else 0,
        "override_keep_separate_rows": int(override_check["decision"].isin(["KEEP_SEPARATE", "SEPARATE"]).sum()) if not override_check.empty else 0,
        "override_failures": int((override_check["ok"] != 1).sum()) if not override_check.empty else 0,
        "mapping_inconsistencies": int(len(inconsistent)),
        "same_paper_identity_collisions": int(len(collisions)),
        "dot_multiple_candidate_rows": int(len(dot)),
        "dot_multiple_candidate_rows_closed": int(dot["closed_to_single_canonical_cluster"].sum()) if not dot.empty else None,
    }
    with open(out / "author_identity_review_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    fail = summary["override_failures"] or summary["mapping_inconsistencies"]
    if not args.allow_existing_same_paper_collisions:
        fail = fail or summary["same_paper_identity_collisions"]
    if not dot.empty:
        fail = fail or (summary["dot_multiple_candidate_rows_closed"] != summary["dot_multiple_candidate_rows"])
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
