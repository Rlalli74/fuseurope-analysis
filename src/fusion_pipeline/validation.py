"""Validation routines for the canonical FusEurope WoS parser."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import ast
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def summarize_document_types(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df["document_type_raw"]
        .fillna("<MISSING>")
        .value_counts(dropna=False)
        .rename_axis("document_type_raw")
        .reset_index(name="n_records")
    )


def _parse_old_author_cell(value) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return []


def _author_pair_cost(new: Sequence[str], old: Sequence[str]) -> float:
    """Cost for matching duplicate year-title records during validation."""
    if list(new) == list(old):
        return 0.0
    count_penalty = abs(len(new) - len(old)) * 1000.0
    overlap = len(set(new) & set(old))
    union = len(set(new) | set(old)) or 1
    set_penalty = (1.0 - overlap / union) * 100.0
    positional = sum(a != b for a, b in zip(new, old))
    return count_penalty + set_penalty + positional


def _match_group(new_rows: pd.DataFrame, old_rows: pd.DataFrame) -> List[Tuple[int | None, int | None]]:
    n, m = len(new_rows), len(old_rows)
    if n == 0:
        return [(None, j) for j in range(m)]
    if m == 0:
        return [(i, None) for i in range(n)]
    # More than 99.8% of year-title groups are one-to-one. Avoid invoking
    # the assignment algorithm for those groups.
    if n == 1 and m == 1:
        return [(0, 0)]

    size = max(n, m)
    cost = np.full((size, size), 1e9, dtype=float)
    for i in range(n):
        for j in range(m):
            cost[i, j] = _author_pair_cost(new_rows.iloc[i]["authors"], old_rows.iloc[j]["authors_old"])

    row_ind, col_ind = linear_sum_assignment(cost)
    pairs: List[Tuple[int | None, int | None]] = []
    matched_new, matched_old = set(), set()
    for i, j in zip(row_ind, col_ind):
        if i < n and j < m:
            pairs.append((i, j))
            matched_new.add(i)
            matched_old.add(j)
    for i in range(n):
        if i not in matched_new:
            pairs.append((i, None))
    for j in range(m):
        if j not in matched_old:
            pairs.append((None, j))
    return pairs


def validate_against_historical_csv(
    parsed_df: pd.DataFrame,
    historical_csv: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Compare parsed AU separation against the historical analysis CSV.

    The historical CSV has no WoS UT field, so records are grouped by (year, title).
    Duplicate title-year groups are paired by minimum author-list discrepancy using
    the Hungarian assignment algorithm.
    """
    old = pd.read_csv(historical_csv)
    old = old.copy()
    old["authors_old"] = old["authors"].apply(_parse_old_author_cell)

    new = parsed_df[["record_id", "source_file", "year", "title", "authors", "n_authors"]].copy()
    new["authors"] = new["authors"].apply(lambda x: list(x) if isinstance(x, list) else [])

    new_groups = {k: g.reset_index(drop=True) for k, g in new.groupby(["year", "title"], dropna=False)}
    old_groups = {k: g.reset_index(drop=True) for k, g in old.groupby(["year", "title"], dropna=False)}
    keys = set(new_groups) | set(old_groups)

    rows = []
    for key in keys:
        ng = new_groups.get(key, new.iloc[0:0].copy())
        og = old_groups.get(key, old.iloc[0:0].copy())
        for ni, oi in _match_group(ng, og):
            nr = ng.iloc[ni] if ni is not None else None
            orow = og.iloc[oi] if oi is not None else None
            new_auth = list(nr["authors"]) if nr is not None else []
            old_auth = list(orow["authors_old"]) if orow is not None else []

            if nr is None:
                status = "historical_only"
            elif orow is None:
                status = "parsed_only"
            elif new_auth == old_auth:
                status = "exact_author_list_match"
            elif len(new_auth) == len(old_auth):
                status = "same_author_count_names_differ"
            else:
                status = "author_count_differs"

            diffs = []
            if nr is not None and orow is not None:
                for pos in range(max(len(new_auth), len(old_auth))):
                    a = new_auth[pos] if pos < len(new_auth) else "<MISSING>"
                    b = old_auth[pos] if pos < len(old_auth) else "<MISSING>"
                    if a != b:
                        diffs.append(f"{pos+1}: {a} -> {b}")

            rows.append({
                "year": key[0],
                "title": key[1],
                "record_id": nr["record_id"] if nr is not None else None,
                "source_file": nr["source_file"] if nr is not None else None,
                "status": status,
                "n_authors_parsed": len(new_auth) if nr is not None else np.nan,
                "n_authors_historical": len(old_auth) if orow is not None else np.nan,
                "authors_parsed": json.dumps(new_auth, ensure_ascii=False),
                "authors_historical": json.dumps(old_auth, ensure_ascii=False),
                "position_differences": " | ".join(diffs),
            })

    detail = pd.DataFrame(rows)
    counts = detail["status"].value_counts().to_dict()

    summary = {
        "parsed_records": int(len(parsed_df)),
        "historical_records": int(len(old)),
        "parsed_unique_ut": int(parsed_df["record_id"].nunique(dropna=True)),
        "parsed_missing_ut": int(parsed_df["record_id"].isna().sum()),
        "year_min": int(parsed_df["year"].min()),
        "year_max": int(parsed_df["year"].max()),
        "comparison_status_counts": {k: int(v) for k, v in counts.items()},
        "exact_author_list_match_rate": float((detail["status"] == "exact_author_list_match").mean()),
        "same_author_count_rate_including_name_differences": float(
            detail["status"].isin(["exact_author_list_match", "same_author_count_names_differ"]).mean()
        ),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        detail.to_csv(output_dir / "author_parsing_comparison_detail.csv", index=False)
        detail[detail["status"] != "exact_author_list_match"].to_csv(
            output_dir / "author_parsing_mismatches.csv", index=False
        )
        with open(output_dir / "author_parsing_validation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary, detail
