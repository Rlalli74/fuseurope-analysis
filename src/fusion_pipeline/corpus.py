"""Final preserved-retrieval and analytical-corpus routing.

Author identity is constructed upstream on the preserved retrieval corpus. This
module only subsets those already-resolved records; it never re-estimates an
identity mapping.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import ast
import json

import pandas as pd


PRESERVED_EXPECTED_RECORDS = 46443
ANALYTICAL_EXPECTED_RECORDS = 45453
EDITORIAL_EXPECTED = 985
ADDITIONAL_EXPECTED = 5
ANALYTICAL_NORMALIZED_FORMS_EXPECTED = 41793
ANALYTICAL_CANONICAL_IDENTITIES_EXPECTED = 39906


class LicensedInputMissingError(FileNotFoundError):
    """Raised when a licensed full run lacks a required external input."""


def require_licensed_input(path: str | Path | None, description: str) -> Path:
    if path is None or not Path(path).expanduser().exists():
        shown = "<not supplied>" if path is None else str(Path(path).expanduser())
        raise LicensedInputMissingError(
            f"Licensed full run requires {description} at an external path; "
            f"received {shown}. No Web of Science record-level data are bundled "
            "with the reviewer package. Use reviewer_demo for the public-safe checks."
        )
    return Path(path).expanduser().resolve()


def parse_list_cell(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None:
        return []
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
            if isinstance(parsed, (list, tuple, set)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in text.split(";") if x.strip()]


def load_additional_exclusion_ids(path: str | Path) -> set[str]:
    """Load the five private manually validated event-summary identifiers."""
    path = require_licensed_input(path, "the private additional-event exclusion list")
    if path.suffix.lower() in {".csv", ".tsv"}:
        table = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
        for candidate in ("record_id", "ut", "UT", "id"):
            if candidate in table.columns:
                return {str(x).strip() for x in table[candidate].dropna() if str(x).strip()}
        raise ValueError(f"Exclusion table has no record identifier column: {path}")
    return {
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _id_column(df: pd.DataFrame) -> str:
    for name in ("record_id", "ut", "UT"):
        if name in df.columns:
            return name
    raise ValueError("Corpus must contain record_id, ut, or UT")


def _is_editorial(value: object) -> bool:
    return "Editorial Material" in set(parse_list_cell(value))


@dataclass(frozen=True)
class CorpusAudit:
    preserved_records: int
    analytical_records: int
    exclusions_total: int
    editorial_material_excluded: int
    additional_non_editorial_excluded: int
    preserved_unique_ids: int
    analytical_unique_ids: int


def derive_analytical_corpus(
    preserved: pd.DataFrame,
    additional_exclusion_ids: Iterable[str],
    *,
    strict_final_counts: bool = False,
) -> tuple[pd.DataFrame, CorpusAudit, pd.DataFrame]:
    """Subset an identity-resolved preserved corpus under the final rule."""
    if "document_types" not in preserved.columns:
        raise ValueError("Preserved corpus lacks document_types")
    id_col = _id_column(preserved)
    work = preserved.copy()
    ids = work[id_col].astype(str)
    if ids.duplicated().any():
        raise ValueError("Preserved corpus identifiers are not unique")
    additional = {str(x).strip() for x in additional_exclusion_ids if str(x).strip()}
    editorial = work["document_types"].apply(_is_editorial)
    additional_mask = ids.isin(additional) & ~editorial
    missing = sorted(additional - set(ids))
    if missing:
        raise ValueError(f"Additional exclusions absent from preserved corpus: {missing}")
    overlap = ids.isin(additional) & editorial
    if overlap.any():
        bad = sorted(ids[overlap].tolist())
        raise ValueError(f"Additional exclusions must be non-Editorial records: {bad}")
    excluded = editorial | additional_mask
    analytical = work.loc[~excluded].copy().reset_index(drop=True)
    private_audit = pd.DataFrame({
        "record_id": ids[excluded].tolist(),
        "exclusion_reason": [
            "wos_editorial_material" if flag else "validated_non_editorial_event_summary"
            for flag in editorial[excluded]
        ],
    })
    audit = CorpusAudit(
        preserved_records=len(work),
        analytical_records=len(analytical),
        exclusions_total=int(excluded.sum()),
        editorial_material_excluded=int(editorial.sum()),
        additional_non_editorial_excluded=int(additional_mask.sum()),
        preserved_unique_ids=int(ids.nunique()),
        analytical_unique_ids=int(analytical[id_col].astype(str).nunique()),
    )
    if strict_final_counts:
        expected = {
            "preserved_records": PRESERVED_EXPECTED_RECORDS,
            "analytical_records": ANALYTICAL_EXPECTED_RECORDS,
            "exclusions_total": PRESERVED_EXPECTED_RECORDS - ANALYTICAL_EXPECTED_RECORDS,
            "editorial_material_excluded": EDITORIAL_EXPECTED,
            "additional_non_editorial_excluded": ADDITIONAL_EXPECTED,
        }
        actual = asdict(audit)
        failures = {k: (actual[k], v) for k, v in expected.items() if actual[k] != v}
        if failures:
            raise ValueError(f"Final corpus count validation failed: {failures}")
    return analytical, audit, private_audit


def identity_mapping(df: pd.DataFrame) -> dict[str, str]:
    """Recover normalized-form to canonical-identity pairs stored on records."""
    required = {"authors_normalized", "authors_disambiguated", "authors_network"}
    if not required.issubset(df.columns):
        raise ValueError(f"Corpus lacks identity columns: {sorted(required - set(df.columns))}")
    mapping: dict[str, str] = {}
    for normalized, canonical, network in zip(
        df["authors_normalized"], df["authors_disambiguated"], df["authors_network"]
    ):
        left, right = parse_list_cell(normalized), parse_list_cell(canonical)
        if len(left) != len(right):
            raise ValueError("Author identity columns have incompatible within-record lengths")
        expected_network = list(dict.fromkeys(right))
        if parse_list_cell(network) != expected_network:
            raise ValueError("authors_network is not the order-preserving unique canonical-author projection")
        for form, identity in zip(left, right):
            previous = mapping.setdefault(form, identity)
            if previous != identity:
                raise ValueError(f"Normalized form maps to multiple canonical identities: {form}")
    return mapping


def validate_identity_subset(
    preserved: pd.DataFrame,
    analytical: pd.DataFrame,
    *,
    strict_final_counts: bool = False,
) -> dict[str, int | str]:
    """Prove that subsetting did not rebuild or alter represented identities."""
    full = identity_mapping(preserved)
    subset = identity_mapping(analytical)
    changed = {form: (full.get(form), identity) for form, identity in subset.items() if full.get(form) != identity}
    if changed:
        raise ValueError(f"Analytical identity mapping differs from preserved mapping: {changed}")
    result = {
        "status": "pass",
        "preserved_normalized_forms": len(full),
        "analytical_normalized_forms": len(subset),
        "analytical_canonical_identities": len(set(subset.values())),
        "changed_mappings": 0,
    }
    if strict_final_counts:
        expected = {
            "preserved_normalized_forms": 41966,
            "analytical_normalized_forms": ANALYTICAL_NORMALIZED_FORMS_EXPECTED,
            "analytical_canonical_identities": ANALYTICAL_CANONICAL_IDENTITIES_EXPECTED,
        }
        failures = {key: (result[key], value) for key, value in expected.items() if result[key] != value}
        if failures:
            raise ValueError(f"Final analytical identity count validation failed: {failures}")
    return result
