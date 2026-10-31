"""Robust parser for Clarivate Web of Science plain-text exports.

Design goals
------------
1. Preserve record boundaries exactly (PT ... ER).
2. Preserve Web of Science AU author separation: one physical AU/continuation line
   corresponds to one author, with a semicolon fallback for unusual exports.
3. Preserve raw values while also providing analysis-compatible fields.
4. Never overwrite raw bibliographic information during later normalization.

The parser deliberately separates ingestion from later author disambiguation,
institution normalization, country normalization, and the special JET network
representation used in the historical paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence
import json
import re

import pandas as pd

FIELD_RE = re.compile(r"^([A-Z0-9]{2})\s(.*)$")

# The document-type selection reported by the authors for the original WoS query.
ALLOWED_BASE_DOCUMENT_TYPES = {
    "Article",
    "Proceedings Paper",
    "Meeting Abstract",
    "Review",
    "Note",
    "Editorial Material",
    "Letter",
    "Book Chapter",
}


def _normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _iter_record_lines(text: str) -> Iterator[List[str]]:
    """Yield record lines, using a literal ER line as the record terminator."""
    text = _normalise_newlines(text)
    current: List[str] = []
    in_record = False

    for raw_line in text.split("\n"):
        line = raw_line.rstrip("\n")

        if line.startswith("PT "):
            # A PT line begins a bibliographic record. If a malformed export omitted
            # ER, do not silently merge two records.
            if current and in_record:
                yield current
                current = []
            in_record = True
            current.append(line)
            continue

        if line == "ER":
            if current and in_record:
                yield current
            current = []
            in_record = False
            continue

        if in_record:
            current.append(line)

    if current and in_record:
        # Preserve a final unterminated record for audit rather than dropping it.
        yield current


def _parse_fields(lines: Sequence[str]) -> Dict[str, List[str]]:
    """Parse WoS field codes while preserving physical continuation lines."""
    fields: Dict[str, List[str]] = {}
    current_code: str | None = None

    for line in lines:
        m = FIELD_RE.match(line)
        if m:
            code, value = m.group(1), m.group(2)
            fields.setdefault(code, []).append(value)
            current_code = code
        elif line.startswith("   ") and current_code:
            fields[current_code].append(line[3:])
        elif line and current_code:
            # Defensive fallback for an unusual continuation lacking the canonical
            # three-space prefix. Keeping it is safer than silently discarding text.
            fields[current_code].append(line)

    return fields


def _multiline_legacy(fields: Mapping[str, List[str]], code: str) -> str | None:
    """Recreate the multiline representation used by the historical parser."""
    values = fields.get(code, [])
    if not values:
        return None
    return values[0] + "".join("\n   " + v for v in values[1:])


def _one_line(fields: Mapping[str, List[str]], code: str) -> str | None:
    values = fields.get(code, [])
    if not values:
        return None
    return " ".join(v.strip() for v in values if v.strip()).strip() or None


def _author_lines(fields: Mapping[str, List[str]], code: str = "AU") -> List[str]:
    """Extract authors without splitting commas inside author names.

    WoS plain text exports place the first AU value on the AU line and subsequent
    authors on continuation lines. A semicolon fallback is retained because some
    export variants put several names on one physical line.
    """
    out: List[str] = []
    for value in fields.get(code, []):
        parts = value.split(";") if ";" in value else [value]
        out.extend(p.strip() for p in parts if p.strip())
    return out


def _document_types(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(";") if x.strip()]


def _affiliation_lines(fields: Mapping[str, List[str]]) -> List[str]:
    """Return one C1 physical line per affiliation/address.

    In the supplied exports each C1 affiliation line ends in punctuation and each
    additional address is a continuation line. This representation is preserved
    rather than joining the C1 block into an undifferentiated string.
    """
    return [x.strip() for x in fields.get("C1", []) if x.strip()]


def _institution_from_affiliation(affiliation: str) -> str:
    return affiliation.split(",", 1)[0].strip()


def _country_tail_from_affiliation(affiliation: str) -> str:
    """Preserve the final comma-delimited address segment for later normalization."""
    if "," not in affiliation:
        return affiliation.rstrip(". ").strip()
    return affiliation.rsplit(",", 1)[-1].rstrip(". ").strip()


def parse_wos_file(path: str | Path) -> List[dict]:
    """Parse one WoS plain-text export into record dictionaries."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    records: List[dict] = []

    for local_index, lines in enumerate(_iter_record_lines(text), start=1):
        fields = _parse_fields(lines)
        year_raw = _one_line(fields, "PY")
        year = int(year_raw) if year_raw and year_raw.isdigit() else None

        authors_wos = _author_lines(fields, "AU")
        authors_full_wos = _author_lines(fields, "AF")
        authors_analysis = [a.upper() for a in authors_wos]

        dt_raw = _one_line(fields, "DT")
        affiliations = _affiliation_lines(fields)
        institutions = [_institution_from_affiliation(a) for a in affiliations]
        country_address_tails = [_country_tail_from_affiliation(a) for a in affiliations]

        record = {
            "source_file": path.name,
            "source_record_index": local_index,
            "record_id": _one_line(fields, "UT"),
            "ut": _one_line(fields, "UT"),
            "publication_type": _one_line(fields, "PT"),
            "year": year,
            # Compatibility field: uppercase AU values, as used in the analysis CSV.
            "authors": authors_analysis,
            # Raw/traceable fields.
            "authors_wos": authors_wos,
            "authors_full_wos": authors_full_wos,
            "n_authors": len(authors_wos),
            # Keep legacy multiline title plus a convenient one-line title.
            "title": _multiline_legacy(fields, "TI"),
            "title_one_line": _one_line(fields, "TI"),
            "source_title": _one_line(fields, "SO"),
            "language": _one_line(fields, "LA"),
            "document_type_raw": dt_raw,
            "document_types": _document_types(dt_raw),
            "abstract": _multiline_legacy(fields, "AB"),
            "keywords": _multiline_legacy(fields, "DE"),
            "keywords_plus": _multiline_legacy(fields, "ID"),
            "collaboration": _multiline_legacy(fields, "CA"),
            "affiliations": affiliations,
            "c1_raw": _multiline_legacy(fields, "C1"),
            # RP is the corresponding/reprint-author address.  It is preserved
            # separately because it cannot substitute for the complete C1 block,
            # but it can be used to audit whether C1 missingness differs by the
            # corresponding author's country.
            "reprint_address": _multiline_legacy(fields, "RP"),
            "reprint_address_one_line": _one_line(fields, "RP"),
            "institutions": institutions,
            "country_address_tails": country_address_tails,
            "doi": _one_line(fields, "DI"),
            "wos_categories": _one_line(fields, "WC"),
            "research_areas": _one_line(fields, "SC"),
            "retrieval_date": _one_line(fields, "DA"),
        }
        records.append(record)

    return records


def parse_wos_folder(folder: str | Path, pattern: str = "*.txt") -> List[dict]:
    """Parse all WoS text exports in a folder, sorted by filename."""
    folder = Path(folder)
    paths = sorted(folder.glob(pattern))
    records: List[dict] = []
    for path in paths:
        records.extend(parse_wos_file(path))
    return records


def records_to_dataframe(records: Sequence[dict], serialise_lists: bool = False) -> pd.DataFrame:
    """Convert parsed records to a dataframe.

    With ``serialise_lists=True`` list-valued columns become JSON strings, which
    round-trip safely through CSV without Python ``eval``.
    """
    df = pd.DataFrame(records)
    if serialise_lists:
        for col in [
            "authors", "authors_wos", "authors_full_wos", "document_types",
            "affiliations", "institutions", "country_address_tails",
        ]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False))
    return df


def document_type_selection_check(records: Sequence[dict]) -> dict:
    """Check whether record DT values conform to the reported query selection."""
    unexpected = []
    for r in records:
        types = set(r.get("document_types", []))
        extras = types - ALLOWED_BASE_DOCUMENT_TYPES - {"Retracted Publication"}
        if extras:
            unexpected.append({
                "record_id": r.get("record_id"),
                "document_type_raw": r.get("document_type_raw"),
                "unexpected_types": sorted(extras),
            })
    return {
        "n_records": len(records),
        "n_unexpected_document_type_records": len(unexpected),
        "unexpected": unexpected,
    }
