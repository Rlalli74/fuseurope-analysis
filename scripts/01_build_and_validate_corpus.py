#!/usr/bin/env python3
"""Build the canonical WoS corpus and validate author parsing.

Example
-------
python scripts/01_build_and_validate_corpus.py \
  --raw-dir "/path/to/WoS journals plus word search merged" \
  --historical-csv "/path/to/WoS Fusion Publications ... corrected_NEW.csv" \
  --output-dir outputs/ingestion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fusion_pipeline.wos_parser import (
    document_type_selection_check,
    parse_wos_folder,
    records_to_dataframe,
)
from fusion_pipeline.validation import summarize_document_types, validate_against_historical_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="Folder containing WoS .txt exports")
    parser.add_argument("--historical-csv", default=None, help="Optional historical 46,443-row analysis CSV used only for parser validation")
    parser.add_argument("--output-dir", default="outputs/ingestion")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Parsing WoS files...")
    records = parse_wos_folder(args.raw_dir)
    df = records_to_dataframe(records, serialise_lists=False)
    print(f"Parsed {len(df):,} records; unique UT: {df['record_id'].nunique():,}")

    # CSV is convenient for inspection; list-valued fields are encoded as JSON strings.
    df_csv = records_to_dataframe(records, serialise_lists=True)
    corpus_path = out / "fusion_corpus_canonical_parsed.csv"
    df_csv.to_csv(corpus_path, index=False)
    print(f"Saved canonical parsed corpus: {corpus_path}")

    dt = summarize_document_types(df)
    dt.to_csv(out / "document_type_summary.csv", index=False)

    dt_check = document_type_selection_check(records)
    with open(out / "document_type_selection_check.json", "w", encoding="utf-8") as f:
        json.dump(dt_check, f, ensure_ascii=False, indent=2)

    summary = None
    if args.historical_csv:
        summary, _ = validate_against_historical_csv(df, args.historical_csv, output_dir=out)
        print("\nAuthor parsing validation against historical CSV:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print("\nHistorical CSV not supplied: parser comparison audit skipped.")

    if len(df) != 46443:
        raise SystemExit(f"VALIDATION FAILURE: expected 46,443 records, got {len(df):,}")
    if df["record_id"].nunique(dropna=True) != 46443:
        raise SystemExit("VALIDATION FAILURE: UT identifiers are missing or duplicated")
    if summary is not None and summary["comparison_status_counts"].get("author_count_differs", 0) > 0:
        print("WARNING: some matched records differ in author count; inspect author_parsing_mismatches.csv")


if __name__ == "__main__":
    main()
