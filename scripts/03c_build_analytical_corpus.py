#!/usr/bin/env python3
"""Derive the final analytical corpus after shared author identity is frozen."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusion_pipeline.corpus import (
    derive_analytical_corpus,
    load_additional_exclusion_ids,
    require_licensed_input,
    validate_identity_subset,
)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("preserved_identity_csv", help="Identity-resolved 46,443-record preserved corpus")
    parser.add_argument("--additional-exclusions", required=True, help="Private list of five validated non-Editorial identifiers")
    parser.add_argument("--output-dir", default="pipeline_outputs/03c_analytical_corpus")
    parser.add_argument("--allow-nonfinal-counts", action="store_true", help="Permit synthetic/development inputs")
    args = parser.parse_args()

    source = require_licensed_input(args.preserved_identity_csv, "the identity-resolved preserved corpus")
    additional = load_additional_exclusion_ids(args.additional_exclusions)
    preserved = pd.read_csv(source, low_memory=False)
    analytical, audit, private_audit = derive_analytical_corpus(
        preserved, additional, strict_final_counts=not args.allow_nonfinal_counts,
    )
    identity = validate_identity_subset(
        preserved, analytical, strict_final_counts=not args.allow_nonfinal_counts,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    analytical.to_csv(out / "analytical_corpus_phase3c.csv", index=False)
    # This is record-level provenance and remains private by construction.
    private_audit.to_csv(out / "analytical_exclusion_audit_private.csv", index=False)
    summary = {"corpus": asdict(audit), "identity_subset": identity}
    (out / "analytical_corpus_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
