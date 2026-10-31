# Full licensed-data reproducibility and numerical acceptance

A private complete rerun was executed on 30 October 2026 using the original 94 Web of Science exports, the restricted five-record exclusion list, and the licensed Physics/Engineering comparison series. The complete analysis and figure-generating stages executed; the numerical acceptance matched **563/563** checks after the private provenance reconciliation. The revised public reference targets the definitive 46-column output and was validated against the saved private outputs on the researcher's Mac (563/563 PASS). The record-level source inputs and outputs are NOT distributed.

## Reproduce privately

Install the versions pinned in `environment-reproducibility.yml`. Set `MPLBACKEND=Agg` on Mac or any headless system to prevent interactive GUI crashes. Execute from the repository root, with all input and output paths outside the checkout:

```bash
MPLBACKEND=Agg python scripts/run_pipeline.py --mode licensed_full_run \
  --raw-dir /PRIVATE/wos_exports \
  --additional-exclusions /PRIVATE/five_event_summary_ids.txt \
  --physics-baseline /PRIVATE/All_physics_publications.csv \
  --engineering-baseline /PRIVATE/all_engineering_publications.csv \
  --work-dir /PRIVATE/RUN
```

The orchestrator does not include every separately required final verification step. Complete any missing stages (notably phase 15, seed-robustness phase 06b and figures as applicable) before assembling acceptance results. For the source-only release, the reproducible final assembly commands are:

```bash
python scripts/15_titleword_preprocessing_sensitivity.py \
  /PRIVATE/RUN/03c_analytical_corpus/analytical_corpus_phase3c.csv \
  --output-dir /PRIVATE/RUN/15_titleword_preprocessing_sensitivity

python scripts/assemble_final_acceptance_results.py \
  --run-root /PRIVATE/RUN \
  --output /PRIVATE/final_aggregate_results.yaml

python scripts/validate_final_papers.py \
  --results /PRIVATE/final_aggregate_results.yaml \
  --reference acceptance/FINAL_REFERENCE_VALUES.yaml \
  --output-dir /PRIVATE/final_acceptance
```

**Single final representation:** The assembler reads `/PRIVATE/RUN/03_disambiguation/preserved_identity_corpus_phase3.csv` (46 columns), the same retained retrieval-level input used for publication-trend analysis. It does not need a predecessor dataset or an alternate corpus file. Its byte-level SHA-256 reference is the checksum observed in the validated final private run. File-byte equality can depend on export serialization and the software environment; for other environments, investigate mismatches instead of silently replacing the reference.

Expected membership: 46,443 distinct preserved WoS UTs; 985 Editorial Material plus five separately verified event summaries excluded; 45,453 analytical UTs; 40,073 retained-corpus canonical author identities and 39,906 analytical-corpus identities. Country reconstruction uses deduplicated C1 union RP; author-identity institutional evidence stays separate. The 563 checks include exact integers, rounded scientific metrics, and two provenance attributes (schema width and file digest). They are not 563 independent substantive scientific tests. The complete private run also separately verified the UT set and C1/RP transformation.

**Do not publish** original exports, normalized or analytical record-level tables, identified correction lists, private reports or licensed comparison series.
