# Release status and scope — 30 October 2026

This is a **validated source-code staging candidate**, not an already published or DOI-archived release. The final licensed-data pipeline ran privately on an Apple Silicon Mac using the original dated exports, the restricted five-event exclusion list and the licensed Physics/Engineering aggregate series. All manuscript analytical phases and figure-generation scripts completed, and 563/563 final aggregate acceptance targets matched. A private record-level audit checked the 46,443 UT membership and the C1/RP transformations against the original address evidence; it was not an independent manual historical review of every affiliation.

The sole retained retrieval-level derived representation used by this release has **46 columns**. The analytical subset comprises 45,453 records; identity mappings are constructed once on the entire retrieval set and applied unchanged to the subset.

## Public source-only checks

- 69 synthetic and invariant tests passed on the final revised candidate.
- `notebooks/00_reproducibility_check.ipynb` ran its eight code cells; `scripts/run_pipeline.py --mode reviewer_demo` completed.
- `scripts/audit_public_repository.py` screens common sensitive-content patterns, but a human and legal review are still mandatory.
- The revised 46-column public candidate was checked against saved private outputs on the researcher's Mac and matched 563/563 reference checks. These are aggregate checks and do not transfer licensed records.

## Excluded materials

Source exports, per-publication corpora, original WoS identifiers, the five excluded identifiers, restricted identity evidence, source-level address audits, historical private runs, and the licensed Physics/Engineering comparison files are excluded. Only source code, permitted settings/corrections, synthetic tests, and aggregate reference values can be made public after rights review.

## Publication remains pending

Software author: **Roberto Lalli** (sole named author; see `CITATION.cff`). Confirm the necessary redistribution rights and choose a software licence, audit all contents manually, then publish an immutable GitHub tag and optional Zenodo archive. Do not invent URLs, tags, or DOIs. Exact historical replication requires lawful access to the same dated exports and the private adjudications; rerunning the original search on current WoS holdings is not equivalent.
