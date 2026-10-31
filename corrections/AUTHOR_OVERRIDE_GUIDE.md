# Manual author identity overrides

This file is intentionally small. It is **not** a request to hand-disambiguate the corpus.
Use it only for curated cases supported by record-level or historical evidence. The final
configuration includes the validated historical decisions plus the completed targeted audits of
transliteration, orthographic variants, dot-truncation cases, and historical snapshot discrepancies. It is not a request to hand-disambiguate every
near-matching surname in the corpus.

Columns:

- `author_a`, `author_b`: normalized names exactly as reported in `author_review_priority.csv`;
- `decision`: `MERGE` or `KEEP_SEPARATE`;
- `canonical_name`: optional preferred label when `MERGE` is chosen;
- `reason`: short evidence note (same known physicist, archival check, conflicting identities, etc.);
- `checked_by`: person or audit process making the decision;
- `evidence_level`: optional qualitative evidence grade used for audit documentation (`high`, `medium-high`, etc.).

After editing this CSV, rerun `03_disambiguate_authors.py` or the phase-3 notebook.
