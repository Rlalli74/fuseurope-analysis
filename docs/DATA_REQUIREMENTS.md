# Data requirements

## Not required for public-safe verification

Package import, all unit tests, method-preset validation, synthetic parsing/corpus/network/country checks, and release-safety checks require no WoS records.

## Required for a licensed full run

- Lawfully accessed WoS plain-text exports corresponding to the preserved 46,443-record retrieval corpus.
- A private identifier-only list of the five validated non-editorial event summaries.
- Lawful annual Physics and Engineering series for the relative-publication comparison. Their reuse rights are not established, so they are not redistributed.

Inputs are supplied by explicit external paths. The pipeline creates the canonical identity layer on the preserved corpus, validates it, then derives the analytical corpus without rebuilding identities.

Record-level corpora, exclusion audits, affiliation tables, candidate-pair evidence, graph provenance, and outputs containing titles, UTs, or addresses remain private. Only aggregate non-invertible validation results may enter the reviewer release after scanning.
