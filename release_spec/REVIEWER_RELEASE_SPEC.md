# Reviewer release specification

## Purpose

Provide referees with the final shared analytical code for both articles, clear methodological configuration, synthetic tests, and permitted derived outputs, while withholding licensed WoS record-level data.

## Required contents

- `README.md`
- `docs/REVIEWER_GUIDE.md`
- `docs/DATA_REQUIREMENTS.md`
- `docs/METHODS_CODE_CROSSWALK.md`
- `docs/REPRODUCIBILITY_SCOPE.md`
- Python package under `src/fusion_pipeline/`
- article-relevant scripts
- `notebooks/00_pipeline_control_and_network_explorer.ipynb`
- `notebooks/00_reproducibility_check.ipynb`
- article-relevant notebooks that contain no private cell outputs
- public-safe configuration files
- public-safe correction tables required to reproduce final logic
- dependency/environment specifications
- synthetic tests and fixtures
- aggregate/permitted reference outputs used by regression checks
- `REVIEWER_PACKAGE_MANIFEST.csv`
- `REVIEWER_RELEASE_REPORT.md`

## Required omissions

- any WoS source record;
- record-level derived corpus;
- private record-level audits;
- exploratory Guzzardi–Lalli branch;
- old manuscripts/reviewer correspondence;
- local absolute paths;
- caches and generated binary Python files;
- credentials/tokens;
- final public DOI/tag claims.

## Reviewer experience

A reviewer should be able to:

1. read the README and understand the two corpus layers;
2. see exactly which method supports each article figure/table;
3. inspect all analytical functions;
4. run `pytest` successfully without WoS data;
5. run the reviewer smoke-test notebook without WoS data;
6. see the final expected numerical targets in aggregate form;
7. understand exactly which full-data outputs cannot be regenerated without lawful WoS access;
8. understand how a lawful WoS user would supply external inputs.

## Suggested command sequence in README

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest -q
jupyter notebook notebooks/00_reproducibility_check.ipynb
```

For a full licensed run, provide an explicit example using external paths, but do not bundle those inputs.

## Release name

Use a neutral review-only name, e.g.:

`FusEurope_reproducibility_for_peer_review_2026-09-XX.zip`

Do not call it a final public release and do not assign a DOI at this stage.
