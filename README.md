# FusEurope Reproducibility Package

**Software author and repository maintainer:** Roberto Lalli. The authorship of the two research articles is distinct from the authorship of this software. See `CITATION.cff` for software citation metadata.

This package contains the shared analytical code supporting two related articles based on the same historical Web of Science (WoS) corpus: the Lalli article, *From Institutional Frameworks to Connected Collaboration: European Networks in Controlled Thermonuclear Research, 1957–2007*, and the Guzzardi–Lalli article, *Mapping Continuity and Change in Nuclear Fusion Research from the late 1970s to the early 2000s: A Study of Social Fluidity and Lexical Recurrence*.

The two studies share the same upstream analytical pipeline for corpus construction, WoS parsing, normalization, author-identity resolution, corpus eligibility, and core network construction. They then use separate article-specific configurations and analytical routines.

The package is shared because the two studies rely on the same corpus and common preprocessing pipeline. Article-specific parameters and analyses are separated through configuration files and dedicated scripts.

For the Lalli article, the principal configuration file is:

`config/lalli.yaml`

For the Guzzardi–Lalli article, the principal configuration file is:

`config/guzzardi_lalli.yaml`

Shared settings are stored in:

`config/shared.yaml`

The mapping between manuscript results and code components is documented in:

`docs/METHODS_CODE_CROSSWALK.md`

**SOFTWARE VERSION:** 1.0.0 (pre-release preparation; no release tag or DOI yet). The private licensed-data analysis and final aggregate numerical acceptance were completed (563/563 checks matched) on macOS Apple Silicon with Python 3.13.5. The source repository is hosted at [GitHub](https://github.com/Rlalli74/fuseurope-analysis) and remains private pending final rights and public-release checks. Exact reproduction requires lawful access to the specified licensed inputs. See `docs/RELEASE_STATUS_AND_SCOPE.md`.

For step-by-step publication instructions in Italian, see `docs/GITHUB_PUBLICATION_GUIDE_IT.md`.

## AI-assisted development and author responsibility

Generative AI tools, including ChatGPT, were used to assist in the development, refinement, debugging, and documentation of portions of the analytical code. All methodological decisions, and responsibility for the accuracy, reliability, and reproducibility of the code and its results remain solely with the software author. The computational pipeline was tested and validated against reference results using the underlying research data.

## Main Workflow

The main end-to-end orchestration entry point is:

`scripts/run_pipeline.py`

The analytical implementations used by that runner are contained in:

`src/fusion_pipeline/`

The phase-specific scripts in `scripts/` call the same underlying functions, and the Jupyter notebooks in `notebooks/` provide interactive access to the same workflow rather than separate implementations.

## Two Corpus Layers

The **preserved retrieval corpus** contains 46,443 records. The final post-disambiguation derived representation has 46 columns, including `identity_institutions_normalized`; its geographic and institutional attributes use the documented C1/RP reconstruction. Author normalization and canonical identity construction are performed once on this full corpus. Publication-count and source-coverage analyses use this layer.

The **analytical corpus** contains 45,453 records after excluding 985 WoS `Editorial Material` records and five separately validated non-editorial event-summary records. It inherits the author-identity mapping constructed on the preserved corpus. Co-authorship, community, country, continuity, and title-vocabulary analyses use this analytical layer.

The pipeline never rebuilds author identity on the analytical subset.

## Reproducibility Without Licensed Web of Science Data

This package does not redistribute licensed Web of Science record-level data.

The repository contains the analytical code, configuration files, documented correction rules, synthetic tests, dependency specifications, and permitted aggregate reference values needed to inspect the workflow.

Researchers with lawful access to the required Web of Science source exports can run the full analytical workflow by supplying those files externally.

## Environment and Validation

Python 3.10 or later is supported by the development metadata.

The reported validation environment used Python 3.13.5 with the exact analytical package versions listed in:

`requirements-reproducibility.txt`

The corresponding Conda environment is:

`environment-reproducibility.yml`

To create a development environment and run the synthetic tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,notebook]'
pytest -q
python scripts/run_pipeline.py --mode reviewer_demo
jupyter notebook notebooks/00_reproducibility_check.ipynb
```

To install the historically reported pinned numerical versions, use either:

```bash
python3.13 -m pip install -r requirements-reproducibility.txt
```

or:

```bash
conda env create -f environment-reproducibility.yml
```

These tests exercise configuration, parsing, author-identity safeguards, corpus exclusion, network construction, deterministic Louvain settings, titleword processing, and the two JET representations using synthetic data only.

## Licensed Full Run

WoS exports and the private five-record exclusion decision list must be supplied from lawful external locations:

```bash
python scripts/run_pipeline.py \
  --mode licensed_full_run \
  --raw-dir /external/lawful/wos_exports \
  --additional-exclusions /external/private/five_event_summary_ids.txt \
  --work-dir /external/private/fuseurope_run
```

Physics and Engineering comparison series are not redistributed. When lawful copies are available, they can be supplied with the corresponding command-line options.

The frozen expected aggregate values used for numerical validation are stored in:

`acceptance/FINAL_REFERENCE_VALUES.yaml`

After the full run and separately required stages, assemble and validate the results using `docs/FULL_LICENSED_VALIDATION.md`. The validator accepts aggregate results and compares them with 563 final regression targets. No record-level WoS data are transferred to the public repository.

**Important:** `run_pipeline.py` does not call every separately required acceptance script. To finish a full licensed run, follow `docs/FULL_LICENSED_VALIDATION.md`. Synthetic-test success alone is not equivalent to matching the published quantitative tables.

See also:

- `docs/DATA_REQUIREMENTS.md`
- `docs/METHODS_CODE_CROSSWALK.md`
- `docs/REPRODUCIBILITY_SCOPE.md`

## Pre-push safety check

Run `python scripts/audit_public_repository.py` and personally review the proposed Git commit. The script checks common signs of restricted records, locally embedded credentials, record-level tables, and notebook saved outputs; it cannot independently guarantee that all material is legally redistributable.

The software is licensed under the **MIT License** (copyright 2026 Roberto Lalli; see `LICENSE`), subject to the author having the necessary rights to license the materials. This does **not** license or redistribute the underlying Web of Science data or any other excluded third-party records. See `CITATION.cff` for citation metadata and `docs/LICENSE_AND_CITATION_DECISIONS.md` for scope and remaining rights checks.

## Repository Structure

`src/fusion_pipeline/`  
Core analytical implementation.

`scripts/`  
Phase-specific command-line workflows and the main pipeline runner.

`notebooks/`  
Interactive interfaces to the same analytical functions.

`config/`  
Shared and article-specific analytical parameters.

`corrections/`  
Curated correction rules required by the analytical workflow.

`tests/`  
Synthetic and invariant tests.

`acceptance/`  
Frozen aggregate reference values used to validate the final results.

`docs/`  
Methodological and reproducibility documentation.

`requirements*.txt`, `environment*.yml`, `pyproject.toml`  
Software environment and dependency specifications.
