# Release status and reproducibility scope — 31 October 2026

**Status: validated source repository, version metadata prepared as 1.0.0; first public GitHub release and Zenodo DOI NOT yet created.** The code is currently hosted in a private repository at https://github.com/Rlalli74/fuseurope-analysis.

## Completed research validation

The full licensed-data pipeline ran on Apple Silicon/macOS using the original dated source exports, restricted five-event exclusion decisions and lawful Physics/Engineering comparison files. All manuscript analysis and figure-generation phases completed. The final private verification produced **563/563** aggregate/provenance acceptance checks. An independent record-level consistency audit verified the 46,443 record set and C1/RP derived-field reproduction; it did not independently evaluate every individual historic address.

The definitive retained retrieval-level derived representation has **46 columns** and **46,443 records**; the analytical subset has **45,453 records**. Identity mapping is built once from the full retrieval set and inherited by the analytical set.

## Source-only validation and distribution limits

- 69 public synthetic/invariant tests were passed on the source candidate; public demonstration and sensitive-file-pattern scans were also passed.
- The 46-column candidate was checked again against saved private analytical outputs on the researcher's Mac: 563/563 acceptance checks passed.
- The repository does not include licensed WoS source exports, per-record corpora, private identity adjudication inputs, historical private audits or third-party comparison data. An external researcher must have lawful access to the specified inputs to reproduce the full numerical analysis.

## Publication preparations

Sole software author: **Roberto Lalli**. MIT licence: **2026 Roberto Lalli**, supplied in `LICENSE`; rights over curated corrections and any third-party components still require confirmation before public visibility. The source has a `CITATION.cff`, but **no GitHub tag/release or Zenodo DOI has yet been issued**. Before creating the release: inspect the entire Git history for accidentally committed private material, validate source-only tests, make the repository public, enable its integration in Zenodo and only then publish the first GitHub release.
