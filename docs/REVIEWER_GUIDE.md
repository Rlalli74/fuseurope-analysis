# Reviewer guide

Start with `README.md`, then inspect `config/shared.yaml` and the article preset of interest. Run `pytest -q` and `notebooks/00_reproducibility_check.ipynb` without licensed data.

The exact environment used for the validation reported in the final
Supplementary Information is recorded in `requirements-reproducibility.txt`
and `environment-reproducibility.yml`: Python 3.13.5, pandas 2.2.3, NumPy
2.3.5, SciPy 1.17.0, NetworkX 3.6.1, statsmodels 0.14.6, scikit-learn 1.8.0,
matplotlib 3.10.8, and nbformat 5.10.4. The ordinary requirements remain
flexible for compatible development environments.

The package contains code, configurations, curated author-created normalization decisions, synthetic tests, and aggregate reference targets. It intentionally omits every WoS record and record-level derivative. Full numerical regeneration therefore requires lawful external WoS access; absence of those inputs produces a clear error rather than a substituted dataset.

The general control notebook accepts only project-relative configuration and explicit external input locations. Generated outputs should remain outside a public checkout. This is a confidential review candidate, not a DOI-bearing public archive.

The package also contains the allowlist and blocklist used by the release
builder. From an extracted package root, a referee can reconstruct and rescan a
candidate with `python scripts/build_reviewer_release.py --output-dir dist`.
