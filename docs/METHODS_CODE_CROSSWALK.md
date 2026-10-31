# Methods-to-code crosswalk

| Final method | Corpus | Configuration | Primary implementation / entry point |
|---|---|---|---|
| Retrieval preservation and author identity | Preserved, 46,443 | `shared.yaml` | `wos_parser.py`, `normalization.py`, `disambiguation.py`, scripts 01–03b |
| Analytical eligibility and unchanged identity map | Preserved → analytical, 45,453 | `shared.yaml` | `corpus.py`, script 03c |
| Publication counts and relative trajectories | Preserved | `guzzardi_lalli.yaml` | `publication_trends.py`, script 11 |
| Guzzardi–Lalli co-authorship snapshots | Analytical | `guzzardi_lalli.yaml` | `networks.py`, scripts 04b and 13 |
| Louvain communities and seed checks | Analytical | both article presets | `networks.py`, scripts 06b and 14 |
| Community titlewords and 1/q allocation | Analytical | `guzzardi_lalli.yaml` | `socio_epistemic.py`, scripts 06, 14, and 15 |
| Historical title contrasts | Analytical | `guzzardi_lalli.yaml` | `period_contrast.py`, script 10 |
| Historical author-network periods and robustness | Analytical | `lalli.yaml` | `networks.py`, scripts 04b, 04c, and 06b |
| C1 union RP reconstruction and prefix handling | Analytical | `lalli.yaml` | `normalization.py`, `institutions.py` |
| ICR, Wilson intervals, pooled smoothing | Analytical | `lalli.yaml` | `internationalization.py`, scripts 05 and 09 |
| Modified JET-node graphs / geographic benchmark | Analytical | `lalli.yaml` | `internationalization.py`, script 05 |
| European output, partners, O/E, enlargement sensitivity | Analytical | `lalli.yaml` | `internationalization.py`, script 05 |
| Numerical regression | Aggregate results only | acceptance YAML | `validate_final_papers.py` |
| Reviewer-safe archive | Public allowlist only | release specs | `build_reviewer_release.py` |

Compatibility names cited in the final SI are retained. Obsolete Guzzardi–Lalli lineage/entrant code is excluded from the reviewer package.
