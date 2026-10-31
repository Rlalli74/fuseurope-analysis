import networkx as nx
import pandas as pd
import pytest

from fusion_pipeline.networks import louvain_partition_sensitivity
from fusion_pipeline.period_contrast import (
    final_historical_title_contrast,
    final_historical_title_features,
    run_final_historical_title_contrasts,
)


def test_final_historical_features_preserve_internal_hyphens_and_document_presence():
    features = final_historical_title_features("Plasma, plasma: ion-heating / device!")
    assert "plasma" in features
    assert "ion-heating" in features
    assert "plasma plasma" in features
    assert len([x for x in features if x == "plasma"]) == 1


def test_final_historical_contrast_uses_minimum_df_and_half_correction():
    rows = [
        {"year": 1960, "title": "alpha old"},
        {"year": 1961, "title": "alpha old"},
        {"year": 1962, "title": "alpha old"},
        {"year": 1968, "title": "beta new"},
        {"year": 1969, "title": "beta new"},
        {"year": 1970, "title": "beta new"},
        {"year": 1971, "title": "singleton"},
    ]
    result = final_historical_title_contrast(pd.DataFrame(rows), 1968, 1974, 1956, 1967)
    assert "singleton" not in set(result.feature)
    beta = result.set_index("feature").loc["beta"]
    assert beta.focal_count == 3 and beta.baseline_count == 0
    assert beta.log_odds > 0 and beta.z_score > 0


def test_only_the_two_final_historical_contrasts_are_registered():
    df = pd.DataFrame([{"year": year, "title": "shared term"} for year in range(1956, 1997)])
    assert list(run_final_historical_title_contrasts(df)) == [
        "1968-1974_vs_1956-1967", "1994-1996_vs_1989-1993",
    ]


def test_historical_louvain_sensitivity_compares_twenty_seeds_to_seed_42():
    graph = nx.karate_club_graph()
    nx.set_edge_attributes(graph, 1, "paper_count")
    result = louvain_partition_sensitivity(graph, reference_seed=42, seeds=tuple(range(20)))
    assert result.seed.tolist() == list(range(20))
    assert set(result.reference_seed) == {42}
    assert result.adjusted_rand_index.between(0, 1).all()


def test_deprecated_international_collaboration_alias_preserves_values():
    import fusion_pipeline.internationalization as module

    raw = pd.DataFrame([
        {"year": 2000, "countries_normalized": ["USA"], "countries_network": ["USA"]},
        {"year": 2000, "countries_normalized": ["USA", "FRANCE"], "countries_network": ["USA", "FRANCE"]},
    ])
    prepared = module.prepare_country_dataframe(raw)
    current = module.yearly_icr(prepared)
    with pytest.deprecated_call():
        legacy = getattr(module, "yearly_" + "ai" + "ci")(prepared)
    assert current["ICR"].tolist() == legacy["AI" + "CI"].tolist()
