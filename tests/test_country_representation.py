import pandas as pd

from fusion_pipeline.internationalization import (
    centered_pooled_ratio,
    cocountry_brokerage,
    cocountry_jet_geographic_benchmark,
    european_indicator_components,
    pool_european_indicators,
    prepare_country_dataframe,
)


def test_centered_icr_smoothing_pools_counts_and_has_no_shortened_endpoints():
    annual = pd.DataFrame({
        "year": [2000, 2001, 2002, 2003, 2004],
        "international_papers": [1, 9, 1, 9, 1],
        "papers_with_country": [10, 10, 100, 10, 10],
    })
    out = centered_pooled_ratio(
        annual, "international_papers", "papers_with_country",
        window=3, output_name="ICR",
    )
    assert out.year.tolist() == [2001, 2002, 2003]
    assert out.loc[out.year.eq(2002), "ICR"].iloc[0] == 19 / 120
    assert out.loc[out.year.eq(2002), "ICR"].iloc[0] != (0.9 + 0.01 + 0.9) / 3


def _country_fixture():
    rows = [
        ("P1", ["FRANCE"], ["FRANCE"]),
        ("P2", ["FRANCE", "USA"], ["FRANCE", "USA"]),
        ("P3", ["FRANCE", "GERMANY"], ["FRANCE", "GERMANY"]),
        ("P4", ["USA", "JAPAN"], ["USA", "JAPAN"]),
        ("P5", ["UK"], ["JET"]),
        ("P6", ["UK", "USA"], ["JET", "USA"]),
    ]
    return prepare_country_dataframe(pd.DataFrame([
        {
            "record_id": record_id, "year": 1986,
            "countries_normalized": geo, "countries_network": modified,
            "authors_network": [record_id], "institutions": [],
            "document_types": ["Article"], "affiliations": [],
        }
        for record_id, geo, modified in rows
    ]))


def test_final_european_indicators_use_geographic_dynamic_membership():
    annual = european_indicator_components(_country_fixture(), start=1986, end=1986)
    row = annual.iloc[0]
    # France domestic 1; France-USA 1/2; France-Germany 1; UK domestic 1;
    # JET-USA is geographically UK-USA and contributes 1/2.
    assert row.european_fractional_output_share == 4.0 / 6.0
    # European participations on international papers: FR, FR+DE, UK = 4;
    # only FR+DE has a second European partner, contributing two exposures.
    assert row.conditional_european_partner_share == 2.0 / 4.0
    assert row.oe_european_affinity > 0


def test_period_affinity_pools_observed_and_expected_before_ratio():
    annual = european_indicator_components(_country_fixture(), start=1986, end=1986)
    pooled = pool_european_indicators(annual, [("1986", 1986, 1986)]).iloc[0]
    assert pooled.oe_european_affinity == (
        pooled.european_observed_dyadic_weight / pooled.european_expected_dyadic_weight
    )


def test_modified_cocountry_brokerage_uses_unweighted_lcc_and_geographic_benchmark():
    data = _country_fixture()
    graph, lcc, table = cocountry_brokerage(data, 1986, 1986, modified_jet_representation=True)
    assert "JET" in graph and "JET" in lcc
    assert set(table.betweenness_topology) == {"unweighted_LCC"}
    assert graph["JET"]["USA"]["paper_count"] == 1
    assert graph["JET"]["USA"]["fractional_weight"] == 1.0
    benchmark = cocountry_jet_geographic_benchmark(data, 1986, 1986)
    assert {"modified_betweenness", "geographic_betweenness"}.issubset(benchmark.columns)
