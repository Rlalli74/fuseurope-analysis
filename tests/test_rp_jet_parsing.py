from fusion_pipeline.normalization import (
    add_country_lists_to_records,
    build_affiliation_table,
    is_jet_affiliation,
    strip_rp_author_prefix,
)


def test_rp_corresponding_author_prefix_is_removed_before_jet_detection():
    raw = "Smith, AB (reprint author), JET Joint Undertaking, Abingdon, England."
    assert strip_rp_author_prefix(raw).startswith("JET Joint Undertaking")
    assert is_jet_affiliation(raw)


def test_rp_prefix_fix_preserves_jet_false_positive_guards():
    assert not is_jet_affiliation(
        "Doe, J (corresponding author), Jet Propulsion Laboratory, Pasadena, CA 91109, USA."
    )
    assert not is_jet_affiliation(
        "Doe, J (reprint author), Aerojet, Sacramento, CA, USA."
    )


def test_country_and_modified_jet_sets_are_deduplicated_c1_union_rp():
    records = [{
        "record_id": "R1",
        "year": 1986,
        "affiliations": ["Princeton Plasma Physics Laboratory, Princeton, NJ 08544, USA."],
        "reprint_address_one_line": (
            "Smith, AB (reprint author), JET Joint Undertaking, Abingdon, England."
        ),
    }]
    affiliations = build_affiliation_table(records)
    assert set(affiliations.address_source) == {"C1", "RP"}
    assert int(affiliations.rp_prefix_stripped.sum()) == 1
    enriched = add_country_lists_to_records(records, affiliations)[0]
    assert enriched["countries_normalized"] == ["USA", "UK"]
    assert enriched["countries_network"] == ["USA", "JET"]


def test_exact_duplicate_rp_address_does_not_duplicate_c1_evidence():
    address = "JET Joint Undertaking, Abingdon, England."
    records = [{
        "record_id": "R2", "year": 1986, "affiliations": [address],
        "reprint_address_one_line": f"Smith, AB (reprint author), {address}",
    }]
    affiliations = build_affiliation_table(records)
    assert len(affiliations) == 1
