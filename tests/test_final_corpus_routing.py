from pathlib import Path

import pandas as pd
import pytest

from fusion_pipeline.config import load_method_preset, validate_method_presets
from fusion_pipeline.corpus import (
    LicensedInputMissingError,
    derive_analytical_corpus,
    require_licensed_input,
    validate_identity_subset,
)
from fusion_pipeline.networks import _doc_type_allowed


ROOT = Path(__file__).resolve().parents[1]


def test_final_method_presets_are_complete_and_distinct():
    result = validate_method_presets(ROOT / "config")
    assert result["status"] == "pass"
    gl = load_method_preset(ROOT / "config", "guzzardi_lalli")
    la = load_method_preset(ROOT / "config", "lalli")
    assert gl["corpus_routing"]["publication_counts"] == "preserved_retrieval"
    assert gl["corpus_routing"]["networks"] == "analytical"
    assert la["country_reconstruction"]["sources"] == ["C1", "RP"]
    assert la["jet"]["modified_representation"] == "separate_JET_node"


def test_analytical_subset_excludes_editorials_and_private_five_rule_without_rebuilding_identity():
    preserved = pd.DataFrame([
        {"record_id": "P1", "document_types": ["Article"],
         "authors_normalized": ["SMITH, A"], "authors_disambiguated": ["SMITH, A"], "authors_network": ["SMITH, A"]},
        {"record_id": "P2", "document_types": ["Editorial Material"],
         "authors_normalized": ["JONES, B"], "authors_disambiguated": ["JONES, B"], "authors_network": ["JONES, B"]},
        {"record_id": "P3", "document_types": ["Article", "Proceedings Paper"],
         "authors_normalized": ["SMYTH, A"], "authors_disambiguated": ["SMITH, A"], "authors_network": ["SMITH, A"]},
        {"record_id": "P4", "document_types": ["Meeting Abstract"],
         "authors_normalized": ["DOE, C"], "authors_disambiguated": ["DOE, C"], "authors_network": ["DOE, C"]},
    ])
    analytical, audit, private = derive_analytical_corpus(
        preserved, {"P3"}, strict_final_counts=False,
    )
    assert analytical.record_id.tolist() == ["P1", "P4"]
    assert audit.editorial_material_excluded == 1
    assert audit.additional_non_editorial_excluded == 1
    assert set(private.exclusion_reason) == {
        "wos_editorial_material", "validated_non_editorial_event_summary"
    }
    identity = validate_identity_subset(preserved, analytical)
    assert identity["status"] == "pass" and identity["changed_mappings"] == 0


def test_identity_subset_handles_duplicate_occurrences_before_network_deduplication():
    preserved = pd.DataFrame([{
        "record_id": "P1", "document_types": ["Article"],
        "authors_normalized": ["SMITH, A", "SMYTH, A", "SMITH, A"],
        "authors_disambiguated": ["SMITH, A", "SMITH, A", "SMITH, A"],
        "authors_network": ["SMITH, A"],
    }])
    identity = validate_identity_subset(preserved, preserved.copy())
    assert identity["preserved_normalized_forms"] == 2
    assert identity["analytical_canonical_identities"] == 1


def test_licensed_input_missing_error_is_explanatory(tmp_path):
    with pytest.raises(LicensedInputMissingError, match="No Web of Science record-level data"):
        require_licensed_input(tmp_path / "not-present.csv", "the preserved corpus")


def test_final_document_type_modes_describe_the_already_filtered_corpus():
    assert _doc_type_allowed(["Meeting Abstract"], "unrestricted_analytical_corpus")
    assert not _doc_type_allowed(["Meeting Abstract"], "exclude_meeting_abstracts")
    assert not _doc_type_allowed(
        ["Article", "Proceedings Paper"], "exclude_proceedings_and_meeting_abstracts"
    )
    assert _doc_type_allowed(["Article"], "articles_only")
    assert not _doc_type_allowed(["Article", "Proceedings Paper"], "articles_only")
