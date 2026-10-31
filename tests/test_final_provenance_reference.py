"""Acceptance contract: the single 46-column preserved derived corpus."""
import importlib.util
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

def test_single_final_corpus_reference_is_46_columns():
    reference = yaml.safe_load((ROOT/'acceptance/FINAL_REFERENCE_VALUES.yaml').read_text())
    settings = yaml.safe_load((ROOT/'config/shared.yaml').read_text())
    r = reference['shared']['preserved_retrieval_corpus']
    s = settings['corpora']['preserved_retrieval']
    assert r['records'] == r['unique_ut'] == 46443
    assert r['columns'] == s['columns'] == 46
    assert r['sha256'] == s['sha256']
    assert len(r['sha256']) == 64

def test_assembler_does_not_need_predecessor_input():
    script = (ROOT/'scripts/assemble_final_acceptance_results.py').read_text()
    assert '--frozen-preserved' not in script
    assert "03_disambiguation/preserved_identity_corpus_phase3.csv" in script
    assert 'identity_institutions_normalized' in script
