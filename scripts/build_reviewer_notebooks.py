#!/usr/bin/env python3
"""Regenerate the two output-free, project-relative reviewer notebooks."""
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "id": hashlib.sha1(("md:" + text).encode()).hexdigest()[:12], "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "id": hashlib.sha1(("code:" + text).encode()).hexdigest()[:12], "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


def write(name, cells):
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
    (ROOT / "notebooks" / name).write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


write("00_pipeline_control_and_network_explorer.ipynb", [
    md("# FusEurope pipeline control and network explorer\n\nOne configuration-driven control interface. Licensed WoS inputs remain external; this notebook does not embed private paths or data."),
    code("from pathlib import Path\nimport sys\nROOT = Path.cwd().resolve()\nif ROOT.name == 'notebooks': ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT / 'src'))\nfrom fusion_pipeline.config import load_method_preset, validate_method_presets\nvalidate_method_presets(ROOT / 'config')"),
    md("## User settings\n\nSet external paths only in your local execution state. Do not save licensed-data paths or cell outputs into a release copy."),
    code("ARTICLE = 'guzzardi_lalli'  # or 'lalli'\nMODE = 'reviewer_demo'  # or 'licensed_full_run'\nEXTERNAL_RAW_DIR = None\nPRIVATE_ADDITIONAL_EXCLUSIONS = None\nWORK_DIR = ROOT / 'pipeline_outputs'\nCONFIG = load_method_preset(ROOT / 'config', ARTICLE)\nCONFIG"),
    code("from fusion_pipeline.corpus import require_licensed_input\nif MODE == 'licensed_full_run':\n    raw_dir = require_licensed_input(EXTERNAL_RAW_DIR, 'lawfully accessed WoS exports')\n    exclusions = require_licensed_input(PRIVATE_ADDITIONAL_EXCLUSIONS, 'private five-record exclusion list')\nelse:\n    print('Reviewer demo mode: no licensed input is read.')"),
    md("## Launch\n\nFor an auditable complete run, use `python scripts/run_pipeline.py --mode licensed_full_run --raw-dir ... --additional-exclusions ...`. Article functions are shared library calls; this notebook contains no parallel analytical implementation."),
])

write("00_reproducibility_check.ipynb", [
    md("# Reviewer reproducibility check\n\nPublic-safe smoke checks using only synthetic records. The same package functions are used by the full pipeline."),
    code("from pathlib import Path\nimport sys, platform, tempfile\nimport pandas as pd, networkx as nx\nROOT = Path.cwd().resolve()\nif ROOT.name == 'notebooks': ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT / 'src'))\nimport fusion_pipeline\nprint({'python': platform.python_version(), 'pandas': pd.__version__, 'networkx': nx.__version__})"),
    code("from fusion_pipeline.config import validate_method_presets\nassert validate_method_presets(ROOT / 'config')['status'] == 'pass'"),
    code("from fusion_pipeline.wos_parser import parse_wos_file\ntext = 'PT J\\nAU Smith, AB\\n   Jones, CD\\nTI Synthetic fusion title\\nDT Article\\nPY 1986\\nUT SYNTHETIC\\nER\\n'\nwith tempfile.TemporaryDirectory() as tmp:\n    path = Path(tmp) / 'synthetic.txt'; path.write_text(text)\n    parsed = parse_wos_file(path)\nassert len(parsed) == 1 and parsed[0]['authors_wos'] == ['Smith, AB', 'Jones, CD']"),
    code("from fusion_pipeline.corpus import derive_analytical_corpus, validate_identity_subset\nrecords = pd.DataFrame([\n {'record_id':'a','document_types':['Article'],'authors_normalized':['SMITH, AB'],'authors_disambiguated':['SMITH, AB'],'authors_network':['SMITH, AB']},\n {'record_id':'e','document_types':['Editorial Material'],'authors_normalized':['JONES, CD'],'authors_disambiguated':['JONES, CD'],'authors_network':['JONES, CD']},\n {'record_id':'x','document_types':['Meeting Abstract'],'authors_normalized':['LEE, EF'],'authors_disambiguated':['LEE, EF'],'authors_network':['LEE, EF']},])\nanalytical, audit, private = derive_analytical_corpus(records, {'x'})\nassert analytical.record_id.tolist() == ['a']\nassert validate_identity_subset(records, analytical)['changed_mappings'] == 0"),
    code("from fusion_pipeline.networks import prepare_network_dataframe, build_coauthorship_graph, largest_component_graph, louvain_partition\npapers = prepare_network_dataframe(pd.DataFrame([\n {'record_id':'p1','year':1986,'authors_network':['A','B','C'],'document_types':['Article']},\n {'record_id':'p2','year':1986,'authors_network':['C','D'],'document_types':['Article']}]))\ngraph = build_coauthorship_graph(papers); lcc = largest_component_graph(graph)\nassert graph['A']['B']['fractional_weight'] == 0.5\nassert louvain_partition(lcc, seed=42, weight='paper_count') == louvain_partition(lcc, seed=42, weight='paper_count')"),
    code("from fusion_pipeline.socio_epistemic import tokenize_title, community_titleword_profiles\nassert tokenize_title('Electron-cyclotron heating', set(), {}) == ['electron-cyclotron','heating']\nsocial = (papers, graph, lcc, {'A':0,'B':0,'C':1,'D':1})\nprofiles = community_titleword_profiles(papers.assign(title=['alpha','beta']), 1986, set(), {}, social=social)\nassignment = profiles['paper_assignments'].iloc[0]\nassert assignment.n_communities == 2 and '0:0.5' in assignment.community_weights and '1:0.5' in assignment.community_weights"),
    code("from fusion_pipeline.normalization import build_affiliation_table\njet = build_affiliation_table([{'record_id':'r','year':1986,'affiliations':[], 'reprint_address_one_line':'Smith, AB (reprint author), JET Joint Undertaking, Abingdon, England'}])\nassert jet.iloc[0].country_normalized == 'UK' and jet.iloc[0].network_country == 'JET'"),
    code("from fusion_pipeline.corpus import require_licensed_input, LicensedInputMissingError\ntry:\n    require_licensed_input(None, 'synthetic licensed input')\nexcept LicensedInputMissingError as exc:\n    assert 'No Web of Science record-level data are bundled' in str(exc)\nelse:\n    raise AssertionError('missing licensed input must fail clearly')\nprint('All reviewer smoke checks passed.')"),
])

# Article/phase notebooks are deliberately thin launchers.  They expose no
# second implementation and save no private paths or outputs in the notebook.
PHASE_NOTEBOOKS = {
    "01_build_and_validate_corpus.ipynb": ("01_build_and_validate_corpus.py", "preserved retrieval corpus", "--raw-dir", "--output-dir"),
    "02_normalize_authors_affiliations_countries.ipynb": ("02_normalize_authors_affiliations_countries.py", "preserved retrieval corpus", None, "--output-dir"),
    "03_disambiguate_authors.ipynb": ("03_disambiguate_authors.py", "preserved retrieval corpus", None, "--output-dir"),
    "04a_network_diagnostics.ipynb": ("04a_network_diagnostics.py", "analytical corpus", None, "--output-dir"),
    "04b_snapshot_networks.ipynb": ("04b_snapshot_networks.py", "analytical corpus", None, "--output-dir"),
    "04c_network_robustness.ipynb": ("04c_network_robustness.py", "analytical corpus", None, "--output-dir"),
    "05_internationalization_and_cocountry.ipynb": ("05_internationalization_and_cocountry.py", "analytical corpus", None, "--output-dir"),
    "06_socio_epistemic_analysis.ipynb": ("06_socio_epistemic_analysis.py", "analytical corpus", None, "--output-dir"),
    "06b_louvain_seed_robustness.ipynb": ("06b_louvain_seed_robustness.py", "analytical corpus", None, "--output-dir"),
    "07_institution_audit.ipynb": ("07_institution_audit.py", "analytical corpus", None, "--output-dir"),
    "09_historical_paper_figures.ipynb": ("09_historical_paper_figures.py", "analytical corpus", None, "--output-dir"),
    "10_period_contrast_explorer.ipynb": ("10_period_contrast_explorer.py", "analytical corpus", None, "--output-dir"),
    "11_publication_trend_comparison.ipynb": ("11_publication_trend_comparison.py", "preserved retrieval corpus plus external comparison series", "--fusion-corpus", "--output-dir"),
    "12_historical_annotation_layer.ipynb": ("12_historical_annotation_layer.py", "analytical corpus", None, "--output"),
    "13_guzzardi_lalli_network_figures.ipynb": ("13_guzzardi_lalli_network_figures.py", "analytical corpus", None, "--output-dir"),
}
for notebook_name, (script_name, corpus_role, input_option, output_option) in PHASE_NOTEBOOKS.items():
    write(notebook_name, [
        md(f"# {notebook_name.removesuffix('.ipynb')}\n\nThin, output-free launcher for `{script_name}`. Final corpus routing: **{corpus_role}**. Analytical logic remains in the shared package and script."),
        code("from pathlib import Path\nimport subprocess, sys\nROOT = Path.cwd().resolve()\nif ROOT.name == 'notebooks': ROOT = ROOT.parent\nSCRIPT = ROOT / 'scripts' / " + repr(script_name) + "\nINPUT_OPTION = " + repr(input_option) + "\nOUTPUT_OPTION = " + repr(output_option) + "\nINPUT_PATH = None  # set to an explicit lawful external input\nOUTPUT_PATH = None  # keep generated/private outputs outside the reviewer checkout\nEXTRA_ARGS = []\nRUN = False"),
        code("if RUN:\n    if INPUT_PATH is None or OUTPUT_PATH is None:\n        raise ValueError('Set explicit INPUT_PATH and OUTPUT_PATH before a licensed run')\n    input_args = [INPUT_OPTION, str(INPUT_PATH)] if INPUT_OPTION else [str(INPUT_PATH)]\n    command = [sys.executable, str(SCRIPT), *input_args, OUTPUT_OPTION, str(OUTPUT_PATH), *EXTRA_ARGS]\n    subprocess.run(command, check=True)\nelse:\n    print('Not run: configure explicit external input/output paths and set RUN=True.')"),
    ])

print("Reviewer notebooks regenerated without outputs or private paths.")
