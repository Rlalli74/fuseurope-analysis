import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_reviewer_release.py"
SPEC = importlib.util.spec_from_file_location("build_reviewer_release", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_release_scanner_rejects_record_level_header_and_real_wos_id(tmp_path):
    suspect = tmp_path / "records.csv"
    suspect.write_text("record_id,title,source,address\n" + "WO" + "S:A1234567890,t,s,a\n", encoding="utf-8")
    problems = MODULE.scan_file(suspect, "records.csv", 1_000_000)
    assert "suspicious record-level table header" in problems
    assert "WoS record identifier" in problems


def test_release_scanner_accepts_synthetic_source_and_output_free_notebook(tmp_path):
    source = tmp_path / "test_fixture.py"
    source.write_text("sample = 'PT J\\nUT WOS:TEST001\\nER\\n'\n", encoding="utf-8")
    assert MODULE.scan_file(source, "tests/test_fixture.py", 1_000_000) == []
    notebook = tmp_path / "check.ipynb"
    notebook.write_text('{"cells":[{"cell_type":"code","outputs":[]}]}', encoding="utf-8")
    assert MODULE.scan_file(notebook, "notebooks/check.ipynb", 1_000_000) == []


def test_blocklist_matching_is_fail_closed_for_retired_script():
    retired = "scripts/16_guzzardi_lalli_longitudinal_" + "explor" + "atory.py"
    assert MODULE.matches(retired, retired)


def test_reviewer_facing_notebooks_are_output_free_thin_launchers():
    root = Path(__file__).resolve().parents[1]
    forbidden = ["seg" + "mented", "all" + "_selected", "/" + "Users" + "/"]
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        if path.name == "08_" + "explor" + "atory_analysis.ipynb":
            continue
        notebook = json.loads(path.read_text(encoding="utf-8"))
        text = path.read_text(encoding="utf-8")
        assert not any(term in text for term in forbidden)
        assert all(not cell.get("outputs") for cell in notebook["cells"] if cell.get("cell_type") == "code")


def test_release_scanner_rejects_saved_notebook_outputs(tmp_path):
    notebook = tmp_path / "bad.ipynb"
    notebook.write_text('{"cells":[{"cell_type":"code","outputs":[{"output_type":"stream","text":"private"}]}]}', encoding="utf-8")
    assert "notebook contains saved outputs" in MODULE.scan_file(notebook, "notebooks/bad.ipynb", 1_000_000)


def test_release_scanner_rejects_local_user_paths(tmp_path):
    source = tmp_path / "paths.md"
    source.write_text("/" + "Users" + "/someone/private/project", encoding="utf-8")
    assert "local absolute path" in MODULE.scan_file(source, "docs/paths.md", 1_000_000)


def test_exact_reproducibility_environment_matches_reported_validation_stack():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "python": "3.13.5", "pandas": "2.2.3", "numpy": "2.3.5",
        "scipy": "1.17.0", "networkx": "3.6.1", "statsmodels": "0.14.6",
        "scikit-learn": "1.8.0", "matplotlib": "3.10.8", "nbformat": "5.10.4",
    }
    requirements = {}
    for line in (root / "requirements-reproducibility.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            requirements[name.lower()] = version
    environment = yaml.safe_load((root / "environment-reproducibility.yml").read_text(encoding="utf-8"))
    conda = {}
    for item in environment["dependencies"]:
        if isinstance(item, str) and "=" in item:
            name, version = item.split("=", 1)
            conda[name.lower()] = version
    assert conda == expected | {
        "graphviz": "14.1.2", "pygraphviz": "2.0.2", "pyyaml": "6.0.3",
        "nbclient": "0.10.2", "pytest": "9.0.2", "jupyter": "1.1.1",
    }
    assert requirements == {key: value for key, value in expected.items() if key != "python"}


def test_release_has_policy_inputs_and_extracted_builder_uses_them(tmp_path):
    archive_dir = tmp_path / "first"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(archive_dir), "--name", "candidate.zip"],
        check=True, capture_output=True, text=True,
    )
    archive = Path(result.stdout.strip().splitlines()[-1])
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(extracted)
    package = next(extracted.iterdir())
    assert (package / "release_spec/REVIEWER_RELEASE_ALLOWLIST.txt").is_file()
    assert (package / "release_spec/REVIEWER_RELEASE_BLOCKLIST.txt").is_file()
    rebuilt = tmp_path / "rebuilt"
    subprocess.run(
        [sys.executable, "scripts/build_reviewer_release.py", "--output-dir", str(rebuilt), "--name", "rebuilt.zip"],
        cwd=package, check=True, capture_output=True, text=True,
    )
    assert (rebuilt / "rebuilt.zip").is_file()


def test_public_release_sources_have_current_icr_and_no_obsolete_labels():
    root = Path(__file__).resolve().parents[1]
    allow = MODULE.read_rules(root / "release_spec/REVIEWER_RELEASE_ALLOWLIST.txt")
    block = MODULE.read_rules(root / "release_spec/REVIEWER_RELEASE_BLOCKLIST.txt")
    selected = [
        path for path in MODULE.expand_allowlist(root, allow)
        if not any(MODULE.matches(path.relative_to(root).as_posix(), rule) for rule in block)
    ]
    violations = []
    for path in selected:
        relative = path.relative_to(root).as_posix()
        violations.extend(f"{relative}: {item}" for item in MODULE.scan_file(path, relative, 5_000_000))
    assert violations == []
