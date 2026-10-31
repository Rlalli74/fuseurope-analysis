#!/usr/bin/env python3
"""Build and scan a fail-closed, allowlist-only reviewer candidate ZIP."""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def read_rules(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]


def matches(path: str, pattern: str) -> bool:
    value = path.replace("\\", "/")
    return fnmatch.fnmatchcase(value, pattern) or PurePosixPath(value).match(pattern)


def expand_allowlist(root: Path, patterns: list[str]) -> list[Path]:
    selected = set()
    for pattern in patterns:
        for candidate in root.glob(pattern):
            if candidate.is_file():
                selected.add(candidate.resolve())
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def classification(relative: str) -> str:
    first = relative.split("/", 1)[0]
    return {
        "src": "analytical source code", "scripts": "workflow source code",
        "tests": "synthetic/public-safe tests", "notebooks": "output-free reviewer interface",
        "config": "public method configuration", "corrections": "author-created correction decisions",
        "docs": "reviewer documentation", "acceptance": "aggregate reference values",
    }.get(first, "project metadata")


def scan_file(path: Path, relative: str, max_bytes: int) -> list[str]:
    problems = []
    if path.stat().st_size > max_bytes:
        problems.append(f"oversized file ({path.stat().st_size} bytes)")
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    if "\x00" in text:
        problems.append("binary/NUL content")
    local_prefix = "/" + "Users" + "/"
    if local_prefix in text or ("Documents - " + "MacBook") in text:
        problems.append("local absolute path")
    real_wos_id = re.compile(r"\bWOS:[A-Z]\d[A-Z0-9]{8,}\b")
    if real_wos_id.search(text):
        problems.append("WoS record identifier")
    tagged_ids = re.findall(r"(?m)^UT\s+WOS:([^\s]+)", text)
    tagged_real = [value for value in tagged_ids if not re.match(r"^(?:TEST|SYNTHETIC)", value, re.I)]
    if re.search(r"(?m)^PT\s+", text) and re.search(r"(?m)^ER\s*$", text) and tagged_real:
        problems.append("WoS tagged-record signature")
    secret = re.compile(r"(?i)(?:api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{12,}")
    if secret.search(text):
        problems.append("possible embedded credential")
    if path.suffix.lower() in {".csv", ".tsv"} and text.strip():
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        header = next(csv.reader([text.splitlines()[0]], delimiter=delimiter), [])
        normalized = {re.sub(r"[^a-z0-9]", "", value.lower()) for value in header}
        combos = [
            {"ut", "ti", "so", "c1"},
            {"recordid", "title", "source", "address"},
            {"recordid", "title", "sourcetitle", "affiliations"},
        ]
        if any(combo.issubset(normalized) for combo in combos):
            problems.append("suspicious record-level table header")
    if path.suffix.lower() == ".ipynb":
        try:
            notebook = json.loads(text)
            if any(cell.get("outputs") for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"):
                problems.append("notebook contains saved outputs")
        except json.JSONDecodeError:
            problems.append("invalid notebook JSON")
    lower = text.lower()
    policy_file = relative.startswith("release_spec/")
    forbidden_terms = [
        "private" + "_data", "canonical" + "_corpus", "download" + "_",
        "wos" + "_analysis_snapshot", "explor" + "atory",
        "reference" + "_only", "old" + "_freeze", "seg" + "mented",
        "ai" + "ci" + "_seg" + "mented", ".pytest" + "_cache",
    ]
    obsolete_public_terms = [
        "core" + "_periphery", "core" + "-periphery",
        "v0." + "18", "aug" + "ust", "final" + "_freeze",
        "reference" + "_only",
    ]
    for term in forbidden_terms + obsolete_public_terms:
        if term in relative.lower() or (not policy_file and term in lower):
            problems.append(f"forbidden release term: {term}")
    if not policy_file and relative not in {
        "src/fusion_pipeline/internationalization.py",
        "src/fusion_pipeline/historical_figures.py",
    } and re.search(r"\b" + "ai" + "ci" + r"\b", lower):
        problems.append("obsolete reviewer-facing term: " + "ai" + "ci")
    return problems


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(args) -> Path:
    allowlist = read_rules(Path(args.allowlist))
    blocklist = read_rules(Path(args.blocklist))
    selected = [
        path for path in expand_allowlist(ROOT, allowlist)
        if not any(matches(path.relative_to(ROOT).as_posix(), rule) for rule in blocklist)
    ]
    if not selected:
        raise RuntimeError("Allowlist selected no files")
    violations = []
    for source in selected:
        relative = source.relative_to(ROOT).as_posix()
        if any(part in {"__py" + "cache__", ".ipynb_checkpoints", "private" + "_data", "private_inputs"} for part in PurePosixPath(relative).parts):
            violations.append(f"{relative}: prohibited path component")
        for problem in scan_file(source, relative, args.max_file_bytes):
            violations.append(f"{relative}: {problem}")
    if violations:
        raise RuntimeError("Release safety scan failed:\n- " + "\n- ".join(sorted(set(violations))))

    dist = Path(args.output_dir).resolve()
    dist.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fuseurope_reviewer_", dir=dist) as temporary:
        stage = Path(temporary) / "FusEurope_reproducibility_for_peer_review"
        stage.mkdir()
        for source in selected:
            relative = source.relative_to(ROOT)
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        report = stage / "REVIEWER_RELEASE_REPORT.md"
        report.write_text(
            "# Reviewer release report\n\n"
            "Status: **PASS — allowlist and content safety scans completed.**\n\n"
            "This confidential-review candidate contains shared source code, output-free notebooks, public method configuration, synthetic tests, curated author-created decisions, documentation, and aggregate reference values.\n\n"
            "It intentionally omits all WoS source records and analytical corpora; record-level metadata and audits; private identity-review evidence; GraphML provenance; unlicensed Physics/Engineering comparison series; generated analysis outputs; caches; and the obsolete lineage/entrant research branch.\n\n"
            "The manifest lists every payload file except itself (a manifest cannot contain its own stable hash). Human review of the manifest remains required. No publication, upload, tag, or DOI action was performed.\n",
            encoding="utf-8",
        )
        manifest = stage / "REVIEWER_PACKAGE_MANIFEST.csv"
        payload = sorted((path for path in stage.rglob("*") if path.is_file() and path != manifest), key=lambda path: path.relative_to(stage).as_posix())
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256", "release_classification"])
            writer.writeheader()
            for path in payload:
                relative = path.relative_to(stage).as_posix()
                writer.writerow({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path), "release_classification": classification(relative)})

        # Scan the final staged tree again, including generated report/manifest.
        final_violations = []
        for path in stage.rglob("*"):
            if path.is_file():
                relative = path.relative_to(stage).as_posix()
                for problem in scan_file(path, relative, args.max_file_bytes):
                    final_violations.append(f"{relative}: {problem}")
        if final_violations:
            raise RuntimeError("Final staging scan failed:\n- " + "\n- ".join(final_violations))

        archive = dist / args.name
        if archive.exists():
            archive.unlink()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    bundle.write(path, (Path(stage.name) / path.relative_to(stage)).as_posix())
        shutil.copy2(manifest, ROOT / "REVIEWER_RELEASE_MANIFEST.csv")
    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowlist", default=str(ROOT / "release_spec" / "REVIEWER_RELEASE_ALLOWLIST.txt"))
    parser.add_argument("--blocklist", default=str(ROOT / "release_spec" / "REVIEWER_RELEASE_BLOCKLIST.txt"))
    parser.add_argument("--output-dir", default=str(ROOT / "dist"))
    parser.add_argument("--name", default="FusEurope_reproducibility_for_peer_review_2026-09-23.zip")
    parser.add_argument("--max-file-bytes", type=int, default=5_000_000)
    args = parser.parse_args()
    archive = build(args)
    print(archive)


if __name__ == "__main__":
    main()
