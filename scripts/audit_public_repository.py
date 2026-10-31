#!/usr/bin/env python3
"""Conservative content audit of a prospective PUBLIC Git repository.

This is a pre-push guard, not a proof of privacy or licence compliance.
Run from any folder. Checks files in the intended repository working tree.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest' + '_cache', '.ipynb_checkpoints', 'dist'}
PRIVATE_DIRS = {'private' + '_data', 'private_inputs', 'raw_data', 'data_private', 'pipeline_outputs', 'working_data', 'release_staging'}
PRIVATE_SUFFIXES = {'.parquet', '.pkl', '.pickle', '.graphml', '.gexf', '.sqlite', '.db'}
MAX_BYTES = 5_000_000


def scan(root: Path) -> tuple[list[str], int]:
    problems = []
    nfiles = 0
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        parts = rel.parts
        if any(part in SKIP_DIRS for part in parts):
            continue
        nfiles += 1
        label = rel.as_posix()
        if any(part in PRIVATE_DIRS for part in parts):
            problems.append(f'{label}: forbidden private directory')
            continue
        if path.suffix.lower() in PRIVATE_SUFFIXES or ('ut_only' in path.name.lower()):
            problems.append(f'{label}: forbidden record-level file type/name')
        if path.stat().st_size > MAX_BYTES:
            problems.append(f'{label}: unusually large file ({path.stat().st_size} bytes)')
        try:
            text = path.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            problems.append(f'{label}: non-UTF8 or binary payload')
            continue
        # Allow documentation to mention *patterns*, but not actual real UTs.
        if re.search(r'\bWOS:[A-Z][0-9][A-Z0-9]{8,}\b', text):
            problems.append(f'{label}: actual-looking Web of Science UT detected')
        if re.search(r'(?m)^PT\s+\w+\s*$[\s\S]*^ER\s*$', text):
            problems.append(f'{label}: apparent tagged WoS record')
        if re.search(r'/(Users|home/oai|mnt/data)/[^\s\"\']+', text):
            problems.append(f'{label}: local absolute file path')
        if re.search(r'(?i)(?:github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}|glpat-[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)', text):
            problems.append(f'{label}: possible credential/private key')
        if path.suffix.lower() == '.ipynb':
            try:
                nb=json.loads(text)
                for cell in nb.get('cells', []):
                    if cell.get('cell_type') == 'code' and (cell.get('outputs') or cell.get('execution_count') is not None):
                        problems.append(f'{label}: notebook has saved output or execution state')
                        break
            except ValueError:
                problems.append(f'{label}: malformed notebook')
        if path.suffix.lower()=='.csv' and text:
            header=text.splitlines()[0].lower()
            if ('record_id' in header and 'title' in header and ('address' in header or 'affiliation' in header)):
                problems.append(f'{label}: possible record-level bibliographic CSV')
    return problems, nfiles


def main() -> int:
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root', default=str(ROOT))
    args=a.parse_args()
    root=Path(args.root).resolve()
    problems, count=scan(root)
    print(f'Checked {count} files under {root}')
    if problems:
        print('FAIL - possible restricted content:')
        for problem in problems:
            print(' - '+problem)
        return 1
    print('PASS - automated sensitive-content patterns not detected; human review still required.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
