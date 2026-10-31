#!/usr/bin/env python3
"""Convenience orchestrator for the canonical FusEurope pipeline.

The individual phase scripts remain the authoritative entry points. This runner
simply calls them in order so the same workflow can be reproduced from a clean
working directory. Slow robustness analyses are optional.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PYTHON=sys.executable

def run(cmd):
    print("\n>>>", " ".join(map(str,cmd)), flush=True)
    env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1'
    env.setdefault('MPLBACKEND', 'Agg')  # save figures without opening GUI windows
    env.setdefault('MPLBACKEND', 'Agg')  # save figures without opening GUI windows
    subprocess.run([str(x) for x in cmd], check=True, env=env)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['licensed_full_run','reviewer_demo'],default='licensed_full_run')
    ap.add_argument('--raw-dir',help='External folder containing lawful WoS plain-text exports')
    ap.add_argument('--additional-exclusions',help='Private file containing the five validated non-Editorial record identifiers')
    ap.add_argument('--physics-baseline',help='External annual Physics comparison series; redistribution rights are not assumed')
    ap.add_argument('--engineering-baseline',help='External annual Engineering comparison series; redistribution rights are not assumed')
    ap.add_argument('--work-dir',default='pipeline_outputs',help='Directory in which phase outputs are created')
    ap.add_argument('--historical-csv',default=None,help='Optional historical analysis CSV, used only for validation comparisons')
    ap.add_argument('--legacy-analysis-dir',default=None,help='Optional legacy figure/output directory, used only for audit comparisons')
    ap.add_argument('--skip-ingestion-validation',action='store_true',help='Skip phase 01 (phase 02 reparses the same raw files)')
    ap.add_argument('--skip-slow-robustness',action='store_true',help='Skip phase 04c team-size/document-type robustness grid')
    ap.add_argument('--skip-socio-epistemic',action='store_true',help='Skip phase 06')
    ap.add_argument('--skip-institution-audit',action='store_true',help='Skip phase 07')
    args=ap.parse_args()
    work=Path(args.work_dir); work.mkdir(parents=True,exist_ok=True)

    if args.mode == 'reviewer_demo':
        run([PYTHON,'-m','pytest','-q','-p','no:cacheprovider'])
        run([PYTHON,'-c',
             f"import sys; sys.path.insert(0, r'{ROOT / 'src'}'); "
             "from fusion_pipeline.config import validate_method_presets; "
             f"print(validate_method_presets(r'{ROOT / 'config'}'))"])
        print("Reviewer demo completed without licensed inputs.")
        return
    if not args.raw_dir:
        ap.error('--raw-dir is required for licensed_full_run; licensed inputs are not bundled')
    if not args.additional_exclusions:
        ap.error('--additional-exclusions is required for licensed_full_run and must point to the private five-record decision list')

    if not args.skip_ingestion_validation:
        cmd=[PYTHON,ROOT/'scripts/01_build_and_validate_corpus.py','--raw-dir',args.raw_dir,'--output-dir',work/'01_ingestion']
        if args.historical_csv: cmd += ['--historical-csv',args.historical_csv]
        run(cmd)

    cmd=[PYTHON,ROOT/'scripts/02_normalize_authors_affiliations_countries.py',args.raw_dir,'--output-dir',work/'02_normalization']
    if args.historical_csv: cmd += ['--historical-csv',args.historical_csv]
    run(cmd)
    phase2=work/'02_normalization/preserved_corpus_phase2.csv'

    cmd=[PYTHON,ROOT/'scripts/03_disambiguate_authors.py',phase2,
         '--manual-overrides',ROOT/'corrections/author_identity_overrides.csv',
         '--output-dir',work/'03_disambiguation']
    if args.legacy_analysis_dir: cmd += ['--legacy-analysis-dir',args.legacy_analysis_dir]
    run(cmd)
    phase3=work/'03_disambiguation/preserved_identity_corpus_phase3.csv'

    # Validate curated identity decisions and close the phase-2 multiple-candidate
    # dot-truncation audit before any downstream network analysis.
    run([PYTHON,ROOT/'scripts/03b_validate_author_identity_review.py',phase3,
         '--overrides',ROOT/'corrections/author_identity_overrides.csv',
         '--dot-truncation-audit',work/'02_normalization/author_dot_truncation_audit.csv',
         '--output-dir',work/'03b_identity_review_validation'])

    run([PYTHON,ROOT/'scripts/03c_build_analytical_corpus.py',phase3,
         '--additional-exclusions',args.additional_exclusions,
         '--output-dir',work/'03c_analytical_corpus'])
    analytical=work/'03c_analytical_corpus/analytical_corpus_phase3c.csv'

    run([PYTHON,ROOT/'scripts/04a_network_diagnostics.py',analytical,'--output-dir',work/'04a_network_diagnostics'])
    run([PYTHON,ROOT/'scripts/04b_snapshot_networks.py',analytical,'--output-dir',work/'04b_snapshot_networks','--analyses','metrics,centralities,communities,graph_export','--graph-attribute-profile','manuscript'])
    if not args.skip_slow_robustness:
        run([PYTHON,ROOT/'scripts/04c_network_robustness.py',analytical,'--output-dir',work/'04c_network_robustness'])

    run([PYTHON,ROOT/'scripts/05_internationalization_and_cocountry.py',analytical,'--output-dir',work/'05_internationalization','--analyses','icr,cocountry,european'])
    if not args.skip_socio_epistemic:
        run([PYTHON,ROOT/'scripts/06_socio_epistemic_analysis.py',analytical,'--output-dir',work/'06_socio_epistemic','--config-dir',ROOT/'config'])
        if not args.skip_slow_robustness:
            run([PYTHON,ROOT/'scripts/06b_louvain_seed_robustness.py',analytical,'--output-dir',work/'06b_louvain_seed_robustness','--config-dir',ROOT/'config'])
    if not args.skip_institution_audit:
        run([PYTHON,ROOT/'scripts/07_institution_audit.py',analytical,'--output-dir',work/'07_institutions','--aliases',ROOT/'config/institution_aliases.csv'])

    # Article-facing procedures. Publication/source coverage uses the preserved
    # retrieval corpus; every network and lexical procedure uses the analytical
    # corpus produced after identity construction.
    run([PYTHON,ROOT/'scripts/09_historical_paper_figures.py',analytical,'--output-dir',work/'09_lalli_figures'])
    run([PYTHON,ROOT/'scripts/10_period_contrast_explorer.py',analytical,'--output-dir',work/'10_historical_title_contrasts'])
    run([PYTHON,ROOT/'scripts/13_guzzardi_lalli_network_figures.py',analytical,'--output-dir',work/'13_guzzardi_lalli_network_figures'])
    run([PYTHON,ROOT/'scripts/14_guzzardi_lalli_community_vocabulary.py',analytical,'--output-dir',work/'14_guzzardi_lalli_community_vocabulary','--config-dir',ROOT/'config'])
    if args.physics_baseline and args.engineering_baseline:
        run([PYTHON,ROOT/'scripts/11_publication_trend_comparison.py',
             '--fusion-corpus',phase3,
             '--baselines',f'physics={args.physics_baseline}',f'engineering={args.engineering_baseline}',
             '--smooth-window','2','--min-consecutive-years','3','--min-avg-slope','0.04','--exclude-years','1962',
             '--output-dir',work/'11_publication_trends'])
    else:
        print("Skipping relative publication trajectories: both external comparison series must be supplied.")

    print(f"\nPipeline completed. Outputs: {work.resolve()}")

if __name__=='__main__':
    main()
