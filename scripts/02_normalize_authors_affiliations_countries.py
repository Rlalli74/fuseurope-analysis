#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'src'))

from fusion_pipeline.wos_parser import parse_wos_folder, records_to_dataframe
from fusion_pipeline.normalization import add_author_normalization, build_affiliation_table, add_country_lists_to_records
from fusion_pipeline.phase2_validation import (
    validate_author_count_invariance, summarize_truncation_audit,
    summarize_affiliations, compare_country_lists_to_historical,
)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('wos_folder', help='Folder containing original WoS .txt exports')
    ap.add_argument('--historical-csv', default=None)
    ap.add_argument('--output-dir', default='outputs_phase2')
    args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)

    records=parse_wos_folder(args.wos_folder)
    records,trunc_audit=add_author_normalization(records)
    aff=build_affiliation_table(records)
    records=add_country_lists_to_records(records,aff)

    records_to_dataframe(records,serialise_lists=True).to_csv(out/'preserved_corpus_phase2.csv',index=False)
    aff.to_csv(out/'affiliations_phase2.csv',index=False)
    trunc_audit.to_csv(out/'author_dot_truncation_audit.csv',index=False)

    summary={
      'author_count_invariance':validate_author_count_invariance(records),
      'truncated_author_recovery':summarize_truncation_audit(trunc_audit),
      'affiliations':summarize_affiliations(aff),
    }
    if args.historical_csv:
        cs,detail=compare_country_lists_to_historical(records,args.historical_csv)
        summary['historical_country_comparison']=cs
        detail.to_csv(out/'country_comparison_to_historical.csv',index=False)
    with open(out/'phase2_validation_summary.json','w',encoding='utf-8') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
