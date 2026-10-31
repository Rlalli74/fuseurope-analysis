#!/usr/bin/env python3
"""Team-size/document-type robustness over user-selected periods."""
from pathlib import Path
import argparse,json,sys
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(ROOT/'src'))
from fusion_pipeline.networks import prepare_network_dataframe,snapshot_metrics
from fusion_pipeline.exploration import parse_period_specs
DEFAULT_PERIODS=['1979','1983','1986','1996','2001']
DOC_MODES=['unrestricted_analytical_corpus','exclude_meeting_abstracts','exclude_proceedings_and_meeting_abstracts','articles_only']

def ints_or_none(vals):return [None if str(v).lower() in {'none','all'} else int(v) for v in vals]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase3_csv');ap.add_argument('--output-dir',default='outputs_phase4c');ap.add_argument('--periods',nargs='*',default=DEFAULT_PERIODS)
 ap.add_argument('--analyses',default='team_size,document_type',help='team_size,document_type');ap.add_argument('--team-thresholds',nargs='*',default=['none','100','50','20','10']);ap.add_argument('--document-type-modes',nargs='*',default=DOC_MODES)
 args=ap.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);analyses={x.strip() for x in args.analyses.split(',') if x.strip()};df=prepare_network_dataframe(pd.read_csv(args.phase3_csv,low_memory=False));periods=parse_period_specs(args.periods);thresholds=ints_or_none(args.team_thresholds)
 if 'team_size' in analyses: snapshot_metrics(df,periods,thresholds,('unrestricted_analytical_corpus',)).to_csv(out/'snapshot_team_size_sensitivity.csv',index=False)
 if 'document_type' in analyses: snapshot_metrics(df,periods,(None,),tuple(args.document_type_modes)).to_csv(out/'snapshot_document_type_sensitivity.csv',index=False)
 summary={'analyses':sorted(analyses),'periods':args.periods,'team_size_thresholds':args.team_thresholds,'document_type_modes':args.document_type_modes};(out/'phase4c_analysis_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
