#!/usr/bin/env python3
"""Annual/rolling co-authorship diagnostics with user-selectable range and metrics."""
from pathlib import Path
import argparse,json,sys
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(ROOT/'src'))
from fusion_pipeline.networks import prepare_network_dataframe,yearly_team_size_diagnostics,yearly_document_type_diagnostics,large_team_papers,author_overlap_1983_1986
from fusion_pipeline.exploration import longitudinal_author_metrics,BASIC_GRAPH_METRICS,EXTENDED_GRAPH_METRICS

def csv_list(s): return [x.strip() for x in s.split(',') if x.strip()]

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('phase3_csv');ap.add_argument('--output-dir',default='outputs_phase4a')
 ap.add_argument('--start-year',type=int,default=1960);ap.add_argument('--end-year',type=int,default=2007)
 ap.add_argument('--window-mode',choices=['annual','rolling','period'],default='annual');ap.add_argument('--window-size',type=int,default=1);ap.add_argument('--step',type=int,default=1)
 ap.add_argument('--analyses',default='metrics,team_diagnostics,document_type_diagnostics,large_team,overlap',help='metrics,team_diagnostics,document_type_diagnostics,large_team,overlap')
 ap.add_argument('--metrics',default='basic',help="'basic', 'extended', or comma-separated metric names")
 ap.add_argument('--document-type-mode',default='all_selected');ap.add_argument('--max-team-size',type=int,default=None)
 ap.add_argument('--modularity-method',choices=['louvain','greedy'],default='louvain')
 ap.add_argument('--large-team-threshold',type=int,default=20)
 args=ap.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);analyses=set(csv_list(args.analyses))
 df=prepare_network_dataframe(pd.read_csv(args.phase3_csv,low_memory=False))
 metrics=BASIC_GRAPH_METRICS if args.metrics=='basic' else EXTENDED_GRAPH_METRICS if args.metrics=='extended' else csv_list(args.metrics)
 selected=df[(df.year>=args.start_year)&(df.year<=args.end_year)].copy()
 if 'team_diagnostics' in analyses: yearly_team_size_diagnostics(selected).to_csv(out/'yearly_team_size_diagnostics.csv',index=False)
 if 'document_type_diagnostics' in analyses: yearly_document_type_diagnostics(selected).to_csv(out/'yearly_document_type_diagnostics.csv',index=False)
 if 'large_team' in analyses: large_team_papers(selected,args.large_team_threshold).to_csv(out/f'large_team_papers_{args.large_team_threshold}plus.csv',index=False)
 if 'metrics' in analyses:
  result=longitudinal_author_metrics(df,args.start_year,args.end_year,args.window_mode,args.window_size,args.step,metrics,args.document_type_mode,args.max_team_size,args.modularity_method)
  result.to_csv(out/'coauthorship_longitudinal_metrics.csv',index=False)
 if 'overlap' in analyses and args.start_year<=1983 and args.end_year>=1986: author_overlap_1983_1986(df).to_csv(out/'author_overlap_1983_to_1986_lcc.csv',index=False)
 summary={'analyses':sorted(analyses),'analysis_range':[args.start_year,args.end_year],'window_mode':args.window_mode,'window_size':args.window_size,'step':args.step,'metrics':list(metrics),'document_type_mode':args.document_type_mode,'max_team_size':args.max_team_size,'modularity_method':args.modularity_method}
 (out/'phase4a_analysis_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
