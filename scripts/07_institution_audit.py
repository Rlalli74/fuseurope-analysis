#!/usr/bin/env python3
"""Conservative institution analysis over user-selected periods/ranges."""
from pathlib import Path
import argparse,json,sys
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(ROOT/'src'))
from fusion_pipeline.institutions import *
from fusion_pipeline.exploration import parse_period_specs,BASIC_GRAPH_METRICS,EXTENDED_GRAPH_METRICS,compute_centralities
from fusion_pipeline.networks import graph_to_graphml_safe

def csv_list(s):return [x.strip() for x in s.split(',') if x.strip()]
def main():
 ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter);ap.add_argument('phase3_csv');ap.add_argument('--output-dir',default='outputs_phase7');ap.add_argument('--aliases',default=str(ROOT/'config/institution_aliases.csv'))
 ap.add_argument('--periods',nargs='*',default=['1959:1968','1969:1973','1974:1978','1983','1986']);ap.add_argument('--start-year',type=int,default=1956);ap.add_argument('--end-year',type=int,default=2007);ap.add_argument('--window-mode',choices=['annual','rolling','period'],default='annual');ap.add_argument('--window-size',type=int,default=1);ap.add_argument('--step',type=int,default=1)
 ap.add_argument('--analyses',default='inventory,period_graphs',help='inventory,period_graphs,metrics,centralities');ap.add_argument('--metrics',default='extended');ap.add_argument('--centralities',default='degree,degree_normalized,weighted_degree,fractional_degree,betweenness,closeness,eigenvector');ap.add_argument('--scope',choices=['full','lcc'],default='lcc');ap.add_argument('--focal-institutions',default='');ap.add_argument('--top-n',type=int,default=20);ap.add_argument('--modularity-method',choices=['louvain','greedy'],default='louvain');ap.add_argument('--betweenness-exact-max-nodes',type=int,default=1000);ap.add_argument('--betweenness-samples',type=int,default=250)
 args=ap.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);analyses=set(csv_list(args.analyses));df=pd.read_csv(args.phase3_csv,low_memory=False);aliases=load_aliases(args.aliases);aff=affiliation_table_from_dataframe(df,aliases);aff.to_csv(out/'affiliation_institution_table.csv',index=False)
 if 'inventory' in analyses:
  inv=institution_inventory(aff,args.start_year,args.end_year);inv.to_csv(out/'institution_inventory.csv',index=False);alias_candidates(inv,2,.88).to_csv(out/'institution_alias_review_candidates.csv',index=False)
 if 'period_graphs' in analyses:
  rows=[]
  for label,s,e in parse_period_specs(args.periods):
   G=build_coinstitution_graph(aff,s,e);m=graph_summary(G);m.update(period=label,start_year=s,end_year=e);rows.append(m);graph_to_graphml_safe(G,str(out/f'coinstitution_{label}_surface_normalized.graphml'))
  pd.DataFrame(rows).to_csv(out/'coinstitution_period_summary.csv',index=False)
 if 'metrics' in analyses:
  metrics=BASIC_GRAPH_METRICS if args.metrics=='basic' else EXTENDED_GRAPH_METRICS if args.metrics=='extended' else csv_list(args.metrics);coinstitution_longitudinal_metrics(aff,args.start_year,args.end_year,args.window_mode,args.window_size,args.step,metrics,args.modularity_method).to_csv(out/'coinstitution_longitudinal_metrics.csv',index=False)
 if 'centralities' in analyses:
  institution_centrality_over_time(aff,args.start_year,args.end_year,csv_list(args.centralities),csv_list(args.focal_institutions) or None,args.top_n if not args.focal_institutions else None,args.window_mode,args.window_size,args.step,args.scope).to_csv(out/'institution_centrality_over_time.csv',index=False)
 summary={'status':'preliminary conservative institution layer','analysis_range':[args.start_year,args.end_year],'periods':args.periods,'analyses':sorted(analyses),'window_mode':args.window_mode,'window_size':args.window_size,'step':args.step,'automatic_merging':'only exact surface-normalized identity plus explicit aliases in config/institution_aliases.csv','manual_alias_review_required':True};(out/'phase7_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
