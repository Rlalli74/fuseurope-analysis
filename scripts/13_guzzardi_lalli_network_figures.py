#!/usr/bin/env python3
"""Generate the canonical Guzzardi-Lalli network overview and repository views.

v0.13 keeps the v0.12 single-network outputs for reproducibility but adds the
paper-facing Figure 4: a 2x2 overview of the complete annual co-authorship
networks in 1979, 1986, 1996, and 2001.  The multipanel highlights only LCC
membership versus all other components; Louvain/betweenness encodings remain
in the repository-level LCC figures rather than the main overview.
"""
from pathlib import Path
import argparse,json,sys
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(ROOT/'src'))
from fusion_pipeline.networks import prepare_network_dataframe,filter_publications,build_coauthorship_graph
from fusion_pipeline.guzzardi_figures import plot_full_network,plot_full_network_multipanel,plot_lcc_network,network_overview_stats


def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('phase3_csv')
 ap.add_argument('--output-dir',default='outputs_guzzardi_figure4_v013')
 ap.add_argument('--years',nargs='*',type=int,default=[1979,1986,1996,2001])
 ap.add_argument('--full-years',nargs='*',type=int,default=[1979,1986,1996,2001])
 ap.add_argument('--lcc-years',nargs='*',type=int,default=[1979,1986,1996,2001])
 ap.add_argument('--betweenness-samples',type=int,default=500)
 ap.add_argument('--seed',type=int,default=42)
 ap.add_argument('--label-top-n',type=int,default=10)
 ap.add_argument('--dpi',type=int,default=360)
 ap.add_argument('--tiff-dpi',type=int,default=600)
 args=ap.parse_args()
 if len(args.years) != 4:
  raise SystemExit('--years must contain exactly the four paper snapshot years.')
 out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
 df=prepare_network_dataframe(pd.read_csv(args.phase3_csv,low_memory=False));audit=[];graphs={}

 for y in args.years:
  sub=filter_publications(df,y,y,None,'unrestricted_analytical_corpus')
  G=build_coauthorship_graph(sub);graphs[y]=G
  stats=network_overview_stats(G)
  row={'year':y,'papers':len(sub),**stats}
  if y in args.full_years:
   a=plot_full_network(
       G,out/f'{y}_full_repository.png',seed=args.seed,
       samples=args.betweenness_samples,labels=args.label_top_n if y==1979 else 0,dpi=args.dpi,
   )
   row['full_display_betweenness']=a['betweenness_method']
  if y in args.lcc_years:
   a=plot_lcc_network(
       G,out/f'{y}_lcc_repository.png',seed=args.seed,
       samples=args.betweenness_samples,labels=args.label_top_n,dpi=args.dpi,
   )
   row.update(louvain_communities=a['communities'],lcc_display_betweenness=a['betweenness_method'])
   a['node_table'].to_csv(out/f'{y}_lcc_nodes_repository.csv',index=False)
  audit.append(row);print(y,row)

 multi=plot_full_network_multipanel(
     graphs,
     out/'Figure_4_network_evolution_1979_2001.png',
     years=args.years,seed=args.seed,dpi=args.dpi,
     tiff_output=out/'Figure_4_network_evolution_1979_2001.tiff',tiff_dpi=args.tiff_dpi,
 )
 multi.to_csv(out/'Figure_4_network_evolution_1979_2001_audit.csv',index=False)
 pd.DataFrame(audit).to_csv(out/'figure_generation_audit.csv',index=False)
 (out/'phase13_summary.json').write_text(json.dumps({
     'pipeline_version':'0.18.4rc1',
     'main_paper_figure':'Figure_4_network_evolution_1979_2001',
     'years':args.years,
     'full_years_repository':args.full_years,
     'lcc_years_repository':args.lcc_years,
     'seed':args.seed,
     'betweenness_samples_repository_only':args.betweenness_samples,
     'main_figure_encoding':{
         'lcc':'blue',
         'other_components':'grey',
         'node_size':'display-only function of network order',
         'louvain':False,
         'betweenness':False,
         'labels':False,
     },
     'dpi_png':args.dpi,
     'dpi_tiff':args.tiff_dpi,
 },indent=2))


if __name__=='__main__':main()
