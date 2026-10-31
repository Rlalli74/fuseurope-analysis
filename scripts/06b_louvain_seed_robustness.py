#!/usr/bin/env python3
"""Louvain-seed sensitivity for community-balanced titleword profiles.

This is separated from the main phase-06 analysis because repeated Louvain
partitions are deliberately a slower robustness check.
"""
from pathlib import Path
import argparse, sys
import pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'src'))
from fusion_pipeline.networks import prepare_network_dataframe
from fusion_pipeline.networks import build_coauthorship_graph, filter_publications, largest_component_graph, louvain_partition_sensitivity
from fusion_pipeline.socio_epistemic import load_titleword_config,louvain_seed_sensitivity

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase3_csv')
    ap.add_argument('--output-dir',default='outputs_phase6b')
    ap.add_argument('--config-dir',default=str(ROOT/'config'))
    ap.add_argument('--seeds',nargs='*',type=int,default=[1,7,21,42,84])
    ap.add_argument('--years',nargs='*',type=int,default=[1979,1986,1996,2001])
    ap.add_argument('--historical-periods',nargs='*',default=['1959:1968','1969:1973','1974:1978','1983','1986'])
    ap.add_argument('--historical-seeds',nargs='*',type=int,default=list(range(20)))
    ap.add_argument('--skip-historical',action='store_true')
    args=ap.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    df=prepare_network_dataframe(pd.read_csv(args.phase3_csv,low_memory=False))
    exclusions,merge_map=load_titleword_config(args.config_dir)
    tables=[]
    for y in args.years:
        t=louvain_seed_sensitivity(df,y,exclusions,merge_map,seeds=tuple(args.seeds),reference_seed=42)
        t.to_csv(out/f'louvain_seed_{y}.csv',index=False);tables.append(t)
    pd.concat(tables,ignore_index=True).to_csv(out/'louvain_seed_titleword_sensitivity.csv',index=False)
    if not args.skip_historical:
        historical=[]
        for spec in args.historical_periods:
            if ':' in spec:
                start,end=(int(x) for x in spec.split(':',1))
            else:
                start=end=int(spec)
            graph=build_coauthorship_graph(filter_publications(df,start,end))
            lcc=largest_component_graph(graph)
            table=louvain_partition_sensitivity(
                lcc, reference_seed=42, seeds=tuple(args.historical_seeds),
                weight='paper_count', resolution=1.0,
            )
            table.insert(0,'period',f'{start}-{end}' if start != end else str(start))
            table.insert(1,'start_year',start);table.insert(2,'end_year',end)
            historical.append(table)
        pd.concat(historical,ignore_index=True).to_csv(out/'historical_lcc_louvain_seed_sensitivity.csv',index=False)

if __name__=='__main__':main()
