#!/usr/bin/env python3
"""Generate the Guzzardi-Lalli community x vocabulary Figure 5."""
from pathlib import Path
import argparse, sys
import pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'src'))
from fusion_pipeline.networks import prepare_network_dataframe
from fusion_pipeline.socio_epistemic import load_titleword_config, load_titleword_stopword_profiles, community_titleword_profiles, community_terms_table, community_tfidf_table
from fusion_pipeline.community_vocabulary import community_vocabulary_figure_data, plot_community_vocabulary_panels


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase3_csv')
    ap.add_argument('--output-dir',default='outputs_guzzardi_community_vocabulary')
    ap.add_argument('--config-dir',default=str(ROOT/'config'))
    ap.add_argument('--years',nargs='*',type=int,default=[1979,1986,1996,2001])
    ap.add_argument('--top-communities',type=int,default=3)
    ap.add_argument('--terms-per-community',type=int,default=5)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--stopword-profile',choices=['main','english_only','expanded_domain'],default='main')
    ap.add_argument('--preserve-ordinals',action='store_true',help='Robustness mode: retain ordinal tokens such as 1st/2nd')
    args=ap.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    df=prepare_network_dataframe(pd.read_csv(args.phase3_csv,low_memory=False))
    exclusions,merge_map=load_titleword_config(args.config_dir)
    stop_profiles=load_titleword_stopword_profiles(args.config_dir)
    exclusions=stop_profiles[args.stopword_profile]
    panels={}; all_rows=[]
    for y in args.years:
        r=community_titleword_profiles(df,y,exclusions,merge_map,seed=args.seed,remove_ordinals=not args.preserve_ordinals)
        # Preserve complete auditable tables, not only what is displayed.
        community_terms_table(r,25).to_csv(out/f'{y}_community_titlewords_top25.csv',index=False)
        community_tfidf_table(r,25).to_csv(out/f'{y}_community_tfidf_top25.csv',index=False)
        d=community_vocabulary_figure_data(r,y,args.top_communities,args.terms_per_community)
        d.to_csv(out/f'{y}_figure5_community_vocabulary_data.csv',index=False)
        panels[y]=d;all_rows.append(d)
    combined=pd.concat(all_rows,ignore_index=True)
    combined.to_csv(out/'Figure_5_community_vocabulary_data.csv',index=False)
    plot_community_vocabulary_panels(panels,out/'Figure_5_community_vocabulary.png',args.years,dpi=300)
    plot_community_vocabulary_panels(panels,out/'Figure_5_community_vocabulary.tiff',args.years,dpi=600)
    print(f'Wrote {out}')

if __name__=='__main__': main()
