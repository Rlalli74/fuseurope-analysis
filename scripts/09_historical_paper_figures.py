#!/usr/bin/env python3
"""Generate user-selected data/plots corresponding to quantitative historical-paper figures."""
from pathlib import Path
import argparse, json, sys
import pandas as pd
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; sys.path.insert(0,str(ROOT/'src'))
from fusion_pipeline.historical_figures import *


def csv_list(s): return [x.strip() for x in s.split(',') if x.strip()]

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('phase3_csv'); ap.add_argument('--output-dir',default='outputs_historical_figures')
    ap.add_argument('--figures',default='lcc,icr,country_output,intra_extra',help='lcc,icr,country_output,intra_extra,europe_robustness')
    ap.add_argument('--start-year',type=int,default=1973); ap.add_argument('--end-year',type=int,default=2007)
    ap.add_argument('--plot-start-year',type=int,default=None); ap.add_argument('--plot-end-year',type=int,default=None)
    ap.add_argument('--document-type-mode',default='unrestricted_analytical_corpus'); ap.add_argument('--max-team-size',type=int,default=None)
    ap.add_argument('--icr-smoothing-window',type=int,default=3)
    ap.add_argument('--country-series',default='USA,JAPAN,USSR_RUSSIA,CHINA,EURATOM')
    ap.add_argument('--focal-countries',default='GERMANY,FRANCE,ITALY,UK')
    ap.add_argument('--europe-definition',choices=['dynamic','fixed'],default='dynamic')
    ap.add_argument('--fixed-europe-set',choices=['EC6_1958','EC10_1981','EC12_1986','EU15_1995','EU25_2004'],default='EC12_1986')
    ap.add_argument('--exclude-jet-partner',action='store_true')
    args=ap.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    requested=set(csv_list(args.figures)); df=pd.read_csv(args.phase3_csv,low_memory=False)
    summary={'figures':sorted(requested),'analysis_range':[args.start_year,args.end_year],
             'plot_range':[args.plot_start_year,args.plot_end_year],
             'document_type_mode':args.document_type_mode,'max_team_size':args.max_team_size}
    if 'lcc' in requested:
        x=author_lcc_series(df,args.start_year,args.end_year,args.document_type_mode,args.max_team_size)
        x.to_csv(out/'historical_lcc_series.csv',index=False)
        plot_author_lcc_series(x,out/'historical_lcc_series.png',args.plot_start_year,args.plot_end_year,include_density=False)
        plot_author_lcc_series(x,out/'historical_lcc_and_density_diagnostic.png',args.plot_start_year,args.plot_end_year,include_density=True)
    if 'icr' in requested:
        annual=icr_series_for_manuscript(df,args.start_year,args.end_year,args.icr_smoothing_window)
        annual.to_csv(out/'historical_ICR_series.csv',index=False)
        plot_icr_series(annual,out/'historical_ICR_plot.png',args.plot_start_year,args.plot_end_year)
        summary['icr_smoothing_window']=args.icr_smoothing_window
        summary['icr_manuscript_specification']='observed annual ICR + Wilson interval + centered three-year pooled ratio; no shortened endpoints or change-point model selection'
    if 'country_output' in requested:
        x=aggregate_fractional_country_output(df,args.start_year,args.end_year,csv_list(args.country_series),args.europe_definition,args.fixed_europe_set)
        x.to_csv(out/'historical_country_fractional_output.csv',index=False)
        plot_fractional_country_output(x,out/'historical_country_fractional_output.png',args.plot_start_year,args.plot_end_year)
    if 'intra_extra' in requested:
        x=focal_intra_extra_series(df,csv_list(args.focal_countries),args.start_year,args.end_year,args.europe_definition,args.fixed_europe_set,not args.exclude_jet_partner)
        x.to_csv(out/'historical_focal_intra_extra.csv',index=False)
        plot_focal_intra_extra(x,out/'historical_focal_intra_extra.png',args.plot_start_year,args.plot_end_year)
    if 'europe_robustness' in requested:
        output,focal=europe_composition_robustness(df,csv_list(args.focal_countries),args.start_year,args.end_year,args.fixed_europe_set)
        output.to_csv(out/'historical_country_output_dynamic_vs_fixed.csv',index=False)
        focal.to_csv(out/'historical_focal_partner_dynamic_fixed_jet_sensitivity.csv',index=False)
        summary['europe_robustness']={'fixed_set':args.fixed_europe_set,'jet_variants':[True,False]}
    (out/'historical_figure_generation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
