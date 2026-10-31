#!/usr/bin/env python3
"""Final historical title contrasts, with an explicit generic-explorer mode."""
from pathlib import Path
import argparse,json,sys
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(ROOT/'src'))
from fusion_pipeline.period_contrast import *

def csv_list(s):return [x.strip() for x in s.split(',') if x.strip()]

def main():
 ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter);ap.add_argument('phase3_csv');ap.add_argument('--output-dir',default='outputs_period_contrast')
 ap.add_argument('--generic-explorer',action='store_true',help='Run the non-paper generic explorer instead of the two frozen final contrasts')
 ap.add_argument('--focal-start',type=int);ap.add_argument('--focal-end',type=int);ap.add_argument('--baseline-start',type=int,default=None);ap.add_argument('--baseline-end',type=int)
 ap.add_argument('--analyses',default='availability,distinctiveness,trends',help='availability,distinctiveness,trends,institution_term_cooccurrence')
 ap.add_argument('--features',default='title,authors,institutions,countries',help='title,abstract,keywords,title_abstract_keywords,authors,institutions,countries')
 ap.add_argument('--ngrams',default='1,2');ap.add_argument('--min-doc-frequency',type=int,default=3);ap.add_argument('--top-n-trends',type=int,default=40);ap.add_argument('--top-n-cooccurrence',type=int,default=20)
 args=ap.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
 raw=pd.read_csv(args.phase3_csv,low_memory=False)
 if not args.generic_explorer:
  tables=run_final_historical_title_contrasts(raw)
  for label,table in tables.items(): table.to_csv(out/f'historical_title_contrast_{label}.csv',index=False)
  summary={'mode':'final_historical_title_contrasts','corpus':'analytical','contrasts':[x[0] for x in FINAL_HISTORICAL_CONTRASTS],
           'features':'lowercase unigram+bigram document presence; punctuation removed; internal hyphens retained',
           'min_document_frequency':3,'haldane_anscombe':0.5,'effect_size':'log_odds','ranking':'z_score'}
  (out/'period_contrast_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2));return
 required=(args.focal_start,args.focal_end,args.baseline_end)
 if any(x is None for x in required): ap.error('--generic-explorer requires --focal-start, --focal-end, and --baseline-end')
 analyses=set(csv_list(args.analyses));features=csv_list(args.features);ng=tuple(int(x) for x in csv_list(args.ngrams))
 df=prepare_period_contrast(raw,args.focal_start,args.focal_end,args.baseline_start,args.baseline_end)
 if 'availability' in analyses:
  cols=['title','abstract','keywords','authors_network','institutions_normalized','countries_normalized'];availability_by_period(df,cols).to_csv(out/'period_field_availability.csv',index=False)
 prepared={};results={}
 for feat in features:
  work,col=add_feature_column(df,feat,ng);prepared[feat]=(work,col)
  if 'distinctiveness' in analyses or 'trends' in analyses or 'institution_term_cooccurrence' in analyses:
   dist=distinctive_features(work,col,args.min_doc_frequency);results[feat]=dist;dist.to_csv(out/f'{feat}_distinctive_features.csv',index=False)
  if 'trends' in analyses and not results[feat].empty:
   top=results[feat].head(args.top_n_trends).feature.tolist();yearly_presence_trends(work,col,top,args.focal_start,args.focal_end).to_csv(out/f'{feat}_focal_period_trends.csv',index=False)
 if 'institution_term_cooccurrence' in analyses and 'institutions' in prepared:
  textfeat='title_abstract_keywords' if 'title_abstract_keywords' in prepared else 'title' if 'title' in prepared else None
  if textfeat:
   iw,icol=prepared['institutions'];tw,tcol=prepared[textfeat]
   # feature columns were made on equivalent period rows; put both into one frame by index.
   combo=iw.copy();combo[tcol]=tw[tcol]
   insts=results['institutions'].head(args.top_n_cooccurrence).feature.tolist();terms=results[textfeat].head(args.top_n_cooccurrence).feature.tolist()
   institution_term_cooccurrence(combo,icol,tcol,insts,terms,args.focal_start,args.focal_end).to_csv(out/'institution_term_cooccurrence.csv',index=False)
 summary={'focal_period':[args.focal_start,args.focal_end],'baseline_period':[args.baseline_start,args.baseline_end],'analyses':sorted(analyses),'features':features,'ngrams':ng,'min_doc_frequency':args.min_doc_frequency}
 (out/'period_contrast_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
