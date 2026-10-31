"""Conservative institution-level reconstruction utilities.

This phase deliberately separates surface cleaning from institution identity
merging. Punctuation/case and C1 author-prefix removal are automatic. Possible
aliases are exported for review; they are not silently merged unless listed in a
manual alias table.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from itertools import combinations
from difflib import SequenceMatcher
import ast, json, re

import pandas as pd
import networkx as nx

from .networks import parse_list_cell
from .normalization import strip_address_prefix

BRACKET_PREFIX=re.compile(r'^\s*\[([^\]]+)\]\s*')


def c1_author_prefix(affiliation:str):
    m=BRACKET_PREFIX.match('' if affiliation is None else str(affiliation))
    if not m:return []
    return [x.strip() for x in m.group(1).split(';') if x.strip()]


def strip_c1_author_prefix(affiliation:str)->str:
    return BRACKET_PREFIX.sub('', '' if affiliation is None else str(affiliation).strip())


def institution_surface(affiliation:str)->str:
    s=strip_address_prefix(affiliation)
    return s.split(',',1)[0].strip() if s else ''


def institution_key(label:str)->str:
    s=str(label or '').upper().replace('&',' AND ')
    s=re.sub(r'[^A-Z0-9]+',' ',s)
    return re.sub(r'\s+',' ',s).strip()


def load_aliases(path=None):
    if not path:return {}
    d=pd.read_csv(path)
    return {institution_key(r.variant):str(r.canonical).strip() for _,r in d.iterrows() if str(r.variant).strip() and str(r.canonical).strip()}


def canonical_institution(label, aliases=None):
    k=institution_key(label); aliases=aliases or {}
    return aliases.get(k,k)


def affiliation_table_from_dataframe(df, alias_map=None):
    """Reconstruct deduplicated institutions from C1 union RP."""
    rows=[]; alias_map=alias_map or {}
    for _,r in df.iterrows():
        affs=[('C1',a) for a in parse_list_cell(r.get('affiliations',[]))]
        rp=r.get('reprint_address_one_line',r.get('reprint_address',''))
        if pd.notna(rp) and str(rp).strip(): affs.append(('RP',str(rp)))
        seen=set()
        for source,a in affs:
            cleaned=strip_address_prefix(a); dedup=institution_key(cleaned)
            if not dedup or dedup in seen: continue
            seen.add(dedup)
            raw=institution_surface(a); key=institution_key(raw); canon=alias_map.get(key,key)
            rows.append({'record_id':r.get('record_id',''),'year':int(r.year),'affiliation_no':len(seen),
                         'address_source':source,
                         'affiliation_raw':a,'affiliation_authors_raw':'|'.join(c1_author_prefix(a)),
                         'institution_surface':raw,'institution_key':key,'institution_canonical':canon})
    return pd.DataFrame(rows)


def institution_inventory(aff_df,start=1956,end=2007):
    d=aff_df[(aff_df.year>=start)&(aff_df.year<=end)&(aff_df.institution_canonical!='')]
    rows=[]
    for inst,g in d.groupby('institution_canonical'):
        rows.append({'institution':inst,'affiliations':len(g),'papers':g.record_id.nunique(),
                     'first_year':int(g.year.min()),'last_year':int(g.year.max()),'years_present':g.year.nunique(),
                     'surface_forms':' | '.join(sorted(set(g.institution_surface))[:20])})
    return pd.DataFrame(rows).sort_values(['papers','institution'],ascending=[False,True]).reset_index(drop=True)


def alias_candidates(inventory,min_papers=2,threshold=.88,max_candidates=5000):
    d=inventory[inventory.papers>=min_papers].copy(); names=d.institution.tolist();freq=dict(zip(d.institution,d.papers))
    buckets=defaultdict(list)
    for n in names:
        toks=n.split(); key=(toks[0][:4] if toks else '', len(toks))
        buckets[key].append(n)
    rows=[]
    for vals in buckets.values():
        for a,b in combinations(vals,2):
            ratio=SequenceMatcher(None,a,b).ratio()
            if ratio>=threshold:
                rows.append({'institution_a':a,'institution_b':b,'similarity':ratio,
                             'papers_a':freq[a],'papers_b':freq[b],'decision':'REVIEW'})
    rows.sort(key=lambda x:(-x['similarity'],-max(x['papers_a'],x['papers_b'])))
    return pd.DataFrame(rows[:max_candidates])


def paper_institution_lists(aff_df,start,end):
    d=aff_df[(aff_df.year>=start)&(aff_df.year<=end)&(aff_df.institution_canonical!='')]
    return d.groupby('record_id').institution_canonical.apply(lambda s:list(dict.fromkeys(s))).to_dict()


def build_coinstitution_graph(aff_df,start,end):
    G=nx.Graph()
    for rid,insts in paper_institution_lists(aff_df,start,end).items():
        n=len(insts)
        for a in insts:G.add_node(a)
        if n<2:continue
        frac=1/(n-1)
        for a,b in combinations(insts,2):
            if not G.has_edge(a,b):G.add_edge(a,b,paper_count=0,weight=0,fractional_weight=0.0)
            G[a][b]['paper_count']+=1;G[a][b]['weight']+=1;G[a][b]['fractional_weight']+=frac
    return G


def graph_summary(G):
    n=G.number_of_nodes();m=G.number_of_edges()
    if not n:return {'nodes':0,'edges':0,'components':0,'lcc_nodes':0,'lcc_share_nodes':0.0,'density':0.0}
    l=max(nx.connected_components(G),key=len);return {'nodes':n,'edges':m,'components':nx.number_connected_components(G),
        'lcc_nodes':len(l),'lcc_share_nodes':len(l)/n,'density':nx.density(G)}


def coinstitution_longitudinal_metrics(
    aff_df,
    start_year,
    end_year,
    mode='annual',
    window_size=1,
    step=1,
    metrics=('number_of_nodes','number_of_edges','average_degree','density','largest_component_size','ratio_largest_component_nodes'),
    modularity_method='louvain',
    seed=42,
):
    """User-selectable longitudinal metrics for co-institution networks."""
    from .exploration import make_time_windows, compute_graph_metrics
    rows=[]
    for w in make_time_windows(start_year,end_year,mode,window_size,step):
        G=build_coinstitution_graph(aff_df,w.start_year,w.end_year)
        row=compute_graph_metrics(G,metrics,modularity_method=modularity_method,seed=seed)
        row.update({'label':w.label,'start_year':w.start_year,'end_year':w.end_year,'window_mode':mode,'window_size':window_size})
        rows.append(row)
    return pd.DataFrame(rows)


def institution_centrality_over_time(
    aff_df,
    start_year,
    end_year,
    measures=('degree','betweenness'),
    institutions=None,
    top_n=None,
    mode='annual',
    window_size=1,
    step=1,
    scope='lcc',
    seed=42,
    betweenness_exact_max_nodes=1000,
    betweenness_samples=250,
):
    """Centrality trajectories/top institutions for arbitrary user-defined windows."""
    from .exploration import make_time_windows, compute_centralities
    rows=[]; focal=list(institutions or [])
    for w in make_time_windows(start_year,end_year,mode,window_size,step):
        G=build_coinstitution_graph(aff_df,w.start_year,w.end_year)
        cent=compute_centralities(G,measures,scope=scope,seed=seed,betweenness_exact_max_nodes=betweenness_exact_max_nodes,betweenness_samples=betweenness_samples)
        if focal:
            ci=cent.set_index('node') if not cent.empty else pd.DataFrame()
            for inst in focal:
                row={'institution':inst,'label':w.label,'start_year':w.start_year,'end_year':w.end_year,'centrality_scope':scope}
                for measure in measures:
                    row[measure]=float(ci.loc[inst,measure]) if (not ci.empty and inst in ci.index) else 0.0
                rows.append(row)
        elif top_n:
            for measure in measures:
                for rank,(_,r) in enumerate(cent.nlargest(int(top_n),measure).iterrows(),1):
                    rows.append({'institution':r.node,'measure':measure,'value':float(r[measure]),'rank':rank,'label':w.label,
                                 'start_year':w.start_year,'end_year':w.end_year,'centrality_scope':scope})
        else:
            temp=cent.rename(columns={'node':'institution'}).copy();temp['label']=w.label;temp['start_year']=w.start_year;temp['end_year']=w.end_year
            rows.extend(temp.to_dict('records'))
    return pd.DataFrame(rows)
