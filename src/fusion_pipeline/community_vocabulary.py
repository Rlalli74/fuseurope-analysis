"""Paper-facing community x vocabulary summaries for the Guzzardi-Lalli article."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .socio_epistemic import community_tfidf_table


def community_vocabulary_figure_data(result, year, top_communities=3, terms_per_community=5):
    """Select largest communities and union of their top c-TF-IDF terms.

    Scores remain the canonical c-TF-IDF values. Community selection uses author
    count, matching the manuscript's current community-level table.
    """
    summary=result['community_summary'].copy()
    if summary.empty:
        return pd.DataFrame(columns=['year','community','authors','edge_support_papers','term','tfidf','tf','idf','selected_term'])
    selected=(summary.sort_values(['authors','community'],ascending=[False,True])
              .head(int(top_communities))['community'].tolist())
    tfidf=community_tfidf_table(result, top_n=None)
    tfidf=tfidf[tfidf.community.isin(selected)].copy()
    top=tfidf.sort_values(['community','tfidf','term'],ascending=[True,False,True]).groupby('community').head(int(terms_per_community))
    terms=set(top.term)
    grid=tfidf[tfidf.term.isin(terms)].copy()
    meta=summary[['community','authors','edge_support_papers']]
    grid=grid.merge(meta,on='community',how='left')
    grid['year']=int(year)
    grid['selected_term']=grid.set_index(['community','term']).index.isin(set(zip(top.community,top.term)))
    return grid[['year','community','authors','edge_support_papers','term','tfidf','tf','idf','selected_term']]


def _panel_matrix(data):
    communities=(data[['community','authors']].drop_duplicates()
                 .sort_values(['authors','community'],ascending=[False,True]))
    comms=communities.community.tolist()
    # Terms ordered by strongest c-TF-IDF in any selected community, then alphabetically.
    order=(data.groupby('term',as_index=False).tfidf.max()
           .sort_values(['tfidf','term'],ascending=[False,True]).term.tolist())
    mat=(data.pivot_table(index='community',columns='term',values='tfidf',aggfunc='max')
         .reindex(index=comms,columns=order).fillna(0.0))
    labels=[f"C{c} (n={int(communities.set_index('community').loc[c,'authors'])})" for c in comms]
    return mat,labels


def plot_community_vocabulary_panels(data_by_year, output_path, years=(1979,1986,1996,2001), dpi=300):
    """Render a 2x2 c-TF-IDF heatmap, one panel per snapshot.

    Colour intensity is scaled independently within each snapshot because c-TF-IDF
    magnitudes depend on that year's partition and vocabulary. The figure is for
    within-snapshot community comparison, not cross-year score comparison.
    """
    years=[int(y) for y in years]
    fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    for ax,year in zip(axes.flat,years):
        d=data_by_year[int(year)]
        if d.empty:
            ax.text(.5,.5,'No data',ha='center',va='center'); ax.set_axis_off(); continue
        mat,labels=_panel_matrix(d)
        vals=mat.to_numpy(float)
        vmax=float(np.nanmax(vals)) if vals.size else 1.0
        norm=vals/vmax if vmax>0 else vals
        im=ax.imshow(norm,aspect='auto',vmin=0,vmax=1,cmap='Greys')
        ax.set_yticks(range(len(labels)),labels=labels)
        ax.set_xticks(range(len(mat.columns)),labels=mat.columns,rotation=55,ha='right',fontsize=8)
        ax.set_title(str(year))
        ax.set_xlabel('Distinctive titlewords (union of community top terms)')
        # Mark terms that belong to each community's own selected top list.
        selected={(r.community,r.term) for r in d.itertuples() if bool(r.selected_term)}
        for i,c in enumerate(mat.index):
            for j,t in enumerate(mat.columns):
                if (c,t) in selected:
                    ax.text(j,i,'•',ha='center',va='center',fontsize=11)
    fig.supylabel('Largest Louvain communities')
    fig.suptitle('Community-specific title vocabulary in selected co-authorship snapshots')
    out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=dpi,bbox_inches='tight')
    plt.close(fig)
    return out
