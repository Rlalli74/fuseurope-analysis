"""Reusable data/plot helpers for the historical-paper figures.

The functions in this module are deliberately parameter-driven.  The dates used
in the manuscript are presets in the CLI/notebook, not assumptions embedded in
these functions.  The same helpers can therefore be used for alternative ranges
and robustness subsets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .exploration import longitudinal_author_metrics
from .internationalization import (
    EURATOM_ENTRY,
    FIXED_EUROPE_SETS,
    prepare_country_dataframe,
    yearly_icr,
    icr_with_pooled_smoothing,
    country_year_fractional_stats,
    focal_country_intra_extra,
)
from .networks import prepare_network_dataframe


def author_lcc_series(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    document_type_mode: str = "unrestricted_analytical_corpus",
    max_team_size: int | None = None,
) -> pd.DataFrame:
    """Annual author count and LCC share for an arbitrary date range."""
    ndf = prepare_network_dataframe(df)
    return longitudinal_author_metrics(
        ndf,
        start_year,
        end_year,
        mode="annual",
        metrics=("number_of_nodes", "largest_component_size", "ratio_largest_component_nodes", "density"),
        document_type_mode=document_type_mode,
        max_team_size=max_team_size,
    )


def plot_author_lcc_series(
    data: pd.DataFrame,
    output_path: str | Path,
    plot_start_year: int | None = None,
    plot_end_year: int | None = None,
    include_density: bool = False,
):
    """Plot author count and LCC share; optionally make density a separate panel.

    The plotting range may be narrower than the analysis range.
    """
    d = data.copy()
    if plot_start_year is not None:
        d = d[d["start_year"] >= int(plot_start_year)]
    if plot_end_year is not None:
        d = d[d["end_year"] <= int(plot_end_year)]
    x = d["start_year"]
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    if include_density:
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(9, 5.5))
    line1 = ax.plot(x, d["number_of_nodes"], label="Authors")
    ax.set_ylabel("Authors")
    ax2 = ax.twinx()
    line2 = ax2.plot(x, d["ratio_largest_component_nodes"], label="LCC share")
    ax2.set_ylabel("Share of authors in LCC")
    ax2.set_ylim(0, 1)
    ax.set_xlabel("Year" if not include_density else "")
    handles = line1 + line2
    ax.legend(handles, [h.get_label() for h in handles], loc="best")
    if include_density:
        axes[1].plot(x, d["density"], label="Network density")
        axes[1].set_ylabel("Density")
        axes[1].set_xlabel("Year")
        axes[1].legend(loc="best")
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)


def icr_series_for_manuscript(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    smoothing_window: int = 3,
) -> pd.DataFrame:
    """Annual geographic ICR used in the manuscript.

    The evidentiary series consists of the observed annual proportion, its Wilson
    interval, country-metadata coverage, and a centered pooled ratio used only as
    a visual summary. Change-point fitting is not part of manuscript reproduction.
    """
    cdf = prepare_country_dataframe(df)
    annual, pooled = icr_with_pooled_smoothing(
        cdf, start=start_year, end=end_year, windows=(int(smoothing_window),)
    )
    smoothed = pooled[int(smoothing_window)][["year", "ICR_pooled"]].rename(
        columns={"ICR_pooled": "ICR_centered_pooled"}
    )
    annual = annual.merge(smoothed, on="year", how="left")
    return annual


def plot_icr_series(
    annual: pd.DataFrame,
    output_path: str | Path,
    plot_start_year: int | None = None,
    plot_end_year: int | None = None,
):
    """Plot manuscript ICR: annual values, Wilson intervals, and centered pooled ratio."""
    a = annual.copy()
    if plot_start_year is not None:
        a = a[a.year >= int(plot_start_year)]
    if plot_end_year is not None:
        a = a[a.year <= int(plot_end_year)]
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(a.year, a.ICR, marker="o", markersize=3, label="Observed geographic ICR")
    if {"ICR_low", "ICR_high"}.issubset(a.columns):
        ax.fill_between(a.year, a.ICR_low, a.ICR_high, alpha=0.15, label="95% Wilson interval")
    if "ICR_centered_pooled" in a.columns:
        ax.plot(a.year, a.ICR_centered_pooled, label="3-year centered pooled ICR")
    ax.set_xlabel("Year"); ax.set_ylabel("Internationally co-authored share")
    ax.set_ylim(bottom=0); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)


# Deprecated internal aliases retained for callers of pre-final helper names.
def aici_series_for_manuscript(*args, **kwargs):
    frame = icr_series_for_manuscript(*args, **kwargs)
    return frame.rename(columns={name: name.replace("ICR", "AICI") for name in frame.columns if "ICR" in name})


def plot_aici_series(annual, *args, **kwargs):
    renamed = annual.rename(columns={name: name.replace("AICI", "ICR") for name in annual.columns if "AICI" in name})
    return plot_icr_series(renamed, *args, **kwargs)


def aggregate_fractional_country_output(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    series: Sequence[str] = ("USA", "JAPAN", "USSR_RUSSIA", "CHINA", "EURATOM"),
    europe_definition: str = "dynamic",
    fixed_europe_set: str = "EC12_1986",
) -> pd.DataFrame:
    """Aggregate fractional country output into user-selected historical series.

    ``EURATOM`` uses contemporaneous membership. ``USSR_RUSSIA`` combines the
    two canonical country labels without rewriting either raw label.
    """
    cdf = prepare_country_dataframe(df)
    cy = country_year_fractional_stats(cdf, start_year, end_year)
    rows = []
    for year in range(int(start_year), int(end_year) + 1):
        gy = cy[cy.year == year]
        for label in series:
            key = label.upper()
            if key == "EURATOM":
                if europe_definition == "dynamic":
                    members = {c for c, entry in EURATOM_ENTRY.items() if year >= entry}
                elif europe_definition == "fixed":
                    if fixed_europe_set not in FIXED_EUROPE_SETS:
                        raise ValueError(f"Unknown fixed Europe set: {fixed_europe_set}")
                    members = FIXED_EUROPE_SETS[fixed_europe_set]
                else:
                    raise ValueError("europe_definition must be 'dynamic' or 'fixed'")
                value = float(gy.loc[gy.country.isin(members), "frac_total"].sum())
            elif key in {"USSR_RUSSIA", "USSR/RUSSIA", "USSR-RUSSIA"}:
                value = float(gy.loc[gy.country.isin(["USSR", "RUSSIA"]), "frac_total"].sum())
            else:
                value = float(gy.loc[gy.country.eq(key), "frac_total"].sum())
            rows.append({"year": year, "series": label, "fractional_papers": value,
                         "europe_definition": europe_definition if key=="EURATOM" else "not_applicable",
                         "fixed_europe_set": fixed_europe_set if key=="EURATOM" and europe_definition=="fixed" else ""})
    return pd.DataFrame(rows)


def plot_fractional_country_output(
    data: pd.DataFrame,
    output_path: str | Path,
    plot_start_year: int | None = None,
    plot_end_year: int | None = None,
):
    d = data.copy()
    if plot_start_year is not None: d = d[d.year >= int(plot_start_year)]
    if plot_end_year is not None: d = d[d.year <= int(plot_end_year)]
    out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,5.5))
    for label,g in d.groupby("series", sort=False):
        ax.plot(g.year,g.fractional_papers,label=label)
    ax.set_xlabel("Year"); ax.set_ylabel("Fractional publication count")
    ax.legend(loc="best"); fig.tight_layout(); fig.savefig(out,dpi=300,bbox_inches="tight"); plt.close(fig)


def focal_intra_extra_series(
    df: pd.DataFrame,
    countries: Sequence[str],
    start_year: int,
    end_year: int,
    europe_definition: str = "dynamic",
    fixed_europe_set: str = "EC12_1986",
    include_jet_partner: bool = True,
) -> pd.DataFrame:
    cdf=prepare_country_dataframe(df)
    parts=[]
    for country in countries:
        _, x=focal_country_intra_extra(
            cdf,country.upper(),start=start_year,end=end_year,
            europe_definition=europe_definition,fixed_europe_set=fixed_europe_set,
            include_jet_partner=include_jet_partner,
        )
        if not x.empty:
            x["country"]=country.upper();x["europe_definition"]=europe_definition
            x["fixed_europe_set"]=fixed_europe_set if europe_definition=="fixed" else ""
            x["jet_as_european_partner"]=bool(include_jet_partner);parts.append(x)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()


def europe_composition_robustness(
    df:pd.DataFrame,countries:Sequence[str],start_year:int,end_year:int,
    fixed_europe_set:str='EC12_1986',
) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Return Figure-9 and Figure-10 data under fixed-composition/JET variants."""
    output=[]
    for definition in ('dynamic','fixed'):
        x=aggregate_fractional_country_output(
            df,start_year,end_year,['USA','JAPAN','USSR_RUSSIA','CHINA','EURATOM'],
            europe_definition=definition,fixed_europe_set=fixed_europe_set,
        )
        output.append(x)
    focal=[]
    for definition,include_jet in [('dynamic',True),('dynamic',False),('fixed',True),('fixed',False)]:
        x=focal_intra_extra_series(
            df,countries,start_year,end_year,europe_definition=definition,
            fixed_europe_set=fixed_europe_set,include_jet_partner=include_jet,
        )
        focal.append(x)
    return pd.concat(output,ignore_index=True),pd.concat(focal,ignore_index=True)


def plot_focal_intra_extra(
    data: pd.DataFrame,
    output_path: str | Path,
    plot_start_year: int | None = None,
    plot_end_year: int | None = None,
):
    """One panel per focal country; intra/extra partner shares use the same data."""
    d=data.copy()
    if plot_start_year is not None:d=d[d.year>=int(plot_start_year)]
    if plot_end_year is not None:d=d[d.year<=int(plot_end_year)]
    countries=list(dict.fromkeys(d.country.tolist()))
    if not countries: raise ValueError("No country data to plot")
    out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(len(countries),1,figsize=(9,max(3.0,2.7*len(countries))),sharex=True)
    if len(countries)==1: axes=[axes]
    for ax,country in zip(axes,countries):
        g=d[d.country==country]
        ax.plot(g.year,g.papers_with_euratom_partner_share,label="With Euratom/EC partner")
        ax.plot(g.year,g.papers_with_extra_partner_share,label="With extra-Euratom partner")
        ax.set_ylabel(country); ax.set_ylim(0,1); ax.legend(loc="best")
    axes[-1].set_xlabel("Year")
    fig.tight_layout(); fig.savefig(out,dpi=300,bbox_inches="tight"); plt.close(fig)
