"""Publication-count trend comparison for the fusion corpus.

External disciplinary baselines (for example WoS Physics or Engineering annual
counts) are inputs, not part of the fusion corpus.  The functions in this module
keep analysis and plot ranges separate and can derive fusion annual counts from
an analysis-ready record-level corpus.
"""
from __future__ import annotations
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import pandas as pd


def standardize_year_count(df: pd.DataFrame, name: str = "series") -> pd.DataFrame:
    """Return a sorted two-column ``year,count`` table.

    Only the first two input columns are used deliberately.  This keeps the
    function compatible with historical WoS exports that also contain a third
    percentage column whose denominator is not needed for the comparison.
    """
    if len(df.columns) < 2:
        raise ValueError(f"{name} must contain at least year and count columns")
    out = df.iloc[:, :2].copy()
    out.columns = ["year", "count"]
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["count"] = pd.to_numeric(out["count"], errors="coerce")
    out = out.dropna(subset=["year", "count"]).copy()
    out["year"] = out["year"].astype(int)
    out = out.groupby("year", as_index=False)["count"].sum()
    return out.sort_values("year").reset_index(drop=True)


def annual_publication_counts(
    records: pd.DataFrame,
    year_col: str = "year",
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Derive annual publication counts from a record-level corpus."""
    if year_col not in records.columns:
        raise ValueError(f"Missing year column: {year_col}")
    years = pd.to_numeric(records[year_col], errors="coerce")
    tmp = pd.DataFrame({"year": years}).dropna()
    tmp["year"] = tmp["year"].astype(int)
    if start_year is not None:
        tmp = tmp[tmp.year >= int(start_year)]
    if end_year is not None:
        tmp = tmp[tmp.year <= int(end_year)]
    return tmp.groupby("year").size().rename("count").reset_index().sort_values("year").reset_index(drop=True)


def compare_publication_series(
    series: Mapping[str, pd.DataFrame],
    focal_name: str = "fusion",
    start_year: int | None = None,
    end_year: int | None = None,
    smooth_window: int = 2,
) -> pd.DataFrame:
    """Compare a focal publication series with arbitrary annual baselines.

    For each baseline the output contains the raw ratio, log-ratio, smoothed
    log-ratio, and year-to-year difference of the smoothed log-ratio.
    """
    if focal_name not in series:
        raise ValueError("focal_name must be present in series")
    if int(smooth_window) < 1:
        raise ValueError("smooth_window must be >= 1")
    merged = None
    for name, df in series.items():
        x = standardize_year_count(df, name).rename(columns={"count": f"{name}_count"})
        merged = x if merged is None else merged.merge(x, on="year", how="inner")
    if merged is None or merged.empty:
        raise ValueError("No overlapping years across publication series")
    if start_year is not None:
        merged = merged[merged.year >= int(start_year)]
    if end_year is not None:
        merged = merged[merged.year <= int(end_year)]
    merged = merged.sort_values("year").reset_index(drop=True)
    eps = 1e-12
    for name in series:
        if name == focal_name:
            continue
        r = f"{focal_name}_vs_{name}"
        merged[r] = merged[f"{focal_name}_count"] / merged[f"{name}_count"]
        merged[f"log_{r}"] = np.log(merged[r] + eps)
        merged[f"log_{r}_smooth"] = merged[f"log_{r}"].rolling(int(smooth_window), min_periods=1).mean()
        merged[f"diff_{r}"] = merged[f"log_{r}_smooth"].diff()
    return merged.reset_index(drop=True)


def detect_common_trend_periods(
    comparison: pd.DataFrame,
    ratio_names: Sequence[str],
    min_consecutive_years: int = 3,
    min_avg_slope: float = 0.0,
    exclude_years: Sequence[int] = (),
) -> pd.DataFrame:
    """Detect runs where all focal/baseline log-ratios move in one direction.

    ``exclude_years`` reproduces the historical analysis convention by removing
    the smoothed log-ratio value for each excluded year *before* recomputing
    first differences.  Consequently an excluded year also breaks the adjacent
    trend run rather than silently allowing the next year's difference to use it.
    """
    if int(min_consecutive_years) < 1:
        raise ValueError("min_consecutive_years must be >= 1")
    d = comparison.copy().sort_values("year").reset_index(drop=True)
    smoothcols = [f"log_{r}_smooth" for r in ratio_names]
    missing = [c for c in smoothcols if c not in d.columns]
    if missing:
        raise ValueError(f"Missing smoothed ratio columns: {missing}")
    if exclude_years:
        d.loc[d.year.isin(list(exclude_years)), smoothcols] = np.nan
    diffcols = []
    for r in ratio_names:
        dc = f"diff_{r}"
        d[dc] = d[f"log_{r}_smooth"].diff()
        diffcols.append(dc)

    directions = []
    for _, row in d.iterrows():
        vals = [row[c] for c in diffcols]
        signs = [0 if pd.isna(v) or v == 0 else 1 if v > 0 else -1 for v in vals]
        directions.append(signs[0] if signs and all(s == signs[0] and s != 0 for s in signs) else 0)
    d["common_direction"] = directions

    rows = []
    start = None
    current = 0

    def close(end_idx: int, start_idx: int | None, direction: int) -> None:
        if start_idx is None:
            return
        sub = d.loc[start_idx:end_idx]
        if len(sub) < int(min_consecutive_years):
            return
        slopes = {c: float(sub[c].mean()) for c in diffcols}
        if direction > 0 and not all(v >= float(min_avg_slope) for v in slopes.values()):
            return
        if direction < 0 and not all(v <= -float(min_avg_slope) for v in slopes.values()):
            return
        row = {
            "start_year": int(sub.year.iloc[0]),
            "end_year": int(sub.year.iloc[-1]),
            "direction": "increase" if direction > 0 else "decrease",
            "n_years": len(sub),
        }
        row.update({f"avg_{c}": v for c, v in slopes.items()})
        rows.append(row)

    for i, row in d.iterrows():
        direction = int(row.common_direction)
        if direction != 0 and start is None:
            start = i
            current = direction
        elif start is not None and direction != current:
            close(i - 1, start, current)
            start = i if direction != 0 else None
            current = direction
    if start is not None:
        close(d.index[-1], start, current)
    return pd.DataFrame(rows)


def trend_detection_sensitivity(
    series: Mapping[str, pd.DataFrame],
    focal_name: str = "fusion",
    start_year: int | None = None,
    end_year: int | None = None,
    smooth_windows: Sequence[int] = (1, 2, 3),
    min_avg_slopes: Sequence[float] = (0.0, 0.02, 0.04),
    min_consecutive_years: int = 3,
    exclude_years: Sequence[int] = (),
) -> pd.DataFrame:
    """Run period detection over a transparent parameter grid."""
    rows = []
    ratios = [f"{focal_name}_vs_{name}" for name in series if name != focal_name]
    for window in smooth_windows:
        comp = compare_publication_series(series, focal_name, start_year, end_year, int(window))
        for slope in min_avg_slopes:
            periods = detect_common_trend_periods(
                comp, ratios, min_consecutive_years=min_consecutive_years,
                min_avg_slope=float(slope), exclude_years=exclude_years,
            )
            for _, r in periods.iterrows():
                rows.append({"smooth_window": int(window), "min_avg_slope": float(slope), **r.to_dict()})
    return pd.DataFrame(rows)


def plot_publication_comparison(
    comparison: pd.DataFrame,
    periods: pd.DataFrame | None = None,
    focal_name: str = "fusion",
    baseline_names: Sequence[str] | None = None,
    plot_start_year: int | None = None,
    plot_end_year: int | None = None,
    title: str = "Fusion publication output relative to baseline fields",
    shade_periods: bool = True,
):
    """Create the publication log-ratio figure with an independent plot range.

    Returns ``(fig, ax)`` so callers can further customize the visual without
    changing the analytical functions.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    d = comparison.copy()
    if plot_start_year is not None:
        d = d[d.year >= int(plot_start_year)]
    if plot_end_year is not None:
        d = d[d.year <= int(plot_end_year)]
    if d.empty:
        raise ValueError("Plot range contains no data")
    if baseline_names is None:
        prefix = f"log_{focal_name}_vs_"
        suffix = "_smooth"
        baseline_names = [c[len(prefix):-len(suffix)] for c in d.columns if c.startswith(prefix) and c.endswith(suffix)]
    baseline_names = list(baseline_names)
    if not baseline_names:
        raise ValueError("No baseline series selected for plotting")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    if shade_periods and periods is not None and not periods.empty:
        for _, row in periods.iterrows():
            start = max(int(row.start_year), int(d.year.min()))
            end = min(int(row.end_year), int(d.year.max()))
            if start > end:
                continue
            is_increase = str(row.direction).lower() == "increase"
            shade = "#d9eaf7" if is_increase else "#f3d6d6"
            edge = "#7aa6c2" if is_increase else "#c28a8a"
            ax.axvspan(start, end, color=shade, alpha=0.45, zorder=0)
            ax.axvline(start, color=edge, linestyle="--", linewidth=1)
            ax.axvline(end, color=edge, linestyle="--", linewidth=1)

    lines = []
    for baseline in baseline_names:
        col = f"log_{focal_name}_vs_{baseline}_smooth"
        if col not in d.columns:
            raise ValueError(f"Missing comparison column: {col}")
        line, = ax.plot(d.year, d[col], linewidth=2, label=f"log({focal_name.title()} / {baseline.title()})")
        lines.append(line)

    handles = list(lines)
    if shade_periods and periods is not None and not periods.empty:
        handles.extend([
            Patch(facecolor="#d9eaf7", edgecolor="none", alpha=0.45, label="Relative increase period"),
            Patch(facecolor="#f3d6d6", edgecolor="none", alpha=0.45, label="Relative decrease period"),
        ])
    years = d.year.tolist()
    tick_step = 2 if len(years) > 25 else 1
    ax.set_xticks(years[::tick_step])
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Year")
    ax.set_ylabel("Log relative publication ratio")
    ax.set_title(title)
    ax.legend(handles=handles, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    return fig, ax


def save_publication_figure(fig, output_stem: str | Path, formats: Sequence[str] = ("png",), dpi: int = 600) -> list[Path]:
    """Save one publication-trend figure in user-selected formats."""
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        fmt = str(fmt).lower().lstrip(".")
        path = stem.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight"}
        if fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
            kwargs["dpi"] = int(dpi)
        if fmt in {"tif", "tiff"}:
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths
