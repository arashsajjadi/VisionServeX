"""Shared plot helpers for VisionServeX task notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def horizontal_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str,
    xlabel: str,
    figsize=(10, 5),
    color="#3b82f6",
    out=None,
) -> None:
    df_s = df.dropna(subset=[x]).sort_values(x, ascending=False)
    if df_s.empty:
        print(f"No data for {title}")
        return
    fig, ax = plt.subplots(figsize=(figsize[0], max(3, 0.35 * len(df_s))))
    ax.barh(df_s[y][::-1], df_s[x][::-1], color=color)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for i, v in enumerate(df_s[x][::-1]):
        ax.text(float(v) * 1.005, i, f"{float(v):.4f}", va="center", fontsize=8)
    plt.tight_layout()
    if out:
        from pathlib import Path

        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
    plt.show()
