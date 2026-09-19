"""Shared visualisation utilities.

Figures are always saved to disk and returned as ``matplotlib.Figure``
objects — never shown interactively so experiments work headlessly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from sfm.types import FloatArray


def plot_elbo_trace(
    elbo_history: FloatArray | list[float],
    *,
    title: str = "ELBO trace",
    out_path: Path | None = None,
) -> Figure:
    """Plot ELBO vs. iteration and optionally save to disk.

    Parameters
    ----------
    elbo_history:
        Sequence of ELBO values, one per CAVI iteration.
    title:
        Figure title.
    out_path:
        If given, the figure is saved here (PNG, 150 dpi).

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.asarray(elbo_history), lw=1.5, color="steelblue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("ELBO")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)

    return fig


def plot_factor_heatmap(
    matrix: FloatArray,
    *,
    row_label: str = "Observations",
    col_label: str = "Factors",
    title: str = "",
    out_path: Path | None = None,
) -> Figure:
    """Heatmap of a matrix (e.g., factor scores Z or loadings W).

    Parameters
    ----------
    matrix:
        2-D array of shape ``(rows, cols)``.
    out_path:
        Optional save path.

    Returns
    -------
    matplotlib.Figure
    """
    arr = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(max(3, arr.shape[1] * 0.5), max(3, arr.shape[0] * 0.05 + 1)))
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    ax.set_title(title)
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)

    return fig
