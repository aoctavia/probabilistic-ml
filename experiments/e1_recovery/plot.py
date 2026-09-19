"""E1 — Generate figures from stored results.

Usage:
    python experiments/e1_recovery/plot.py \
        experiments/e1_recovery/results/ \
        experiments/e1_recovery/figures/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: Path) -> list[dict]:  # type: ignore[type-arg]
    with open(results_dir / "results.json") as f:
        return json.load(f)


def plot_sweep(
    records: list[dict],  # type: ignore[type-arg]
    dim: str,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    higher_is_better: bool = True,
) -> None:
    subset = [r for r in records if r["sweep_dim"] == dim]
    if not subset:
        return

    xs = [r["sweep_val"] for r in subset]
    ys = [r[metric] for r in subset]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(xs, ys, "o-", lw=1.8, ms=6, color="steelblue")
    ax.set_xlabel(dim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_elbo_convergence(
    records: list[dict],  # type: ignore[type-arg]
    out_path: Path,
) -> None:
    """Bar chart of iteration counts to convergence."""
    dims = sorted({r["sweep_dim"] for r in records})
    fig, axes = plt.subplots(1, len(dims), figsize=(4 * len(dims), 3.5), sharey=True)
    if len(dims) == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims):
        subset = [r for r in records if r["sweep_dim"] == dim]
        xs = [str(r["sweep_val"]) for r in subset]
        ys = [r["n_iter"] for r in subset]
        colors = ["#2ecc71" if r["converged"] else "#e74c3c" for r in subset]
        ax.bar(xs, ys, color=colors)
        ax.set_xlabel(dim)
        ax.set_title(f"Iterations (sweep {dim})")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("CAVI iterations")
    fig.suptitle("Iterations to convergence (green=converged, red=max reached)", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(results_dir: Path, figures_dir: Path) -> None:
    records = load_results(results_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dims = ["N", "D", "K", "tau"]
    for dim in dims:
        plot_sweep(
            records, dim,
            metric="subspace_distance",
            ylabel="Subspace distance",
            title=f"Factor recovery vs {dim}\n(lower = better)",
            out_path=figures_dir / f"subspace_dist_vs_{dim}.png",
            higher_is_better=False,
        )
        plot_sweep(
            records, dim,
            metric="matched_correlation",
            ylabel="Matched correlation",
            title=f"Factor recovery vs {dim}\n(higher = better)",
            out_path=figures_dir / f"matched_corr_vs_{dim}.png",
        )

    plot_elbo_convergence(records, figures_dir / "convergence.png")

    # tau recovery
    subset_tau = [r for r in records if r["sweep_dim"] == "tau"]
    if subset_tau:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        true_taus = [r["true_tau"] for r in subset_tau]
        est_taus = [r["e_tau"] for r in subset_tau]
        ax.scatter(true_taus, est_taus, zorder=3)
        lo, hi = min(true_taus) * 0.8, max(true_taus) * 1.2
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
        ax.set_xlabel("True τ")
        ax.set_ylabel("E[τ] posterior mean")
        ax.set_title("Noise precision recovery")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures_dir / "tau_recovery.png", dpi=150)
        plt.close(fig)

    print(f"Figures saved to {figures_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <results_dir> <figures_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
