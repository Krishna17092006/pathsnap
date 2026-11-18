"""Model-comparison plots, regenerated from saved CSVs only (never by hand).

Run from the project root after `python -m src.train`:
    python -m src.compare             # reads results/models/
    python -m src.compare --quick     # reads results/models/quick/

Writes results/models/figures/:
    fig_cv_macro_f1.png        CV macro F1 per model, one dot per fold
    fig_student_test.png       student_test Top-1 and macro F1
    fig_log_loss.png           CV and student_test log loss
    fig_inference_time.png     inference time per 1,000 fixes
    fig_overfitting.png        train vs CV macro F1 (gap = overfitting)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plot_style as ps
from .config import RESULTS_DIR


def _hbar(ax, labels, values, color, fmt="{:.3f}"):
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, height=0.6, zorder=3)
    for yi, v in zip(y, values):
        if np.isfinite(v):
            ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=8, color=ps.INK_2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)


def plot_cv_f1(table: pd.DataFrame, root: Path, fig_dir: Path) -> Path:
    models = [m for m in table["model"] if m != "baseline"]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(models) + 1.4))
    for i, m in enumerate(models):
        f = root / m / "cv_folds.csv"
        if not f.exists():
            continue
        v = pd.read_csv(f)["macro_f1"].to_numpy()
        ax.scatter(v, np.full(len(v), i), s=26, color=ps.BLUE, alpha=0.7, zorder=3, edgecolors=ps.SURFACE)
        ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=ps.INK, lw=2, zorder=4)
        ax.text(v.max(), i, f"  {v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}", va="center",
                fontsize=8, color=ps.INK_2)
    ax.set_yticks(range(len(models)), models)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Macro F1 on validation folds (GroupKFold by trip)")
    ax.set_title("Cross-validated macro F1: dots = folds, bar = mean")
    return ps.save(fig, fig_dir / "fig_cv_macro_f1.png")


def plot_student_test(table: pd.DataFrame, fig_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(table) + 1.6), sharey=True)
    _hbar(axes[0], table["model"], table["st_top1"].to_numpy(dtype=float), ps.BLUE)
    base = table.loc[table["model"] == "baseline", "st_top1"]
    if len(base):
        axes[0].axvline(float(base.iloc[0]), color=ps.CRITICAL, ls="--", lw=1, zorder=4)
    axes[0].set_xlabel("Top-1 accuracy (share of labelled fixes)")
    axes[0].set_title("Student test: Top-1 (dashed = nearest-arc baseline)")
    _hbar(axes[1], table["model"], table["st_macro_f1"].to_numpy(dtype=float), ps.ORANGE)
    axes[1].set_xlabel("Macro F1 on candidate rows")
    axes[1].set_title("Student test: macro F1")
    for ax in axes:
        ax.set_xlim(0, 1.12)
    return ps.save(fig, fig_dir / "fig_student_test.png")


def plot_log_loss(table: pd.DataFrame, fig_dir: Path) -> Path:
    t = table[table["st_log_loss"].notna()]
    fig, ax = plt.subplots(figsize=(7, 0.6 * max(len(t), 1) + 1.4))
    y = np.arange(len(t))
    ax.barh(y - 0.18, t["cv_log_loss_mean"], height=0.34, color=ps.BLUE, label="CV mean", zorder=3)
    ax.barh(y + 0.18, t["st_log_loss"], height=0.34, color=ps.ORANGE, label="Student test", zorder=3)
    ax.set_yticks(y, t["model"])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Log loss (lower is better)")
    ax.set_title("Probability quality: log loss")
    ax.legend(loc="lower right")
    return ps.save(fig, fig_dir / "fig_log_loss.png")


def plot_inference(table: pd.DataFrame, fig_dir: Path) -> Path:
    t = table[table["infer_ms_per_1k_fixes"].notna()]
    fig, ax = plt.subplots(figsize=(7, 0.5 * max(len(t), 1) + 1.4))
    _hbar(ax, t["model"], t["infer_ms_per_1k_fixes"].to_numpy(dtype=float), ps.AQUA, fmt="{:.1f} ms")
    ax.set_xscale("log")
    ax.set_xlabel("Inference time per 1,000 fixes, ms (log scale)")
    ax.set_title("Computational cost at inference")
    return ps.save(fig, fig_dir / "fig_inference_time.png")


def plot_overfitting(table: pd.DataFrame, fig_dir: Path) -> Path:
    t = table[table.get("cv_train_macro_f1_mean", pd.Series(dtype=float)).notna()] \
        if "cv_train_macro_f1_mean" in table.columns else table.iloc[:0]
    fig, ax = plt.subplots(figsize=(7, 0.6 * max(len(t), 1) + 1.4))
    y = np.arange(len(t))
    ax.barh(y - 0.18, t.get("cv_train_macro_f1_mean", []), height=0.34, color=ps.BLUE, label="Training folds",
            zorder=3)
    ax.barh(y + 0.18, t.get("cv_macro_f1_mean", []), height=0.34, color=ps.ORANGE, label="Validation folds",
            zorder=3)
    ax.set_yticks(y, t["model"])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Macro F1 (mean over folds)")
    ax.set_title("Train vs validation macro F1: a large gap means overfitting")
    ax.legend(loc="lower right")
    return ps.save(fig, fig_dir / "fig_overfitting.png")


def make_all(root: Path) -> list[Path]:
    path = root / "model_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.train` first.")
    table = pd.read_csv(path)
    ps.apply_style()
    fig_dir = root / "figures"
    return [plot_cv_f1(table, root, fig_dir), plot_student_test(table, fig_dir), plot_log_loss(table, fig_dir),
            plot_inference(table, fig_dir), plot_overfitting(table, fig_dir)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Plot the model comparison")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "models")
    args = ap.parse_args(argv)
    root = args.results_dir / "quick" if args.quick else args.results_dir
    for p in make_all(root):
        print(p)


if __name__ == "__main__":
    main()
