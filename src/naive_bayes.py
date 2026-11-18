"""Model 3 (KKR): Gaussian Naive Bayes.

Uses the shared features (src/features.py) and the shared evaluation framework
(src/evaluate.py) exactly like every other model:
  1. GroupKFold by trip on the 80 train trips (preprocessing fitted inside each fold)
  2. refit on all train trips, evaluate once on the 10 student_test trips

Gaussian Naive Bayes assumes that, within each class, every feature is
  (1) Gaussian, and
  (2) independent of every other feature.
Neither holds for our features:
  - several features describe the same distance (dist_m, dist_minus_min, dist_ratio, cand_rank)
  - the heading features are near-duplicates
  - many features are binary or heavily skewed
So three configurations test the two assumptions:

  gnb_all           all 19 features, raw                                   (plain model)
  gnb_quantile      quantile-normal transform first                         (targets assumption 1)
  gnb_decorrelated  quantile transform + drop features with |corr| > 0.8    (targets assumptions 1 and 2)

The correlation filter is a pipeline step, so it is fitted inside each training fold only.
The configuration reported for the model comparison is chosen by mean CV Top-1, never by
student_test results.

Run from the project root (after `python -m src.build_dataset` and `python -m src.features`):
    python -m src.naive_bayes

Writes results/models/naive_bayes/:
    config.json, cv_folds.csv, student_test_metrics.json, summary.csv, summary.md, figures/*.png
"""
from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from . import plot_style as ps
from .config import N_FOLDS, PROCESSED_DIR, RESULTS_DIR, SEED
from .evaluate import evaluate_predictions, fit_predict, group_cv, per_1k_fixes_ms, train_fit_metrics
from .features import FEATURE_COLUMNS, load_or_build

OUT_DIR = RESULTS_DIR / "models" / "naive_bayes"
CORR_THRESHOLD = 0.8
CORR_SAMPLE_ROWS = 200_000
METRICS = ["macro_f1", "top1", "log_loss", "route_continuity", "accuracy", "precision", "recall"]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Keep a subset of features whose pairwise |Pearson r| stays <= threshold.

    Features are considered in order of |correlation with the label| (most informative
    first); a feature is kept unless it is too correlated with one already kept.
    Fitted on the training rows it is given, so inside CV it never sees validation data.
    """

    def __init__(self, threshold: float = CORR_THRESHOLD, max_rows: int = CORR_SAMPLE_ROWS, random_state: int = SEED):
        self.threshold = threshold
        self.max_rows = max_rows
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype="float64")
        y = np.asarray(y, dtype="float64")
        if len(X) > self.max_rows:
            idx = np.random.default_rng(self.random_state).choice(len(X), self.max_rows, replace=False)
            X, y = X[idx], y[idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.nan_to_num(np.corrcoef(np.column_stack([X, y]), rowvar=False))
        feat_corr, rel = np.abs(corr[:-1, :-1]), np.abs(corr[:-1, -1])
        keep: list[int] = []
        for j in np.argsort(-rel, kind="stable"):
            if np.std(X[:, j]) == 0:
                continue
            if all(feat_corr[j, k] <= self.threshold for k in keep):
                keep.append(int(j))
        self.keep_ = np.array(sorted(keep))
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_]


def _quantile() -> QuantileTransformer:
    return QuantileTransformer(output_distribution="normal", n_quantiles=1000, subsample=200_000, random_state=SEED)


def gnb_all(var_smoothing: float = 1e-9):
    return GaussianNB(var_smoothing=var_smoothing)


def gnb_quantile(var_smoothing: float = 1e-9):
    return Pipeline([("quantile", _quantile()), ("gnb", GaussianNB(var_smoothing=var_smoothing))])


def gnb_decorrelated(var_smoothing: float = 1e-9, corr_threshold: float = CORR_THRESHOLD):
    return Pipeline([("quantile", _quantile()), ("decorrelate", CorrelationFilter(threshold=corr_threshold)),
                     ("gnb", GaussianNB(var_smoothing=var_smoothing))])


CONFIGS = {
    "gnb_all": (gnb_all, {"var_smoothing": 1e-9},
                "GaussianNB on all 19 raw features"),
    "gnb_quantile": (gnb_quantile, {"var_smoothing": 1e-9},
                     "QuantileTransformer(normal) -> GaussianNB"),
    "gnb_decorrelated": (gnb_decorrelated, {"var_smoothing": 1e-9, "corr_threshold": CORR_THRESHOLD},
                         f"QuantileTransformer(normal) -> drop |r| > {CORR_THRESHOLD} -> GaussianNB"),
}


# ------------------------------------------------------------------------------ figures
def correlation_figure(train: pd.DataFrame, path) -> None:
    """|r| between features on (a sample of) labelled train rows: evidence against independence."""
    rows = train[train["fix_valid"].astype(bool)]
    rows = rows.sample(min(len(rows), CORR_SAMPLE_ROWS), random_state=SEED)
    corr = rows[FEATURE_COLUMNS].corr().fillna(0).to_numpy()
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    n = len(FEATURE_COLUMNS)
    ax.set_xticks(range(n), FEATURE_COLUMNS, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n), FEATURE_COLUMNS, fontsize=8)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            if i != j and abs(corr[i, j]) > CORR_THRESHOLD:
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="#ffffff")
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("Pearson r")
    ax.set_title(f"Feature correlation on train (labelled rows, sample of {len(rows):,})")
    fig.text(0.01, -0.06, f"Numbers mark pairs with |r| > {CORR_THRESHOLD}: Naive Bayes counts that evidence twice.",
             fontsize=8, color=ps.MUTED)
    ps.save(fig, path)


def variants_figure(folds: pd.DataFrame, test: dict, path) -> None:
    ps.apply_style()
    names = list(CONFIGS)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, (metric, label) in zip(axes, [("macro_f1", "Macro F1 (candidate level)"), ("top1", "Top-1 accuracy (fix level)"),
                                          ("log_loss", "Log loss (lower is better)")]):
        for i, name in enumerate(names):
            v = folds.loc[folds["config"] == name, metric].to_numpy()
            ax.scatter(v, np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), s=26, color=ps.BLUE,
                       edgecolors=ps.SURFACE, linewidths=0.8, zorder=3, label="CV fold" if i == 0 else None)
            ax.plot([v.mean()] * 2, [i - 0.28, i + 0.28], color=ps.INK, lw=1.8, zorder=4,
                    label="CV mean" if i == 0 else None)
            ax.scatter([test[name][metric]], [i], marker="D", s=46, color=ps.ORANGE, edgecolors=ps.SURFACE,
                       linewidths=0.8, zorder=5, label="Student test" if i == 0 else None)
        ax.set_yticks(range(len(names)), names if ax is axes[0] else [""] * len(names))
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 1.03), fontsize=8)
    fig.suptitle("Gaussian Naive Bayes configurations", x=0.01, ha="left", fontweight="bold", fontsize=11)
    ps.save(fig, path)


# ------------------------------------------------------------------------------ main
def main(argv=None) -> None:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading features ...", flush=True)
    train = load_or_build("train")
    test = load_or_build("student_test")
    fixes_train = pd.read_parquet(PROCESSED_DIR / "fixes_train.parquet")
    fixes_test = pd.read_parquet(PROCESSED_DIR / "fixes_student_test.parquet")

    fold_frames, test_metrics, kept_features = [], {}, {}
    for name, (factory, params, desc) in CONFIGS.items():
        print(f"\n[{name}] {desc}", flush=True)
        folds = group_cv(factory, params, train, fixes_train, FEATURE_COLUMNS, n_folds=N_FOLDS, seed=SEED)
        folds.insert(0, "config", name)
        fold_frames.append(folds)

        model, prob, fit_s, pred_s = fit_predict(factory, params, train, test, FEATURE_COLUMNS, seed=SEED)
        m = evaluate_predictions(test, prob, fixes_test)
        m.update(train_fit_metrics(model, train, FEATURE_COLUMNS, SEED))
        m.update({"train_time_s": fit_s, "infer_ms_per_1k_fixes": per_1k_fixes_ms(pred_s, test)})
        test_metrics[name] = m
        if isinstance(model, Pipeline) and "decorrelate" in model.named_steps:
            kept_features[name] = [FEATURE_COLUMNS[j] for j in model.named_steps["decorrelate"].keep_]
        print(f"  student_test: macro F1 {m['macro_f1']:.4f}  Top-1 {m['top1']:.4f}  log loss {m['log_loss']:.4f}  "
              f"continuity {m['route_continuity']:.4f}", flush=True)

    folds = pd.concat(fold_frames, ignore_index=True)
    folds.to_csv(OUT_DIR / "cv_folds.csv", index=False)
    (OUT_DIR / "student_test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    rows = []
    for name, (_, params, desc) in CONFIGS.items():
        f = folds[folds["config"] == name]
        t = test_metrics[name]
        rows.append({
            "config": name, "family": "Probabilistic (Gaussian Naive Bayes)", "description": desc,
            "cv_macro_f1_mean": f["macro_f1"].mean(), "cv_macro_f1_std": f["macro_f1"].std(),
            "cv_top1_mean": f["top1"].mean(), "cv_top1_std": f["top1"].std(),
            "cv_log_loss_mean": f["log_loss"].mean(), "cv_route_continuity_mean": f["route_continuity"].mean(),
            "cv_train_macro_f1_mean": f["train_macro_f1"].mean(),
            "st_macro_f1": t["macro_f1"], "st_top1": t["top1"], "st_log_loss": t["log_loss"],
            "st_route_continuity": t["route_continuity"], "st_precision": t["precision"], "st_recall": t["recall"],
            "train_time_s": t["train_time_s"], "infer_ms_per_1k_fixes": t["infer_ms_per_1k_fixes"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    selected = summary.loc[summary["cv_top1_mean"].idxmax(), "config"]

    config = {
        "model": "Gaussian Naive Bayes", "owner": "KKR", "seed": SEED, "n_folds": N_FOLDS,
        "cv": "GroupKFold by trip on the 80 train trips (src/evaluate.py group_cv)",
        "features": FEATURE_COLUMNS,
        "configs": {n: {"description": d, "params": p} for n, (_, p, d) in CONFIGS.items()},
        "kept_features_final_fit": kept_features,
        "selection_rule": "highest mean CV Top-1 (student_test never used for selection)",
        "selected_config": selected,
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    fig_dir = OUT_DIR / "figures"
    correlation_figure(train, fig_dir / "feature_correlation.png")
    variants_figure(folds, test_metrics, fig_dir / "nb_configurations.png")

    def pct(x):
        return f"{x * 100:.1f}%"

    L = ["# Model 3: Gaussian Naive Bayes (KKR)", "",
         "Generated by `python -m src.naive_bayes`. Shared features (`src/features.py`) and evaluation "
         "(`src/evaluate.py`): 5-fold GroupKFold by trip on 80 train trips, then refit on all train trips and "
         "one evaluation on the 10 student_test trips.", "",
         "| Config | CV macro F1 | CV Top-1 | CV log loss | Train macro F1 | Student-test macro F1 | Student-test Top-1 | Student-test log loss | Route continuity (ST) | Inference ms / 1k fixes |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary.itertuples():
        L.append(f"| {r.config}{' (selected)' if r.config == selected else ''} | {r.cv_macro_f1_mean:.4f} ± {r.cv_macro_f1_std:.4f} | "
                 f"{pct(r.cv_top1_mean)} ± {r.cv_top1_std * 100:.1f} | {r.cv_log_loss_mean:.4f} | {r.cv_train_macro_f1_mean:.4f} | "
                 f"{r.st_macro_f1:.4f} | {pct(r.st_top1)} | {r.st_log_loss:.4f} | {pct(r.st_route_continuity)} | "
                 f"{r.infer_ms_per_1k_fixes:.1f} |")
    L += ["", f"**Selected configuration:** `{selected}` (highest mean CV Top-1; student_test not used for selection).", ""]
    for name, feats in kept_features.items():
        dropped = [f for f in FEATURE_COLUMNS if f not in feats]
        L.append(f"`{name}` kept {len(feats)} of {len(FEATURE_COLUMNS)} features in the final fit; dropped: "
                 + ", ".join(f"`{d}`" for d in dropped) + ".")
    L += ["", "Over/underfitting check: compare *Train macro F1* (on training folds) with *CV macro F1* "
          "(on held-out trips). A small gap means no overfitting, which is expected for a model this simple.", "",
          "Figures: `figures/feature_correlation.png`, `figures/nb_configurations.png`.", "",
          f"Runtime: {time.time() - t_start:.0f} s.", ""]
    (OUT_DIR / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L[4:4 + len(summary) + 2]))
    print(f"\nselected: {selected}  ({time.time() - t_start:.0f}s total) -> {OUT_DIR}")


if __name__ == "__main__":
    main()
