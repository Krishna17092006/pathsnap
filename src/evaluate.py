"""Evaluation framework shared by every model (TEAM.md sections 2.4 and 3).

Two levels of metrics:

* candidate level (binary rows: is this candidate the true arc?):
  accuracy, precision, recall, macro F1, confusion matrix, log loss
* fix level (one decision per GPS fix): Top-1 accuracy, candidate miss rate,
  classifier-only route continuity

Top-1 and the miss rate are computed over the **fix table** (every labelled fix),
so a fix whose true arc is missing from its candidates, or that has no candidates
at all, counts as wrong. A fix is correct when its highest-scored candidate has
`label == 1` (the true arc or an exact duplicate of it).

Cross-validation is GroupKFold on `trip_id`: a whole trip stays in one fold, because
consecutive fixes of one trip are nearly identical and a row-level split would leak.
Every model is an sklearn estimator or Pipeline, so preprocessing such as scaling is
fitted inside each training fold only.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold

from .config import N_FOLDS, SEED

PROB_EPS = 1e-6
TRAIN_METRIC_ROWS = 200_000   # rows sampled from the training fold to measure over/underfitting


# ------------------------------------------------------------------------------ metrics
def candidate_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    """Binary metrics on candidate rows. `prob` = P(candidate is the true arc)."""
    y = np.asarray(y_true, dtype="int64")
    p = np.clip(np.asarray(prob, dtype="float64"), PROB_EPS, 1 - PROB_EPS)
    pred = (p >= threshold).astype("int64")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def top_candidates(cand: pd.DataFrame, prob) -> pd.DataFrame:
    """Highest-scored candidate per fix (ties broken by distance rank). One row per fix."""
    d = cand[["trip_id", "fix_idx", "arc_id", "cand_rank", "label"]].copy()
    for c in ("arc_u", "arc_v"):
        if c in cand.columns:
            d[c] = cand[c].to_numpy()
    d["prob"] = np.asarray(prob, dtype="float64")
    d = d.sort_values(["trip_id", "fix_idx", "prob", "cand_rank"], ascending=[True, True, False, True], kind="stable")
    return d.drop_duplicates(["trip_id", "fix_idx"], keep="first").reset_index(drop=True)


def fix_metrics(cand: pd.DataFrame, prob, fixes: pd.DataFrame) -> dict:
    """Top-1 accuracy and candidate miss rate over the valid fixes of the fix table."""
    top = top_candidates(cand, prob)
    valid = fixes[fixes["fix_valid"].astype(bool)][["trip_id", "fix_idx", "true_in_cands"]]
    merged = valid.merge(top[["trip_id", "fix_idx", "label"]], on=["trip_id", "fix_idx"], how="left")
    hit = merged["label"].fillna(0).to_numpy() == 1
    n = len(merged)
    return {
        "top1": float(hit.mean()) if n else float("nan"),
        "candidate_miss_rate": float((~merged["true_in_cands"].astype(bool)).mean()) if n else float("nan"),
        "n_valid_fixes": int(n),
        "n_fixes": int(len(fixes)),
    }


def route_continuity(cand: pd.DataFrame, prob) -> float:
    """Classifier-only continuity: share of consecutive predicted fixes whose arcs are the
    same arc or directly connected (end node of the first = start node of the second).

    Uses every fix that has candidates, labelled or not (no ground truth involved).
    Consecutive = adjacent in time among fixes with a prediction, within one trip.
    """
    if not {"arc_u", "arc_v"} <= set(cand.columns):
        return float("nan")
    top = top_candidates(cand, prob)
    same_trip = top["trip_id"].to_numpy()[1:] == top["trip_id"].to_numpy()[:-1]
    a, u, v = top["arc_id"].to_numpy(), top["arc_u"].to_numpy(), top["arc_v"].to_numpy()
    ok = (a[1:] == a[:-1]) | (v[:-1] == u[1:])
    return float(ok[same_trip].mean()) if same_trip.any() else float("nan")


def evaluate_predictions(cand: pd.DataFrame, prob, fixes: pd.DataFrame, threshold: float = 0.5) -> dict:
    """All metrics for one set of predictions. Candidate metrics use labelled fixes only."""
    valid = cand["fix_valid"].astype(bool).to_numpy()
    out = candidate_metrics(cand["label"].to_numpy()[valid], np.asarray(prob)[valid], threshold)
    out.update(fix_metrics(cand, prob, fixes))
    out["route_continuity"] = route_continuity(cand, prob)
    return out


def baseline_scores(cand: pd.DataFrame) -> np.ndarray:
    """Naive nearest-arc baseline: score 1 for the nearest candidate, 0 for the rest."""
    return (cand["cand_rank"].to_numpy() == 0).astype("float64")


def baseline_metrics(cand: pd.DataFrame, fixes: pd.DataFrame) -> dict:
    """Baseline metrics. It outputs hard 0/1 scores, so log loss is not meaningful (NaN)."""
    out = evaluate_predictions(cand, baseline_scores(cand), fixes)
    out["log_loss"] = float("nan")
    return out


# ------------------------------------------------------------------------------ training
def training_rows(cand: pd.DataFrame, max_rows: int | None = None, seed: int = SEED) -> pd.DataFrame:
    """Labelled candidate rows used for fitting, optionally subsampled (stratified by label)."""
    rows = cand[cand["fix_valid"].astype(bool)]
    if max_rows is not None and len(rows) > max_rows:
        frac = max_rows / len(rows)
        rows = rows.groupby("label", group_keys=False).sample(frac=frac, random_state=seed)
    return rows


def fit_predict(factory: Callable, params: dict, train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                max_train_rows: int | None = None, seed: int = SEED):
    """Fit a fresh model on `train` and score `test`. Returns (model, prob, fit_seconds, predict_seconds)."""
    rows = training_rows(train, max_train_rows, seed)
    if rows["label"].nunique() < 2:
        raise ValueError("training rows contain a single class")
    model = factory(**params)
    t0 = time.perf_counter()
    model.fit(rows[features].to_numpy(dtype="float64"), rows["label"].to_numpy(dtype="int64"))
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    prob = model.predict_proba(test[features].to_numpy(dtype="float64"))[:, 1] if len(test) else np.array([])
    pred_s = time.perf_counter() - t0
    return model, prob, fit_s, pred_s


def train_fit_metrics(model, train: pd.DataFrame, features: list[str], seed: int = SEED) -> dict:
    """Candidate-level macro F1 and log loss on (a sample of) the training rows: overfitting evidence."""
    rows = training_rows(train, TRAIN_METRIC_ROWS, seed)
    prob = model.predict_proba(rows[features].to_numpy(dtype="float64"))[:, 1]
    m = candidate_metrics(rows["label"].to_numpy(), prob)
    return {"train_macro_f1": m["macro_f1"], "train_log_loss": m["log_loss"]}


def per_1k_fixes_ms(seconds: float, cand: pd.DataFrame) -> float:
    n = cand[["trip_id", "fix_idx"]].drop_duplicates().shape[0]
    return float(seconds * 1000.0 / n * 1000.0) if n else float("nan")


def group_folds(cand: pd.DataFrame, n_folds: int = N_FOLDS):
    """Yield (fold, train_trip_ids, val_trip_ids). Deterministic, trip-disjoint."""
    trips = np.array(sorted(cand["trip_id"].unique()))
    if len(trips) < n_folds:
        raise ValueError(f"{len(trips)} trips cannot be split into {n_folds} folds")
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(trips, groups=trips)):
        yield k, set(trips[tr]), set(trips[va])


def group_cv(factory: Callable | None, params: dict, cand: pd.DataFrame, fixes: pd.DataFrame, features: list[str],
             n_folds: int = N_FOLDS, max_train_rows: int | None = None, seed: int = SEED,
             verbose: bool = True) -> pd.DataFrame:
    """Group-aware CV on the train partition. One row of metrics per fold.

    `factory=None` evaluates the nearest-arc baseline on the same validation folds.
    """
    rows = []
    for k, tr_ids, va_ids in group_folds(cand, n_folds):
        tr = cand[cand["trip_id"].isin(tr_ids)]
        va = cand[cand["trip_id"].isin(va_ids)]
        va_fix = fixes[fixes["trip_id"].isin(va_ids)]
        rec = {"fold": k, "n_train_trips": len(tr_ids), "n_val_trips": len(va_ids)}
        if factory is None:
            rec.update(baseline_metrics(va, va_fix))
        else:
            model, prob, fit_s, pred_s = fit_predict(factory, params, tr, va, features, max_train_rows, seed)
            rec.update(evaluate_predictions(va, prob, va_fix))
            rec.update(train_fit_metrics(model, tr, features, seed))
            rec.update({"train_time_s": fit_s, "infer_ms_per_1k_fixes": per_1k_fixes_ms(pred_s, va)})
        rows.append(rec)
        if verbose:
            print(f"  fold {k}: macro F1 {rec['macro_f1']:.4f}  Top-1 {rec['top1']:.4f}  "
                  f"log loss {rec['log_loss']:.4f}", flush=True)
    return pd.DataFrame(rows)
