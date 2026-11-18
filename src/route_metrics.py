"""Metrics for classifier-only and HMM map-matching predictions."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def _arc_equivalent(pred: int, truth: int, net) -> bool:
    if pred < 0 or truth < 0:
        return False
    return bool((pred == truth) or (net.u[pred] == net.u[truth] and net.v[pred] == net.v[truth]))


def pointwise_accuracy(fixes: pd.DataFrame, predicted_arcs, net) -> dict:
    """Top-1 accuracy on valid fixes, accepting exact duplicate directed arcs."""
    pred = np.asarray(predicted_arcs, dtype="int64")
    if len(pred) != len(fixes):
        raise ValueError("predicted_arcs must align with fixes rows")
    if "fix_valid" not in fixes or "true_arc" not in fixes:
        return {"top1": float("nan"), "n_valid_fixes": 0}
    valid = fixes["fix_valid"].astype(bool).to_numpy()
    truth = fixes["true_arc"].to_numpy(dtype="int64")
    idx = np.flatnonzero(valid)
    hits = np.fromiter((_arc_equivalent(int(pred[i]), int(truth[i]), net) for i in idx),
                        dtype=bool, count=len(idx))
    return {"top1": float(hits.mean()) if len(hits) else float("nan"),
            "n_valid_fixes": int(len(idx)), "n_correct_fixes": int(hits.sum())}


def route_continuity(fixes: pd.DataFrame, predicted_arcs, net,
                     gap_threshold_s: float = 60.0) -> float:
    """Share of adjacent, temporally-neighbouring fixes whose arcs are same or connected."""
    pred = np.asarray(predicted_arcs, dtype="int64")
    if len(pred) != len(fixes):
        raise ValueError("predicted_arcs must align with fixes rows")
    if len(pred) < 2:
        return float("nan")
    order = np.argsort(fixes["fix_idx"].to_numpy(dtype="int64"), kind="stable")
    p = pred[order]
    t = fixes["t"].to_numpy(dtype="float64")[order]
    pairs = []
    for i in range(len(p) - 1):
        if p[i] < 0 or p[i + 1] < 0:
            continue
        dt = t[i + 1] - t[i]
        if not np.isfinite(dt) or dt <= 0 or dt > gap_threshold_s:
            continue
        a, b = int(p[i]), int(p[i + 1])
        pairs.append((a == b) or (net.v[a] == net.u[b]))
    return float(np.mean(pairs)) if pairs else float("nan")


def sequence_metrics(predicted_route, reference_route, net=None) -> dict:
    """Multiset precision/recall/F1 of route-arc occurrences (order is not scored).

    Consecutive duplicate arcs are collapsed before scoring, matching the route
    representation returned by the prediction pipeline. If a network is supplied,
    exact duplicate directed arcs are canonicalised by their (u, v) endpoints.
    """
    def collapse(values):
        out = []
        for x in values if values is not None else []:
            x = int(x)
            if x < 0:
                continue
            if not out or out[-1] != x:
                out.append(x)
        return out

    pred = collapse(predicted_route)
    ref = collapse(reference_route)
    if net is not None:
        pred = [(int(net.u[a]), int(net.v[a])) for a in pred]
        ref = [(int(net.u[a]), int(net.v[a])) for a in ref]
    pc, rc = Counter(pred), Counter(ref)
    overlap = sum((pc & rc).values())
    precision = overlap / sum(pc.values()) if pc else 0.0
    recall = overlap / sum(rc.values()) if rc else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"route_precision": float(precision), "route_recall": float(recall),
            "route_f1": float(f1), "n_predicted_route_arcs": int(sum(pc.values())),
            "n_reference_route_arcs": int(sum(rc.values()))}


def compare_predictions(fixes: pd.DataFrame, classifier_arcs, hmm_arcs, reference_route,
                        net, gap_threshold_s: float = 60.0) -> dict:
    """Return the report/API metric set for classifier-only and classifier+HMM."""
    clf = pointwise_accuracy(fixes, classifier_arcs, net)
    hmm = pointwise_accuracy(fixes, hmm_arcs, net)
    clf_route = [int(x) for x in classifier_arcs if int(x) >= 0]
    hmm_route = [int(x) for x in hmm_arcs if int(x) >= 0]
    clf_route = [x for i, x in enumerate(clf_route) if i == 0 or x != clf_route[i - 1]]
    hmm_route = [x for i, x in enumerate(hmm_route) if i == 0 or x != hmm_route[i - 1]]
    return {
        "top1_classifier": clf["top1"], "top1_hmm": hmm["top1"],
        "route_continuity_clf": route_continuity(fixes, classifier_arcs, net, gap_threshold_s),
        "route_continuity_hmm": route_continuity(fixes, hmm_arcs, net, gap_threshold_s),
        **{f"classifier_{k}": v for k, v in sequence_metrics(clf_route, reference_route, net).items()},
        **{f"hmm_{k}": v for k, v in sequence_metrics(hmm_route, reference_route, net).items()},
        "n_fixes": int(len(fixes)), "n_valid_fixes": int(clf["n_valid_fixes"]),
    }
