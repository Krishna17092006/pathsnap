"""Export the final model and everything inference needs into artifacts/ (TEAM.md section 2.5).

    artifacts/
    ├── model.joblib     final model (sklearn Pipeline, preprocessing included)
    ├── features.json    ordered model-input columns
    ├── hmm.json         HMM / Viterbi decoder parameters
    └── metadata.json    model name, source, parameters, scores, checksums, versions

Run from the project root:
    python -m src.export                                  # xgboost; tuned model if available, else refit
    python -m src.export --model xgboost --source tuned   # require the Optuna-tuned model
    python -m src.export --hmm-config results/hmm/decoder_params.json
    python -m src.export --check                          # verify an existing artifacts/ folder

Model source
    tuned  results/tuning/<model>/tuned_model.joblib from `python -m src.tune` (fitted on all train trips)
    refit  fit the registered model (src/model_zoo.py) on all labelled train rows
    auto   tuned when it exists, otherwise refit (default)

Only train trips are used for fitting. Student-test and evaluator trips are never used here.
The API and the prediction pipeline load the folder with `read_artifacts()`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import ARTIFACTS_DIR, CANDIDATE_RADIUS_M, LABEL_MAX_DIST_M, N_FOLDS, PROCESSED_DIR, RESULTS_DIR, SEED
from .data_io import DATA_DIR
from .evaluate import training_rows
from .features import FEATURE_COLUMNS, load_or_build, save_feature_list
from .hmm import DecoderConfig
from .model_zoo import canonical_model_name, get_spec

MODEL_FILE, FEATURES_FILE, HMM_FILE, METADATA_FILE = "model.joblib", "features.json", "hmm.json", "metadata.json"

# hmm.json uses the field names of src.hmm.DecoderConfig, which src.pipeline.load_artifacts reads.
# Priority: --hmm-config (e.g. results/hmm/best_decoder.json) > an existing artifacts/hmm.json > these defaults.
DEFAULT_HMM = {**DecoderConfig().to_dict(),
               "status": "default (untuned): run `python -m src.tune_hmm` or pass --hmm-config"}


def _decoder_fields(values: dict) -> dict:
    """Keep only DecoderConfig fields and check they form a valid configuration."""
    allowed = set(DecoderConfig.__dataclass_fields__)
    fields = {k: v for k, v in values.items() if k in allowed}
    DecoderConfig(**fields)   # raises on invalid values
    return fields


@dataclass
class Artifacts:
    """Everything the prediction pipeline and API need, loaded from an artifacts folder."""
    model: Any
    features: list[str]
    hmm: dict
    metadata: dict
    path: Path

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """P(candidate is the true arc) for rows that contain every feature column."""
        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing}")
        from .pipeline import model_input   # local import: pipeline is the inference module
        return self.model.predict_proba(model_input(self.model, frame[self.features]))[:, 1]


# ------------------------------------------------------------------------------ helpers
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(_json_safe(data), indent=2), encoding="utf-8")


def _versions() -> dict:
    out = {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
           "pandas": pd.__version__, "joblib": joblib.__version__}
    try:
        import xgboost
        out["xgboost"] = xgboost.__version__
    except Exception:   # xgboost missing or its native library cannot load
        out["xgboost"] = None
    return out


def _comparison_row(model: str, results_dir: Path) -> dict:
    path = results_dir / "models" / "model_comparison.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    row = table[table["model"] == model]
    if row.empty:
        return {}
    keep = ["cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_mean", "cv_top1_std", "cv_log_loss_mean",
            "st_top1", "st_macro_f1", "st_log_loss", "st_route_continuity", "infer_ms_per_1k_fixes"]
    return {k: row.iloc[0][k] for k in keep if k in row.columns}


# ------------------------------------------------------------------------------ model sources
def _tuned_source(model: str, results_dir: Path):
    d = results_dir / "tuning" / model
    model_path = d / "tuned_model.joblib"
    if not model_path.exists():
        return None
    info = {"source": "tuned", "source_file": model_path}
    best = d / "best_params.json"
    if best.exists():
        doc = json.loads(best.read_text())
        info["params"] = doc.get("best_params", {})
        info["tuning"] = {k: doc.get(k) for k in ("objective", "best_value", "n_trials_total", "search_seconds",
                                                 "max_train_rows_per_fold_during_search", "final_fit_seconds")}
    comparison = d / "tuned_vs_untuned.csv"
    if comparison.exists():
        info["tuned_vs_untuned"] = pd.read_csv(comparison).to_dict(orient="records")
    return joblib.load(model_path), info


def _refit(model: str, processed_dir: Path, data_dir: Path):
    spec = get_spec(model)
    cand = load_or_build("train", processed_dir, data_dir)
    rows = training_rows(cand, spec.max_train_rows, SEED)
    if rows["label"].nunique() < 2:
        raise ValueError("training rows contain a single class")
    est = spec.factory(**spec.params)
    t0 = time.perf_counter()
    est.fit(rows[FEATURE_COLUMNS].to_numpy(dtype="float64"), rows["label"].to_numpy(dtype="int64"))
    info = {"source": "refit", "params": spec.params, "fit_seconds": time.perf_counter() - t0,
            "train_trips": int(cand["trip_id"].nunique()), "train_rows": int(len(rows)),
            "max_train_rows": spec.max_train_rows}
    return est, info


# ------------------------------------------------------------------------------ export / check
def export(model: str = "xgboost", source: str = "auto", out_dir: Path = ARTIFACTS_DIR,
           results_dir: Path = RESULTS_DIR, processed_dir: Path = PROCESSED_DIR, data_dir: Path = DATA_DIR,
           hmm_config: Path | None = None) -> Path:
    """Write the four artifact files and verify they load. Returns the artifacts folder."""
    model = canonical_model_name(model)
    if source not in {"auto", "tuned", "refit"}:
        raise ValueError("source must be 'auto', 'tuned' or 'refit'")
    spec = get_spec(model)
    out_dir, results_dir = Path(out_dir), Path(results_dir)

    loaded = _tuned_source(model, results_dir) if source in {"auto", "tuned"} else None
    if loaded is None and source == "tuned":
        raise FileNotFoundError(f"{results_dir / 'tuning' / model / 'tuned_model.joblib'} not found. "
                                f"Run `python -m src.tune --model {model}` first, or use --source refit.")
    estimator, info = loaded if loaded is not None else _refit(model, processed_dir, data_dir)
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("the exported model must implement predict_proba")

    hmm = dict(DEFAULT_HMM)
    existing = out_dir / HMM_FILE
    if hmm_config is not None or existing.exists():
        src_path = Path(hmm_config) if hmm_config is not None else existing
        loaded_hmm = json.loads(src_path.read_text())
        if not isinstance(loaded_hmm, dict):
            raise ValueError(f"{src_path} must contain a JSON object")
        status = loaded_hmm.get("status") if hmm_config is None else None
        hmm = {**DecoderConfig().to_dict(), **_decoder_fields(loaded_hmm),
               "status": status or f"tuned (from {src_path.as_posix()})"}

    # write into a temporary folder first so a failure never leaves a half-written artifacts/
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    joblib.dump(estimator, tmp / MODEL_FILE, compress=3)
    save_feature_list(tmp / FEATURES_FILE)
    _write_json(tmp / HMM_FILE, hmm)
    metadata = {
        "model": model,
        "family": spec.family,
        "description": spec.description if info["source"] == "refit" else f"{spec.description} (Optuna-tuned)",
        "source": info.pop("source"),
        "params": info.pop("params", {}),
        "details": info,
        "n_features": len(FEATURE_COLUMNS),
        "model_file_sha256": _sha256(tmp / MODEL_FILE),
        "model_file_mb": (tmp / MODEL_FILE).stat().st_size / 1e6,
        "scores_untuned_model_comparison": _comparison_row(model, results_dir),
        "seed": SEED, "n_folds": N_FOLDS,
        "candidate_radius_m": CANDIDATE_RADIUS_M, "label_max_dist_m": LABEL_MAX_DIST_M,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": _versions(),
        "data_rule": "fitted on train trips only; student_test and evaluator trips never used for fitting",
    }
    _write_json(tmp / METADATA_FILE, metadata)
    check_artifacts(tmp)

    if out_dir.exists():
        for name in (MODEL_FILE, FEATURES_FILE, HMM_FILE, METADATA_FILE):
            (out_dir / name).unlink(missing_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in tmp.iterdir():
        shutil.move(str(f), out_dir / f.name)
    tmp.rmdir()
    return out_dir


def read_artifacts(path: Path = ARTIFACTS_DIR, verify_checksum: bool = True) -> Artifacts:
    """Load an artifacts folder (the API calls this once at startup)."""
    path = Path(path)
    for name in (MODEL_FILE, FEATURES_FILE, HMM_FILE, METADATA_FILE):
        if not (path / name).exists():
            raise FileNotFoundError(f"{path / name} missing. Run `python -m src.export` first.")
    metadata = json.loads((path / METADATA_FILE).read_text())
    if verify_checksum and metadata.get("model_file_sha256") != _sha256(path / MODEL_FILE):
        raise ValueError("model.joblib does not match the checksum in metadata.json")
    return Artifacts(model=joblib.load(path / MODEL_FILE),
                     features=json.loads((path / FEATURES_FILE).read_text()),
                     hmm=json.loads((path / HMM_FILE).read_text()),
                     metadata=metadata, path=path)


def check_artifacts(path: Path = ARTIFACTS_DIR) -> Artifacts:
    """Load the folder and run a prediction smoke test. Raises on any problem."""
    art = read_artifacts(path)
    if art.features != FEATURE_COLUMNS:
        raise ValueError("features.json does not match src.features.FEATURE_COLUMNS (order matters)")
    X = pd.DataFrame(np.random.default_rng(SEED).normal(size=(8, len(art.features))), columns=art.features)
    p = art.predict_proba(X)
    if p.shape != (8,) or not np.all((p >= 0) & (p <= 1)):
        raise ValueError("model.predict_proba returned invalid probabilities")
    return art


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Export the final model and inference artifacts")
    ap.add_argument("--model", default="xgboost")
    ap.add_argument("--source", choices=["auto", "tuned", "refit"], default="auto")
    ap.add_argument("--hmm-config", type=Path, default=None, help="JSON with tuned HMM decoder parameters")
    ap.add_argument("--out", type=Path, default=ARTIFACTS_DIR)
    ap.add_argument("--check", action="store_true", help="only verify an existing artifacts folder")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args(argv)

    if args.check:
        art = check_artifacts(args.out)
        print(f"OK: {art.path} | model={art.metadata['model']} ({art.metadata['source']}) | "
              f"{len(art.features)} features | hmm: {art.hmm.get('status')}")
        return
    out = export(args.model, args.source, args.out, args.results_dir, args.processed_dir, args.data_dir,
                 args.hmm_config)
    meta = json.loads((out / METADATA_FILE).read_text())
    print(f"exported {meta['model']} ({meta['source']}, {meta['model_file_mb']:.2f} MB) -> {out}")
    print(f"hmm.json: {json.loads((out / HMM_FILE).read_text())['status']}")


if __name__ == "__main__":
    main()
