# Julia's PathSnap deliverables — what to run

Julia's part of PathSnap: the HMM/Viterbi decoder, route metrics, the end-to-end prediction pipeline, the MLP model and the HMM experiment outputs. The HMM uses the classifier in `artifacts/model.joblib`, which is the Optuna-tuned XGBoost (`artifacts/metadata.json`).

## Do I run `hmm.py` directly?

No. `src/hmm.py` contains the decoder implementation and is imported by the pipeline. Use the runner at the project root:

```bash
python scripts/run_julia_hmm.py
```

This performs two stages:

1. Tunes decoder parameters using TRAIN trips only and writes `artifacts/hmm.json` plus `results/hmm/xgboost/decoder_grid.csv`.
2. Runs the end-to-end inference pipeline on all 10 `student_test` trips, with the reference route removed during prediction, then scores predictions separately and writes the comparison table, summary, per-fix predictions, plot and non-improvement maps to `results/hmm/xgboost/` and `maps/results/xgboost_hmm/`.

The default tuning uses two training trips as a fast reproducible smoke-grid. Increase the number of tuning trips with `--tune-trips 5` or `--tune-trips 10`; this can be much slower. For the full 80-trip tuning run use `--tune-trips 80` only when you have time/resources.

Other commands:

```bash
# Run HMM unit tests
python -m pytest tests/test_hmm.py -q

# Tune without running student-test evaluation
python -m src.tune_hmm --max-trips 2

# Run evaluation on all 10 student-test trips using the current artifacts/hmm.json
python -m src.evaluate_hmm

# Quick smoke test (2 student-test trips)
python -m src.evaluate_hmm --quick

# Skip tuning and use the already saved decoder parameters
python scripts/run_julia_hmm.py --skip-tune
```

## Notebook (local or Colab)

`notebooks/julia_hmm_viterbi_colab.ipynb` runs the same evaluation and is saved with its outputs.

- **Local:** from the project root, `jupyter notebook notebooks/julia_hmm_viterbi_colab.ipynb`, then run all cells.
- **Colab:** upload the notebook and choose Runtime → Run all. It clones this repository (the dataset is in it), installs `requirements.txt` and runs the evaluation. No zip uploads are needed, and nothing is copied over the repository code.

It checks that `artifacts/model.joblib` is the tuned XGBoost (not a Git LFS pointer), runs `python -m src.evaluate_hmm`, shows the per-trip table and summary, and runs the HMM and classifier-compatibility tests. Set `RETUNE_HMM = True` in the first cell to re-tune the decoder on TRAIN trips first.

## Files Julia owns

- `src/hmm.py` — directed-network HMM emissions, transitions, shortest paths and Viterbi decoder.
- `src/route_metrics.py` — pointwise Top-1, route continuity and sequence precision/recall/F1.
- `src/pipeline.py` — `load_artifacts`, `predict_trip` and `score_trip`; prediction does not read `trip.route`.
- `src/tune_hmm.py`, `src/evaluate_hmm.py`, `scripts/run_julia_hmm.py`, `notebooks/julia_hmm_viterbi_colab.ipynb` — tuning and integration runs.
- `src/model_zoo.py`, `src/train.py`, `notebooks/MLP_Training.ipynb` — MLP registration/training.
- `artifacts/mlp_model.joblib`, `artifacts/features.json`, `artifacts/hmm.json`, `artifacts/mlp_metadata.json` — Julia's standalone inference assets. `mlp_model.joblib` does not replace Joshua's shared `model.joblib`.

## Outputs

- `results/hmm/xgboost/classifier_vs_hmm.csv` — one row per student-test trip (Top-1, continuity, route precision/recall/F1, timings).
- `results/hmm/xgboost/integration_summary.json` — means, trips improved, timings and decoder settings, computed from the CSV of the same run.
- `results/hmm/xgboost/predictions.parquet` — per-fix `trip_id, fix_idx, arc_clf, arc_hmm`; input for `python -m src.result_maps --predictions`.
- `results/hmm/xgboost/hmm_non_improvements.csv` — trips where HMM Top-1 did not exceed classifier-only Top-1.
- `results/hmm/xgboost/classifier_vs_hmm.png` — aggregate comparison plot.
- `maps/results/xgboost_hmm/` — Folium maps of the two trips where the HMM helped least.
- `results/hmm/` (top level) and `maps/results/*_hmm_no_improvement.html` — the earlier run with MLP emissions, kept for reference; it also holds the decoder grid that chose the current `artifacts/hmm.json`.
- `results/models/mlp/` — MLP configuration, CV folds, student-test metrics, loss histories and curves.

## Important interpretation

The HMM is not guaranteed to improve pointwise Top-1 on every trip. With the tuned XGBoost emissions (`results/hmm/xgboost/integration_summary.json`):

| Mean over the 10 student-test trips | Classifier only | Classifier + HMM | Trips improved |
|---|---|---|---|
| Top-1 accuracy | 0.8845 | 0.8665 | 2 / 10 |
| Route continuity | 0.9443 | 0.9633 | 10 / 10 |
| Route F1 | 0.7183 | 0.7434 | 8 / 10 |

The HMM trades a little pointwise accuracy for a connected, drivable route. Keep both the improvements and the regressions in the report; do not claim that the HMM always improves accuracy. Mean inference time was 3.4 s per trip (max 6.9 s, trip 00000056 with 3,958 fixes), most of it in Viterbi decoding; timings depend on the machine.

The decoder settings in `artifacts/hmm.json` (beta = 100, gap 30 s, emission weight 1) come from a small grid on two TRAIN trips run with MLP emissions (`results/hmm/decoder_grid.csv`). Re-tuning with XGBoost emissions (`python -m src.tune_hmm --max-trips 3`, or `RETUNE_HMM = True` in the notebook) is optional; record the number of training trips used whenever reporting the selected parameters.
