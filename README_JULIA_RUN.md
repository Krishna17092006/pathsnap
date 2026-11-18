# Julia's PathSnap deliverables — what to run

This package contains Julia's HMM/Viterbi implementation, route metrics, prediction pipeline, MLP model and experiment outputs. It does not include the raw dataset; keep the dataset in `data/` locally and do not commit it to GitHub.

## Do I run `hmm.py` directly?

No. `src/hmm.py` contains the decoder implementation and is imported by the pipeline. Use the runner at the project root:

```bash
python run_julia_hmm.py
```

This performs two stages:

1. Tunes decoder parameters using TRAIN trips only and writes `artifacts/hmm.json` plus `results/hmm/decoder_grid.csv`.
2. Runs the end-to-end inference pipeline on all 10 `student_test` trips, with the reference route removed during prediction, then scores predictions separately and writes the comparison table, plots and non-improvement maps.

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
python run_julia_hmm.py --skip-tune
```

## Colab setup

1. Upload the original `pathsnap(1).zip` repository snapshot and the `Julia_PathSnap_HMM_Viterbi_Addon.zip` add-on package.
2. Open `notebooks/julia_hmm_viterbi_colab.ipynb` from the add-on package, upload both ZIPs when prompted, and run the cells from top to bottom.
3. The notebook overlays Julia's `src/` and `artifacts/` onto the extracted repository, installs the pinned requirements, then runs the runner.

## Files Julia owns

- `src/hmm.py` — directed-network HMM emissions, transitions, shortest paths and Viterbi decoder.
- `src/route_metrics.py` — pointwise Top-1, route continuity and sequence precision/recall/F1.
- `src/pipeline.py` — `load_artifacts`, `predict_trip` and `score_trip`; prediction does not read `trip.route`.
- `src/tune_hmm.py`, `src/evaluate_hmm.py`, `run_julia_hmm.py` — tuning and integration runs.
- `src/model_zoo.py`, `src/train.py`, `notebooks/julia_mlp_training.ipynb` — MLP registration/training.
- `artifacts/mlp_model.joblib`, `artifacts/features.json`, `artifacts/hmm.json`, `artifacts/mlp_metadata.json` — Julia's standalone inference assets. `mlp_model.joblib` does not replace Joshua's shared `model.joblib`.

## Outputs

- `results/hmm/decoder_grid.csv` — tested decoder configurations.
- `results/hmm/classifier_vs_hmm.csv` — one row per student-test trip.
- `results/hmm/hmm_non_improvements.csv` — cases where HMM Top-1 did not exceed classifier-only Top-1.
- `results/hmm/classifier_vs_hmm.png` — aggregate comparison plot.
- `maps/results/` — example Folium maps for cases where HMM did not improve Top-1.
- `results/models/mlp/` — MLP configuration, CV folds, student-test metrics, loss histories and curves.

## Important interpretation

HMM is not guaranteed to improve pointwise Top-1 for every trip. In the current 10-trip integration run it improved Top-1 on 2/10 trips, while mean route continuity increased from about 0.9437 to 0.9619 and mean route-level F1 increased from about 0.7084 to 0.7374. Keep both improvements and regressions in the report; do not claim that HMM always improves accuracy.

The decoder grid currently uses classifier-probability emissions from Julia's MLP artifact. Its small default tuning subset is a smoke-grid, not a final exhaustive search over all 80 training trips. Record the number of training trips used whenever reporting the selected parameters.
