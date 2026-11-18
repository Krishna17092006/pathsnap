# PathSnap: Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University (RR Campus).

PathSnap matches noisy GPS observations to a road network and reconstructs the route a vehicle actually drove. The pipeline:

1. Exploratory data analysis
2. Candidate generation and feature engineering
3. Comparison of five or more classifiers
4. Bayesian hyperparameter optimisation
5. HMM and Viterbi route reconstruction
6. A FastAPI backend (Laptop 1) and a Streamlit UI (Laptop 2)

- **[PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md):** the problem, data, evaluation and rules
- **[TEAM.md](TEAM.md):** who does what, shared interfaces, and each member's todo list

## Repository layout

```
pathsnap/
├── data/                  # raw dataset (train + student_test included; processed/ and eval/ are not committed)
│   ├── train/             # 80 trips
│   └── student_test/      # 10 trips
├── src/
│   ├── data_io.py         # raw file loaders (.track / .nodes / .arcs / .route)
│   ├── geo.py             # haversine, local metric projection, point-to-segment distance
│   ├── network.py         # road-network geometry, spatial index, connectivity, route helpers
│   ├── plot_style.py      # shared matplotlib style
│   ├── maps.py            # Folium map builders
│   ├── result_maps.py     # result maps + per-fix error breakdown (predictions vs reference route)
│   ├── eda.py             # exploratory data analysis
│   ├── config.py          # shared constants (seed, folds, candidate radius, label threshold)
│   ├── labels.py          # per-fix labels by order-constrained matching to the reference route
│   ├── candidates.py      # candidate arcs per fix + candidate/fix tables
│   ├── build_dataset.py   # builds data/processed/*.parquet and the radius analysis
│   ├── features.py        # geometric/kinematic features per (fix, candidate arc)
│   ├── evaluate.py        # GroupKFold CV, candidate- and fix-level metrics
│   ├── model_zoo.py       # model registry (name -> family, factory, config)
│   ├── train.py           # CV + student_test evaluation, writes results/models/
│   ├── compare.py         # model-comparison figures from the saved CSVs
│   ├── hmm.py             # HMM emissions/transitions and Viterbi decoder (Julia)
│   ├── route_metrics.py   # Top-1, route continuity, route precision/recall/F1
│   ├── pipeline.py        # load_artifacts, predict_trip (never reads .route), score_trip
│   ├── tune_hmm.py        # decoder grid search on train trips
│   ├── evaluate_hmm.py    # classifier-only vs classifier + HMM on student_test
│   ├── export.py          # writes artifacts/ (model, features, HMM params, metadata) for the API
│   ├── naive_bayes.py     # model 3 (KKR): Gaussian Naive Bayes, three-configuration study
│   └── label_noise_experiment.py  # data decision: train with / without the noisy-label trips
├── tests/                 # pytest suite (synthetic trips + real-trip API tests)
├── notebooks/             # naive_bayes.ipynb (+ .html), MLP_Training.ipynb, julia_hmm_viterbi_colab.ipynb (HMM evaluation)
├── api/                   # FastAPI backend (main.py, Laptop 1): serves src.pipeline.predict_trip
├── ui/                    # Streamlit evaluator UI (app.py) and a mock API for development (mock_api.py)
├── results/eda/           # EDA tables (CSV), figures (PNG), summary
├── results/candidates/    # label coverage, candidate miss rate vs radius, data decisions
├── results/models/        # per-model CV/student_test results, model_comparison.csv, figures
├── results/result_maps/   # per-method error breakdown (correct / wrong direction / wrong road)
├── results/hmm/xgboost/   # classifier-only vs HMM with the tuned XGBoost (results/hmm/: earlier MLP run)
├── maps/eda/              # interactive Folium maps (HTML)
├── maps/results/          # result maps per method (baseline/, xgboost_hmm/ cases where the HMM did not help)
└── docs/guidelines/       # the five official hackathon PDFs
```

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

### Data setup

The dataset is included in the repository, so cloning is enough. The trip folders sit directly under `data/`:

```
data/train/00000001/00000001.track
data/train/00000001/00000001.nodes
data/train/00000001/00000001.arcs
data/train/00000001/00000001.route
...
data/student_test/00000000/...
```

Files are stored byte-for-byte as provided (`.gitattributes` disables line-ending conversion). Evaluator trips are never committed: `data/eval/` is git-ignored.

Each trip has four headerless, whitespace-separated files:

| File | Columns | Meaning |
|---|---|---|
| `.track` | lon, lat, t | GPS observations, t in seconds (about 1 Hz) |
| `.nodes` | lon, lat | road-network nodes; row index = node id |
| `.arcs` | u, v | directed road segments; row index = arc id |
| `.route` | arc_id | reference route as a chronological list of arc ids |

## Running

All commands run from the repository root.

### Exploratory data analysis

```bash
python -m src.eda            # all train + student_test trips
python -m src.eda --quick    # 3 trips per partition (smoke test)
```

Outputs:

- `results/eda/*.csv`: per-trip inventory, missing values, GPS quality, network, route and alignment statistics; partition summary; data-quality log
- `results/eda/figures/*.png`: report figures
- `results/eda/eda_summary.md`: summary of the measured numbers
- `maps/eda/*.html`: interactive Folium maps (open in a browser)

### Labels and candidate table

```bash
python -m src.build_dataset            # about 1 minute; all train + student_test trips
python -m src.build_dataset --quick    # 3 trips per partition (smoke test)
```

Outputs:
- `data/processed/candidates_<partition>.parquet`: one row per (GPS fix, candidate arc within 50 m), with labels. Not committed.
- `data/processed/fixes_<partition>.parquet`: one row per GPS fix (matched route arc, label validity, candidate coverage). Not committed.
- `results/candidates/`: miss rate vs radius, per-trip label statistics, figures, and `dataset_summary.md` (data decisions)

The column definitions are in `TEAM.md` section 2.2.

**Noisy-label trips.** Train trips 00000002 and 00000033 are flagged as noisy in `dataset_summary.md`. To decide whether to drop them from training, run:

```bash
python -m src.label_noise_experiment
```

It trains `gaussian_nb` and `logreg` with and without those trips on the same CV folds, then writes `results/candidates/label_noise_experiment.md`.

### API backend (Laptop 1)

`api/main.py` serves the real pipeline from `src/pipeline.py`, with the exported artifacts (`artifacts/model.joblib` = tuned XGBoost, `features.json`, `hmm.json`, `metadata.json`). It does not re-implement map matching.

```bash
# demo: the 10 evaluator trips in data/eval/<trip_id>/<trip_id>.{track,nodes,arcs,route}
uvicorn api.main:app --host 0.0.0.0 --port 8000

# rehearsal with the student_test trips (bash / Windows cmd)
PATHSNAP_TRIPS_DIR=data/student_test uvicorn api.main:app --host 0.0.0.0 --port 8000
set PATHSNAP_TRIPS_DIR=data/student_test && uvicorn api.main:app --host 0.0.0.0 --port 8000
```

| Variable | Default | Meaning |
|---|---|---|
| `PATHSNAP_TRIPS_DIR` | `data/eval` | trip folders the API serves (relative paths resolve from the repo root) |
| `PATHSNAP_ARTIFACTS_DIR` | `artifacts` | exported model + decoder (`python -m src.export`) |
| `PATHSNAP_CORS_ORIGINS` | `*` | allowed UI origins, comma-separated |

At startup the API checks all four artifact files, verifies the model checksum, and loads them **once**. If anything is missing or invalid, startup fails with a clear message. There is no mock fallback.

| Endpoint | Response |
|---|---|
| `GET /health` | `{"status": "ok", "mode": "real", "model": "xgboost", "model_source": "tuned"}` |
| `GET /trips` | `{"trips": ["00000000", ...]}`: the 8-digit folders in `PATHSNAP_TRIPS_DIR` |
| `POST /predict/{trip_id}` | `trip_id`, `n_fixes`, `gps` ([lat, lon] per fix), `predicted_route` (list of [lat, lon] polylines, split where the route breaks), `predicted_arcs`, `metrics`, `timing_ms`, `mode: "real"` |
| `POST /evaluate/{trip_id}` | `metrics` with `top1_classifier`, `top1_hmm`, `route_continuity_clf/hmm`, `route_precision/recall/f1`, `n_fixes`, `n_valid_fixes` |

Details:
- **`/predict`** loads the trip **without opening its `.route` file** and calls `predict_trip`. Its `metrics` contain only values that need no reference route: `n_candidates`, `n_unmatched_fixes`, `n_route_arcs`, `route_continuity_clf/hmm`.
- **`/evaluate`** is the server-side accuracy check the evaluation requires. The prediction is still made without the route. The route is read only afterwards, by `score_trip`, and only aggregate numbers are returned, never the route or per-fix labels.
- **`timing_ms`:** `candidates`, `model` (features + classifier), `hmm`, `pipeline` and `total`.
- **Errors:** an unknown or invalid trip id returns 404, malformed trip files return 422, an inference failure returns 500 (traceback in the server log).
- **Logging:** every request is logged to the terminal with timestamp, trip id, number of fixes, candidates, HMM time and total time. Show this log to the evaluator.

**Network:**
- On Windows, allow inbound TCP 8000 for Python in the firewall (Windows Security → Firewall → Allow an app, or `netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000`).
- Find Laptop 1's IP with `ipconfig` (Windows) or `ipconfig getifaddr en0` (macOS).

Quick check from any machine:

```bash
curl http://<laptop1-ip>:8000/health
curl -X POST http://<laptop1-ip>:8000/predict/00000052
curl -X POST http://<laptop1-ip>:8000/evaluate/00000052
```

### Evaluator UI (Laptop 2)

```bash
streamlit run ui/app.py
```

Enter the API address in the sidebar, for example `http://192.168.1.20:8000` (Laptop 1's IP and the API port), or set it beforehand:

```bash
set PATHSNAP_API_URL=http://192.168.1.20:8000      # Windows cmd
export PATHSNAP_API_URL=http://192.168.1.20:8000   # bash
```

The UI lists the trips the API reports, sends the chosen trip id to `POST /predict/{trip_id}`, and shows the predicted route on a map with the scores and timing returned by the API. It never reads trip files or ground truth.

For a local test, use `http://127.0.0.1:<port>` rather than `localhost`: on Windows, `localhost` tries IPv6 first and adds about 2 s to every request.

**UI development without the real API:** `python -m ui.mock_api` starts a stand-in server on port 8765 that follows the same API contract. It performs no map matching: the "route" is the raw GPS path and all scores are empty. Every response is marked `mock`, and the UI shows a warning banner.

### Result maps (offline analysis)

```bash
python -m src.result_maps                       # nearest-road baseline on student_test
python -m src.result_maps --predictions results/predictions/<name>_student_test.parquet --name <name>
```

Each student_test trip gets one map (`maps/results/<name>/<trip_id>.html`) showing:
- GPS fixes, the nearby road network and the reference route
- classifier-only and HMM predictions
- red rings where a prediction is wrong

`results/result_maps/<name>_error_breakdown.csv` counts each outcome per trip:
- correct
- right road, wrong direction
- wrong road
- no prediction

The predictions file has one row per fix: `trip_id, fix_idx, arc_clf[, arc_hmm]`, with `-1` meaning no prediction. These maps use the reference route, so they are for development and the report, never for the evaluator UI.

### Features

```bash
python -m src.features    # writes data/processed/features_<partition>.parquet
```

`src/features.py` adds 19 model-input columns (`FEATURE_COLUMNS`) to the candidate table: distance (absolute and relative to the nearest candidate), heading vs arc direction with a speed-based confidence, speed and time gaps, arc shape and topology, and links to neighbouring fixes' nearest arcs. Features use only the GPS track and the road network, never `.route`, labels or ids. `add_features(cand_df, trip, net)` is the same function used at inference time.

### Evaluation framework

`src/evaluate.py` is shared by every model:
- `group_cv(factory, params, cand, fixes, FEATURE_COLUMNS)`: GroupKFold by trip on the train partition. Each model is an sklearn estimator or Pipeline, so preprocessing is fitted inside each training fold.
- Candidate-level metrics: accuracy, precision, recall, macro F1, confusion matrix, log loss.
- Fix-level metrics: Top-1 accuracy and candidate miss rate over the **fix table**, so labelled fixes without a correct candidate count as wrong. Plus classifier-only route continuity.
- Training time, inference time per 1,000 fixes, and train-fold macro F1 for over/underfitting checks.

### Model training and comparison

Register a model in `src/model_zoo.py` (see the example in its docstring), then:

```bash
python -m src.train --model baseline         # nearest-arc baseline on the same folds
python -m src.train --model <name>           # one model; repeat --model for several
python -m src.train --all                    # baseline + every registered model
python -m src.train --model <name> --quick   # 12 train trips, 3 folds, results/models/quick/ (smoke test)
python -m src.compare                        # figures from results/models/model_comparison.csv
```

Each run does GroupKFold CV by trip on the 80 train trips, refits on all of them, and evaluates once on student_test. Outputs:
- `results/models/<name>/config.json`: model config, feature list, seed, versions
- `results/models/<name>/cv_folds.csv`: per-fold metrics, including train-fold macro F1 for over/underfitting checks
- `results/models/<name>/student_test_metrics.json`
- `results/models/model_comparison.csv`: one row per model (re-running a model replaces its row)
- `results/models/figures/*.png`: CV macro F1, student_test Top-1 / macro F1, log loss, inference time, train vs CV

### Naive Bayes (model 3)

```bash
python -m src.naive_bayes            # three configurations: CV + student_test, about 2 minutes
python -m src.train --model gaussian_nb
```

`src/naive_bayes.py` compares three Gaussian Naive Bayes configurations that test the model's two assumptions:

| Config | What it does |
|---|---|
| `gnb_all` | raw features |
| `gnb_quantile` | quantile-normal transform (Gaussian assumption) |
| `gnb_decorrelated` | quantile transform + a correlation filter dropping \|r\| > 0.8, fitted inside each training fold (independence assumption) |

- The configuration with the highest mean CV Top-1 is registered in `src/model_zoo.py` as `gaussian_nb`.
- Results go to `results/models/naive_bayes/`: summary, per-fold metrics, student_test metrics, figures.
- `notebooks/naive_bayes.ipynb` walks through the selected model with all metrics. It's saved with outputs, and `naive_bayes.html` is a static copy. To re-run it, open it with `jupyter notebook notebooks/naive_bayes.ipynb` after `python -m src.features`.

### Export the final model (artifacts for the API)

```bash
python -m src.export                                   # xgboost: the Optuna-tuned model if results/tuning/xgboost/ exists, else a refit on all train trips
python -m src.export --model xgboost --source tuned    # insist on the tuned model
python -m src.export --hmm-config results/hmm/xgboost/best_decoder.json   # include tuned HMM decoder parameters
python -m src.export --check                           # verify artifacts/ loads and predicts
```

Writes `artifacts/model.joblib`, `features.json`, `hmm.json` and `metadata.json` (source, parameters, CV/student-test scores, SHA-256 of the model file, library versions). Only train trips are used for fitting. Until the decoder grid search exists, `hmm.json` holds default parameters marked `"status": "default (untuned)"`.

Load the folder in the API or pipeline with:

```python
from src.export import read_artifacts
art = read_artifacts("artifacts")          # model, features, hmm, metadata (checksum verified)
prob = art.predict_proba(candidate_rows)   # P(candidate is the true arc)
```

### HMM route reconstruction (model 6 owner: Julia)

```bash
python -m src.evaluate_hmm                   # classifier-only vs classifier + HMM on all 10 student_test trips
python -m src.tune_hmm --max-trips 3         # decoder grid (beta, gap, emission weight) on train trips; writes artifacts/hmm.json
python scripts/run_julia_hmm.py --skip-tune  # the evaluation, with a check of every output file
```

The HMM uses the classifier in `artifacts/model.joblib` (the tuned XGBoost). Results go to `results/hmm/xgboost/` and maps to `maps/results/xgboost_hmm/`. `results/hmm/xgboost/predictions.parquet` can be passed to `python -m src.result_maps --predictions`. `notebooks/julia_hmm_viterbi_colab.ipynb` runs the same evaluation locally or in Colab, saved with outputs. Details: `docs/README_JULIA_RUN.md`.

### Tests

```bash
python -m pytest tests -q
```

## Data-partition rules

- `train` (80 trips) is used for fitting, cross-validation and tuning.
- `student_test` (10 trips) is used for development-stage model comparison.
- Evaluator trips are used only by the deployed API during the final evaluation. They are never used for training, feature selection or tuning.
