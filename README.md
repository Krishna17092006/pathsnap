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
│   └── compare.py         # model-comparison figures from the saved CSVs
├── tests/                 # pytest suite on synthetic trips
├── results/eda/           # EDA tables (CSV), figures (PNG), summary
├── results/candidates/    # label coverage, candidate miss rate vs radius, data decisions
├── results/models/        # per-model CV/student_test results, model_comparison.csv, figures
├── results/result_maps/   # per-method error breakdown (correct / wrong direction / wrong road)
├── maps/eda/              # interactive Folium maps (HTML)
├── maps/results/          # result maps per method (nearest-road baseline now; classifier + HMM later)
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

### Tests

```bash
python -m pytest tests -q
```

## Data-partition rules

- `train` (80 trips) is used for fitting, cross-validation and tuning.
- `student_test` (10 trips) is used for development-stage model comparison.
- Evaluator trips are used only by the deployed API during the final evaluation. They are never used for training, feature selection or tuning.
