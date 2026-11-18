# PathSnap: Team Plan and Context

This file is the single source of truth for **who does what, and how the pieces fit together**. It is written to be self-contained: give it, together with [`PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md), to anyone helping you (or any tool you use), and say which member you are. Everything needed to start your part is here: project state, existing code, shared interfaces, and your todo list.

| Member | Name | Owns |
|---|---|---|
| **Member 1** | **KKR** | Data, EDA, labels, candidate generation, result maps, Streamlit UI, Gaussian Naive Bayes |
| **Member 2** | **Krishna Jha** | Feature engineering, evaluation framework, Logistic Regression, KNN, model comparison |
| **Member 3** | **Joshua** | Random Forest, XGBoost, Bayesian optimisation, artifact export, FastAPI backend |
| **Member 4** | **Julia** | HMM + Viterbi, route metrics, end-to-end `predict_trip()` pipeline, integration, MLP |

### Model ownership

All 6 models are compared together in **one** table (`results/models/model_comparison.csv`) using the same features, the same GroupKFold and the same metrics. The split below is only about who builds each model. The two members with one model (KKR, Julia) carry the heaviest pipeline stages.

| # | Model | Family | Owner |
|---|---|---|---|
| 0 | Nearest-arc baseline | Rule (no training, reference row) | Krishna Jha |
| 1 | Logistic Regression | Linear / probabilistic | Krishna Jha |
| 2 | KNN | Instance-based | Krishna Jha |
| 3 | Gaussian Naive Bayes | Probabilistic | KKR |
| 4 | Random Forest | Bagging ensemble | Joshua |
| 5 | XGBoost | Boosting | Joshua |
| 6 | MLP | Neural network | Julia |

Each member owns one stage of the pipeline end to end: code, experiments, and the matching report sections. **In the viva anyone can be asked about any part**, so every member must also understand the others' work (see "Knowledge sharing").

---

## 1. Project state (what already exists)

Repository: `https://github.com/Krishna17092006/pathsnap` (private; collaborators: Kr1491, juliajikku, jjosh2005; it must be made public at submission).

```
pathsnap/
├── PROBLEM_STATEMENT.md   # what the problem is: read first
├── TEAM.md                # this file
├── README.md              # setup + run instructions
├── requirements.txt
├── data/                  # NOT in git: unzip "Student folder.zip" here
│   ├── train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
│   └── student_test/<trip_id>/...
├── src/
│   ├── data_io.py         # loaders          (done)
│   ├── geo.py             # geometry helpers (done)
│   ├── network.py         # road network     (done)
│   ├── maps.py            # Folium maps      (done, to be extended by KKR)
│   ├── plot_style.py      # figure style     (done)
│   └── eda.py             # EDA              (done)
├── results/eda/           # EDA CSVs, figures, eda_summary.md
└── maps/eda/              # EDA Folium maps
```

**Done:** data extraction, loaders, geometry utilities, road-network spatial index, full EDA (10 figures, 14 Folium maps, data-quality log, summary). Run it with `python -m src.eda`.

### 1.1 Existing code you should reuse (do not rewrite)

```python
from src.data_io import PARTITIONS, list_trips, load_trip, iter_trips
trip = load_trip("00000007", "student_test")
trip.track   # DataFrame: lon, lat, t        (one row per GPS fix; fix_idx = row index)
trip.nodes   # DataFrame: lon, lat           (node_id = row index)
trip.arcs    # DataFrame: u, v               (arc_id = row index, directed u -> v)
trip.route   # np.ndarray of arc ids in driving order (None if absent)

from src.geo import LocalProjection, haversine_m, bearing_deg, point_segment_distance
proj = LocalProjection.from_points(trip.track.lon, trip.track.lat)   # per-trip metric frame
x, y = proj.to_xy(lon, lat)                                           # metres
haversine_m(lon1, lat1, lon2, lat2)                                   # metres
bearing_deg(lon1, lat1, lon2, lat2)                                   # 0-360, clockwise from north
dist, t = point_segment_distance(px, py, ax, ay, bx, by)              # t in [0,1] along segment

from src.network import RoadNetwork
net = RoadNetwork(trip.nodes, trip.arcs, proj)
net.u, net.v                     # arc endpoints (node ids)
net.ax, net.ay, net.bx, net.by   # arc endpoints in metres
net.length_m                     # arc lengths (haversine)
net.arcs_within(px, py, 50)      # DataFrame(point_idx, arc_id, dist_m, t): every arc within 50 m of each point
net.distance_to_arcs(px, py, arc_ids)   # nearest distance from each point to a (small) set of arcs
net.route_breaks(route)          # indices where consecutive route arcs are not connected
net.route_polylines(route)       # [(lat, lon), ...] lines for drawing
net.connectivity_stats()         # components, isolated nodes, ...

from src.maps import trip_map, add_legend      # Folium helpers
from src import plot_style as ps               # ps.apply_style(), ps.save(fig, path), colours
```

### 1.2 Key data facts (from `results/eda/eda_summary.md`)

- 80 train trips (208,417 fixes) and 10 student-test trips (19,160 fixes). Evaluator trips are not provided.
- **Each trip has its own road network.** Arc ids mean nothing across trips, so **never use arc_id or trip_id as a feature.**
- Median fix-to-reference-route distance is ~5 m. Student test: 99.9% of fixes within 50 m of the route.
- **Train trips 00000054 and 00000033:** the reference route covers only part of the drive (GPS path 5–7× longer). Most of their fixes are more than 100 m from the route and must not get labels.
- 11 train trips have gaps longer than 300 s (max 153,064 s in 00000090).
- Median candidate arcs per fix: 4 within 25 m, 8 within 50 m, ~20 within 100 m.
- Naive "nearest arc" is correct for ~89% of fixes (median per trip). **This is the baseline to beat.**
- Reference routes have occasional breaks, U-turns and revisited arcs. Networks have duplicate arcs, self-loops and zero-length arcs, so code must not crash on them.

---

## 2. Shared interfaces (agree on these; do not change without telling the team)

### 2.1 Shared constants: `src/config.py` (created by KKR)

```python
SEED = 42
N_FOLDS = 5                     # GroupKFold by trip on the 80 train trips
CANDIDATE_RADIUS_M = 50.0       # candidate search radius (final value chosen from miss-rate analysis)
LABEL_MAX_DIST_M = 50.0         # fixes farther than this from their matched route arc get no label
PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = "artifacts"
```

### 2.2 Candidate table (produced by KKR, used by everyone)

Built by `python -m src.build_dataset`. It writes `data/processed/candidates_train.parquet` and `data/processed/candidates_student_test.parquet`. These are not in git; everyone regenerates them with the script.

One row per (GPS fix, candidate arc):

| Column | Type | Meaning |
|---|---|---|
| `trip_id` | str | e.g. `"00000007"` |
| `partition` | str | `train` / `student_test` |
| `fix_idx` | int | row index of the fix in `.track` |
| `t` | float | fix timestamp (s) |
| `lon`, `lat` | float | fix position |
| `arc_id` | int | candidate arc (only meaningful within the trip) |
| `dist_m` | float | fix-to-arc distance (m) |
| `t_along` | float | position of the fix's projection along the arc, 0 = start node, 1 = end node |
| `cand_rank` | int | rank by distance within the fix (0 = nearest) |
| `n_cand` | int | number of candidates for this fix |
| `true_arc` | int | reference arc for the fix (-1 if no reliable label) |
| `fix_valid` | bool | True if the fix has a reliable label (used for training and metrics) |
| `label` | int | 1 if `arc_id == true_arc` else 0 |

Rules: rows with `fix_valid == False` are **excluded from training and metrics** but kept for HMM inference. A valid fix whose `true_arc` is not among its candidates is a **candidate miss**.

### 2.3 Features (produced by Krishna Jha)

`src/features.py` → `add_features(cand_df, trip, net) -> DataFrame`. It adds feature columns and keeps all candidate-table columns. The list of model-input columns is saved in `artifacts/features.json`. **Features may use only the GPS track and the road network, never `.route`, `true_arc` or `label`.**

### 2.4 Models (all members; see "Model ownership")

- Every model is an sklearn-compatible estimator (or Pipeline) with `fit(X, y)` and `predict_proba(X)[:, 1]` = P(candidate is the true arc).
- Models are registered in `src/model_zoo.py` as `name -> (family, factory_function, config_dict)`. Each member adds the entries for their own models.
- `python -m src.train --model <name>` runs group CV on train plus evaluation on student_test and writes:
  - `results/models/<name>/config.json`
  - `results/models/<name>/cv_folds.csv` (per-fold metrics)
  - `results/models/<name>/student_test_metrics.json`
  - one appended row in `results/models/model_comparison.csv`

### 2.5 Artifacts for inference (Joshua exports)

```
artifacts/
├── model.joblib         # final tuned model (includes the scaler if it is a Pipeline)
├── features.json        # ordered list of feature columns
├── hmm.json             # decoder parameters (from Julia)
└── metadata.json        # model name, training date, CV scores, seeds
```

### 2.6 Prediction pipeline (Julia)

```python
from src.pipeline import load_artifacts, predict_trip, score_trip
art = load_artifacts("artifacts")
pred = predict_trip(trip, art)      # MUST NOT read trip.route
# pred.fixes: DataFrame(fix_idx, t, lon, lat, arc_clf, prob_clf, arc_hmm)
# pred.route_arcs: list[int]        (HMM route, consecutive duplicates removed)
# pred.route_polyline: [[lat, lon], ...]
metrics = score_trip(pred, trip)    # server-side only: uses trip.route
# {"top1_classifier": ..., "top1_hmm": ..., "route_continuity_clf": ..., "route_continuity_hmm": ...,
#  "route_precision": ..., "route_recall": ..., "n_fixes": ..., "n_valid_fixes": ...}
```

### 2.7 API contract (Joshua builds, KKR's UI consumes)

- `GET /health` → `{"status": "ok"}`
- `GET /trips` → `{"trips": ["00000015", ...]}`: trip folders found in `PATHSNAP_TRIPS_DIR`. It defaults to `data/eval` and is set to `data/student_test` for rehearsals.
- `POST /predict/{trip_id}` → returns:

```json
{
  "trip_id": "00000015",
  "n_fixes": 1234,
  "gps": [[lat, lon], ...],
  "predicted_route": [[lat, lon], ...],
  "predicted_arcs": [12, 13, 57, ...],
  "metrics": {"top1_classifier": 0.91, "top1_hmm": 0.95, "route_continuity_hmm": 0.99, "...": "..."},
  "timing_ms": {"candidates": 120, "features": 80, "model": 40, "hmm": 300, "total": 560}
}
```

**The response must never contain the reference route or per-fix true labels.** Only aggregate metrics leave the server. Every request is logged to the console (trip id, n_fixes, metrics, timing), and the evaluator will look at this log.

---

## 3. Todo lists

### Member 1: KKR (data, labels, candidates, maps, UI, Naive Bayes)

Branch: `kkr/data` (and later `kkr/ui`)

- [x] Extract data, loaders (`data_io.py`), geometry (`geo.py`), road network (`network.py`), map helpers (`maps.py`)
- [x] EDA script, figures, maps, data-quality log (`eda.py`)
- [ ] Re-run EDA with latest changes and commit outputs; write a short "data decisions" note (which fixes are dropped and why)
- [ ] Create `src/config.py` (section 2.1)
- [ ] **`src/labels.py`: per-fix labels by order-constrained matching to the reference route.** Nearest-arc labelling is wrong where the route revisits or doubles back, so use dynamic programming over route positions:
  - `d[i, j]` = distance from fix `i` to route arc `route[j]`
  - `cost[i, j] = d[i, j] + min(cost[i-1, j'] for j' <= j)` (position along the route never decreases)
  - Backtrack to get `route_pos[i]`, then `true_arc[i] = route[route_pos[i]]`
  - Set `fix_valid = d[i, route_pos[i]] <= LABEL_MAX_DIST_M`. This automatically drops the unmatched parts of trips 00000054 and 00000033.
  - Vectorise row by row with `np.minimum.accumulate`. The largest trip is 19k fixes × 2.3k route arcs, which fits in memory if processed in chunks.
- [ ] **`src/candidates.py`:** `build_candidates(trip, net, radius)` using `net.arcs_within`. Add `cand_rank` and `n_cand`, join labels, and return the section 2.2 schema.
- [ ] **Candidate miss-rate analysis** for R = 25 / 50 / 75 / 100 m on train and student_test. Pick R and justify it (miss rate vs table size). Save `results/candidates/miss_rate.csv` and a plot.
- [ ] **`src/build_dataset.py`:** writes both parquet files. Add `pyarrow` to requirements.
- [ ] **Hand off to the team ASAP.** Everyone depends on the candidate table.
- [ ] **Result maps:** extend `src/maps.py` with `result_map(trip, net, pred)` showing GPS, nearby network, reference route, classifier-only prediction and HMM prediction, with layer control and a legend. Generate maps for all 10 student-test trips → `maps/results/`.
- [ ] **Streamlit UI** (`ui/app.py`, Laptop 2):
  - API URL from a sidebar text box (default from env `PATHSNAP_API_URL`, e.g. `http://<laptop1-ip>:8000`)
  - `GET /trips` fills a selectbox; a "Run" button calls `POST /predict/{trip_id}`
  - Show metrics with `st.metric`, timing, and a Folium map (GPS + predicted route) via `streamlit-folium`
  - Show a clear error if the API is unreachable
  - **Never** load data files or ground truth in the UI. Everything comes from the API.
  - Build it against a mocked JSON response first (section 2.7), then switch to the real API.
- [ ] **Model 3: Gaussian Naive Bayes** (probabilistic). Add it to `src/model_zoo.py` and run it with `src/train.py`. Note the independence assumption vs correlated features (`dist_m`, `dist_minus_min`, `dist_ratio` and `cand_rank` all describe distance) and its effect on log loss.
- [ ] Report: **§2 EDA**, **§3.2 subsection for Naive Bayes**, **§6 Folium maps**, UI screenshots in **§7**, and final assembly of the PDF.

**Done when:** `python -m src.build_dataset` produces both parquet files with the exact schema, the miss rate is reported, and the UI works against the real API on two laptops.

### Member 2: Krishna Jha (features, evaluation framework, Logistic Regression, KNN)

Branch: `krishna-jha/features-models`

- [ ] **`src/features.py`:** `add_features(cand_df, trip, net)`. Suggested features (justify each in the report):
  - `dist_m`, `cand_rank`, `n_cand`, `dist_minus_min` (distance minus the fix's nearest-candidate distance), `dist_ratio`
  - `heading_diff_deg`: angle between the vehicle heading (bearing from fix i-1 to i+1, only when moved more than ~3 m, else NaN → fill) and the arc bearing (u → v). This separates the two directions of a two-way road.
  - `cos_heading_diff`
  - `speed_mps` (from neighbouring fixes), `dt_prev`, `dt_next` (gaps)
  - `arc_length_m`, `t_along`, `is_two_way` (reverse arc exists), `out_degree_v`, `in_degree_u`
  - Optional context: whether this arc (or a connected arc) was the nearest candidate at the previous/next fix. Allowed: no labels involved.
  - **Never:** `arc_id`, `trip_id`, `true_arc`, `label`, anything from `.route`.
- [ ] **`src/evaluate.py`: the evaluation framework everyone uses:**
  - `group_cv(model_factory, df, n_folds=5)` with `GroupKFold` on `trip_id` (train only). Preprocessing is fitted inside each fold (use sklearn `Pipeline`).
  - Candidate-level metrics: accuracy, precision, recall, **macro F1**, confusion matrix, **log loss**.
  - Fix-level metrics: **Top-1 accuracy** (argmax probability per valid fix == `true_arc`; a candidate miss counts as wrong) and **candidate miss rate**.
  - Training time and inference time (per 1,000 fixes).
  - Write the result files listed in section 2.4.
- [ ] **`src/model_zoo.py` + `src/train.py`** (section 2.4)
- [ ] **Models 1–2:**
  1. **Logistic Regression** (linear/probabilistic): standardised features, `class_weight` balanced vs none
  2. **KNN** (instance-based): scaling matters. Train on a subsample if slow and say so.
- [ ] Include the **naive nearest-arc baseline** as a reference row (Top-1 only, no training).
- [ ] **Model comparison:** `results/models/model_comparison.csv` with all 6 models (from every member), and plots in `results/models/figures/`:
  - CV macro F1 per model (with fold spread)
  - Student-test Top-1 and macro F1
  - Log loss
  - Inference time
- [ ] Report: **§3.1 comparison table**, **§3.2 subsections for Logistic Regression and KNN**, **§3.3 comparison plots**, and the feature-engineering part of §2.

**Done when:** `python -m src.train --model logreg` (and `knn`) runs end to end and writes all result files, and the comparison plots regenerate from the CSV.

### Member 3: Joshua (Random Forest, XGBoost, Bayesian optimisation, export, API)

Branch: `joshua/models-tuning-api`

- [ ] **Models 4–5** (add to `src/model_zoo.py`, run with Krishna Jha's `src/train.py`):
  4. **Random Forest** (bagging ensemble)
  5. **XGBoost** (boosting; already installed)
- [ ] **Choose the final model** from the comparison:
  - Primary metric: group-CV macro F1 or Top-1. Decide with Krishna Jha and write why.
  - Also consider probability quality (log loss matters because the HMM uses the probabilities) and inference speed.
- [ ] **`src/tune.py`: Bayesian optimisation with Optuna** (add `optuna` to requirements):
  - TPE sampler, `seed=42`, 30–50 trials
  - Objective: mean group-CV score on the 80 train trips only
  - Justified search space. For XGBoost, for example: `n_estimators`, `learning_rate` (log), `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_lambda` (log).
  - Save every trial to `results/tuning/optuna_trials.csv`, plus optimisation-history and parameter-importance plots
  - Refit on all train data with the best parameters
  - Evaluate on student_test and produce a **tuned vs untuned** table: CV macro F1, student-test Top-1, macro F1, log loss, inference time
  - Record compute time
- [ ] **`src/export.py`:** writes `artifacts/` (section 2.5)
- [ ] **FastAPI backend** `api/main.py` (Laptop 1; add `fastapi`, `uvicorn` to requirements):
  - Endpoints as in section 2.7
  - Load artifacts once at startup; calls Julia's `predict_trip` and `score_trip`
  - Console logging for every request (timestamp, trip id, n_fixes, metrics, total ms)
  - Run with `uvicorn api.main:app --host 0.0.0.0 --port 8000`. Allow CORS. Document the Windows firewall step for the port.
  - Start with a stub that returns a fixed fake response so KKR can build the UI. Replace it with the real pipeline as soon as `predict_trip` exists.
- [ ] README: deployment section (how to run the API on Laptop 1 and the UI on Laptop 2, and where to put the evaluator trips: `data/eval/<trip_id>/...`)
- [ ] Report: **§3.2 subsections for Random Forest and XGBoost**, **§4 Bayesian optimisation**, API screenshots in **§7**.

**Done when:** `artifacts/` loads in a fresh process, `uvicorn` serves all endpoints, and the tuned vs untuned table exists.

### Member 4: Julia (HMM + Viterbi, route metrics, pipeline, integration, MLP)

Branch: `julia/hmm-pipeline`

- [ ] **`src/hmm.py`:**
  - **Hidden states:** the candidate arcs at each fix (from the candidate table / `build_candidates`).
  - **Emission:** `log P(fix | arc)` = log of the classifier probability (clip to ≥ 1e-6). Before models exist, use a Gaussian on distance: `-0.5 * (dist_m / sigma)^2` with sigma ≈ 5–10 m, so you are never blocked.
  - **Transition** from arc `a` (fix t) to arc `b` (fix t+1):
    - Route distance along the network:
      - Same arc moving forward: `(t_along_b - t_along_a) * len_a`
      - Otherwise: remaining part of `a` + shortest path from `v[a]` to `u[b]` + `t_along_b * len_b`
    - Compare it with the haversine distance between the two fixes: `log P = -|d_route - d_gps| / beta` (Newson & Krumm 2009). Tune beta.
    - Arcs are directed, so one-way streets are respected automatically.
  - **Shortest paths:** `scipy.sparse.csgraph.dijkstra` with `limit` = max plausible travel distance (e.g. `max(200 m, 50 m/s * dt)`). Only run it from the end nodes of current candidates, and cache results per trip.
  - **Viterbi** in log space with back-pointers.
  - **Edge cases (must handle and explain in the report):**
    - Fix with no candidates: skip it (carry the previous state) and mark it unmatched
    - All transitions impossible (disconnected): break the chain and restart Viterbi from the next fix
    - GPS gap larger than G seconds (e.g. 60 s): split the trip into independent segments
    - Very long trips (19k fixes): process segment by segment
- [ ] **Route metrics** (`src/route_metrics.py`). Define and document:
  - **Pointwise accuracy:** Top-1 for classifier-only and for HMM, on valid fixes
  - **Route continuity:** share of consecutive fix pairs whose predicted arcs are identical or directly connected (`v[a] == u[b]`)
  - **Sequence-level:** precision/recall/F1 of predicted route arcs vs reference route arcs (optionally length-weighted)
- [ ] **Decoder tuning:** grid over beta, (sigma), gap threshold and emission weighting on train (or train + student_test). **Never on evaluator data.** Save `results/hmm/decoder_grid.csv`.
- [ ] **Classifier-only vs classifier + HMM comparison** on student_test: table + plots in `results/hmm/`. Find and show cases where the HMM **did not** help, with a map from KKR's `result_map`.
- [ ] **`src/pipeline.py`:** `load_artifacts`, `predict_trip`, `score_trip` (section 2.6). `predict_trip` must work on a trip **without** a `.route` file.
- [ ] **Integration:** run `predict_trip` on all 10 student-test trips. Check timing (the demo needs a response in a few seconds) and robustness (gaps, no candidates, huge networks).
- [ ] **Model 6: MLP** (neural network). Add it to `src/model_zoo.py` and run it with `src/train.py`. Train epoch by epoch (`partial_fit` loop or PyTorch) and record **train and validation log loss per epoch**. This is the only model allowed to have loss curves. Plot them.
- [ ] Report: **§3.2 subsection for the MLP** (with loss curves), **§5 HMM and Viterbi** (all six required questions), plus integration notes.

**Done when:** `predict_trip` runs on every student-test trip without touching `.route`, and the classifier-only vs HMM table exists.

---

## 4. Order of work and handoffs

```
KKR: labels + candidates ──► Krishna Jha: features + evaluation ──► All members: 6 models ──► Joshua: tuned model + export
                                                                                                          │
Julia: HMM (distance-only emissions first) ─────────────────────► swap in classifier probabilities ◄─────┘
                                                                                │
                                                       Julia: predict_trip() ──► Joshua: API ──► KKR: UI ──► two-laptop rehearsal
```

**Nobody waits:**

| Member | While blocked, do this |
|---|---|
| KKR | Labels + candidates first (this is the critical path), then the UI against a mocked API response |
| Krishna Jha | Build `evaluate.py` and `train.py` on a small dummy candidate table with the section 2.2 schema |
| Joshua | Set up the Optuna script and the API stub (fixed fake response) |
| Julia | Build the HMM and Viterbi with distance-only emissions, using `net.arcs_within` directly |

### Milestones

1. **First-phase evaluation (before lunch):** EDA complete; labels + candidates ready with the miss rate; evaluation framework working; at least one baseline model result; HMM prototype.
2. **Models:** all 6 models trained and compared; final model chosen.
3. **Tuning + HMM:** tuned model exported; HMM uses classifier probabilities; classifier-only vs HMM results.
4. **System:** API + UI working on two laptops with student-test trips.
5. **Submission:**
   - Report PDF assembled from everyone's sections
   - README complete
   - Repo made public
   - Check that every number in the report regenerates from the repo

---

## 5. Working rules

- **Git:** everyone commits **their own work from their own GitHub account**. Set your identity in your clone to match your GitHub account:

  ```bash
  git config user.name "<your name>"
  git config user.email "<your GitHub email>"
  ```

  Work on your own branch, open a pull request into `main`, and get one teammate to review it. Do not commit `data/`, large intermediate files or secrets.
- **Setup:** clone the repo, then `pip install -r requirements.txt`, then unzip `Student folder.zip` into `data/` (see README), then run `python -m src.eda --quick` to check that it works.
- **Reproducibility:** fixed seeds (`SEED = 42`). Every experiment writes its config and results to `results/`. Plots are generated from saved CSVs, never by hand.
- **Leakage:**
  - Evaluator data never used for anything but the final API run
  - Preprocessing fitted on train (inside CV folds)
  - GroupKFold by trip
  - Features never touch `.route`
- **Honesty:** report real numbers only. No artificial training curves. N/A with a reason where a metric does not apply.
- **Independence:** nothing is shared with other teams.
- **AI-use disclosure:** the report must include the disclosure the event policy requires.

## 6. Knowledge sharing (viva preparation)

Each member gives the others a 10-minute walkthrough of their module covering:
- what it does
- the key design choices and why
- the results
- one thing that did not work

| Member | Must be able to explain |
|---|---|
| KKR | Dataset structure, why labels need order-constrained matching, how the candidate radius was chosen, how the UI talks to the API, Naive Bayes and why its independence assumption hurts here |
| Krishna Jha | Every feature and why it helps, GroupKFold and why random splits leak, macro F1 vs Top-1, log loss, Logistic Regression and KNN |
| Joshua | Random Forest vs XGBoost, why the final model was chosen, Bayesian optimisation (TPE, search space, objective), the API flow and logging |
| Julia | HMM states, emissions and transitions, the Viterbi recurrence, gap and no-candidate handling, when the HMM helped and when it did not, the MLP and its loss curves |

Everyone should be able to answer every row, not just their own.
