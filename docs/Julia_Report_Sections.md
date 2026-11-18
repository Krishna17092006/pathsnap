# Julia — report contribution (§3.2 MLP, §5 HMM and Viterbi, integration notes)

Every number below is read from a file in the repository; the source file is named next to each group of numbers.

## §3.2 Model 6: Multilayer Perceptron (MLP)

### Why it is suitable

The task is binary: for each GPS fix and each candidate directed arc within 50 m, is this the reference arc? The 19 engineered features (`artifacts/features.json`) mix distance, heading, speed and network-context terms that interact non-linearly. For example, a small heading difference matters more when the vehicle is moving than when it is stationary. An MLP learns such interactions without hand-built crossed features. It also outputs a probability through a sigmoid, which the HMM uses directly as its emission score (§5.2). It is the only model in the comparison that trains in epochs, so it is the only one with loss curves.

### Configuration

- Architecture: two fully connected hidden layers (128 and 64 neurons), ReLU, sigmoid output.
- Optimiser: Adam, learning rate 0.001, batch size 8,192, L2 penalty `alpha = 0.0001`, 20 epochs, seed 42.
- Preprocessing: median imputation and `StandardScaler`, both fitted on the training fold only.
- Training data: a 500,000-row sample of the training-fold candidate rows per fold. The full table is too large for 20 epochs in the available time.
- Evaluation: five-fold `GroupKFold` grouped by `trip_id` (64 training and 16 validation trips per fold), so no trip is split between training and validation.
- Training loop: `partial_fit` once per epoch, recording train and validation log loss after each epoch.
- Source: `results/models/mlp/config.json`.

### Loss curves and over/underfitting evidence

Per-fold curves are in `results/models/mlp/cv_loss_curves.png` (data: `loss_history_cv.csv`). The final-fit curve is in `final_training_loss.png` (data: `final_loss_history.csv`).

| Fold | Train log loss, epoch 1 → 20 | Validation log loss, epoch 1 → 20 |
|---|---|---|
| 0 | 0.1341 → 0.0854 | 0.1063 → 0.0628 |
| 1 | 0.1236 → 0.0838 | 0.1110 → 0.0624 |
| 2 | 0.1246 → 0.0764 | 0.1316 → 0.0973 |
| 3 | 0.1211 → 0.0681 | 0.1577 → 0.1253 |
| 4 | 0.1218 → 0.0800 | 0.1153 → 0.0745 |

- **No overfitting.** Validation loss falls with training loss and is flat over epochs 15–20 in every fold. It never turns upward.
- **No underfitting.** Both losses fall by roughly 30–40% from epoch 1 and level off. More epochs would add little.
- **Generalisation gap on harder trip groups.** In folds 2 and 3, validation loss ends above training loss, by 0.021 and 0.057. These folds also have the lowest validation Top-1 (0.804 and 0.749). The gap comes from which trips were held out, not from overtraining: it stays the same size over the last epochs instead of growing.
- **Final fit.** The final model, trained on all training trips, goes from 0.1269 (epoch 1) to 0.0776 (epoch 20).

### Results

Cross-validation results, from `results/models/mlp/cv_folds.csv`:

| Metric | Mean (std) |
|---|---|
| Macro F1 | 0.9075 (0.0347) |
| Fix-level Top-1 | 0.8400 (0.0626) |
| Log loss | 0.0845 (0.0269) |
| Route continuity | 0.9478 |
| Training time per fold | 88–96 s on 500k rows |

Student-test candidate rows, from `results/models/mlp/student_test_metrics.json`:

| Metric | Value |
|---|---|
| Accuracy | 0.9728 |
| Precision | 0.8649 |
| Recall | 0.8322 |
| Macro F1 | 0.9166 |
| Log loss | 0.0688 |
| Top-1 (19,143 valid fixes) | 0.8570 |
| Route continuity | 0.9466 |
| Inference | 23.3 ms per 1,000 fixes |

Confusion matrix: TN 187,920, FP 2,487, FN 3,211, TP 15,923.

**Position in the comparison** (`results/models/model_comparison.csv`). The MLP is third on CV macro F1 (0.9075), behind XGBoost (0.9156) and Random Forest (0.9154). It is ahead of KNN (0.8887), Logistic Regression (0.8860) and Gaussian NB (0.8191). On student-test Top-1 it scores 0.857, against 0.873 for XGBoost and 0.870 for Random Forest. Its log loss (0.0845 CV) is slightly worse than the tree ensembles (0.077–0.078). Its inference is faster than Random Forest (23 vs 163 ms per 1k fixes).

### Strengths and limits

**Strengths:**
- Learns feature interactions automatically.
- Produces well-calibrated probabilities for the HMM.
- Fast inference.
- Gives an explicit view of convergence through the loss curves.

**Limits:**
- Needs feature scaling.
- Sensitive to the learning rate and the number of epochs.
- Was trained on a 500k-row sample rather than the full table.
- Harder to interpret than the tree models, which report feature importances.
- Trails the gradient-boosted trees slightly on tabular features of this kind.

### Registration and artifact

The MLP is registered as `mlp` in `src/model_zoo.py` and can be trained through `src.train`. Its standalone model is `artifacts/mlp_model.joblib`, with metadata in `artifacts/mlp_metadata.json`. It does not overwrite the shared `artifacts/model.joblib`.

## §5 HMM and Viterbi map matching

The subsections below answer the six questions required by `PROBLEM_STATEMENT.md` §6. Implementation: `src/hmm.py`. Tests: `tests/test_hmm.py`.

### 5.1 What are the hidden states?

At each GPS fix, the hidden states are the candidate directed road arcs within the 50 m candidate radius (the candidate table of §2.2, about 9–11 candidates per fix). A state is "the vehicle is on arc *a*, at its projection point." Arcs are directed, so the two directions of a two-way street are different states. Arc IDs are local to a trip and are never compared across trips.

### 5.2 How are emissions computed?

The emission log score of candidate arc `a` at fix `t` is

`E_t(a) = w · log(max(p(a | fix_t), 1e-6))`

- `p` is the classifier probability from the shipped model, the Optuna-tuned XGBoost in `artifacts/model.joblib`. Any registered classifier can be used; `tests/test_pipeline_models.py` checks each one.
- `w` is the emission weight.
- Clipping at `1e-6` prevents `log 0`.

When no classifier is loaded, the decoder falls back to a Gaussian on perpendicular distance, `-0.5 · (dist_m / sigma)^2` with `sigma = 8` m. This was used before models existed. Parameters: `artifacts/hmm.json`.

### 5.3 How are transitions computed?

For consecutive fixes, the transition compares the distance driven along the network, `d_route`, with the straight-line GPS displacement `d_gps` (haversine). This follows Newson & Krumm (2009):

`T(a → b) = -|d_route - d_gps| / beta`, with `beta = 100` m.

`d_route` is computed as follows:
- **Same arc, moving forward:** `(t_along_b - t_along_a) · len_a`.
- **Otherwise:** the remaining part of `a`, plus the shortest directed path from the end node of `a` to the start node of `b`, plus `t_along_b · len_b`.

Shortest paths use `scipy.sparse.csgraph.dijkstra` on the directed graph. Duplicate directed edges are collapsed to the shortest weight. Each search is bounded by `max(200 m, 50 m/s · dt)`, a physically plausible travel distance. Searches run only from the unique end nodes of the current candidates and are cached per trip. Directed edges mean one-way streets are respected automatically. A target beyond the search limit gets transition `-inf`.

Viterbi then works in log space:

`delta_t(j) = E_t(j) + max_i [ delta_(t-1)(i) + T(i, j) ]`

The arg-max is stored as a back-pointer, and the best final state of each segment is traced back. Working in log space avoids underflow over thousands of fixes.

### 5.4 How are missing candidates and gaps handled?

- **Fix with no candidates.** The fix is left unmatched (`arc_hmm = -1`) and removed from the state sequence. The next fix that has candidates is linked to the previous one using the elapsed time and GPS displacement across the missing fix. The candidate miss rate on student_test is 0.05% (`student_test_metrics.json`), so this case is rare. The integration run had 0 unmatched fixes on all 10 trips (`n_unmatched_fixes` in `results/hmm/xgboost/classifier_vs_hmm.csv`).
- **Time gap.** If `dt` exceeds the gap threshold (30 s in `artifacts/hmm.json`), or is zero, negative or not finite, the current segment is closed and traced back. A new, independent Viterbi segment starts at the next fix. A long gap makes the route distance between the fixes meaningless, so no transition is forced across it.
- **Very long trips.** Back-pointers are kept only for the open segment, and each segment is traced back as soon as it closes. This avoids one full-trip dynamic-programming matrix.

### 5.5 What happens with disconnected segments?

If every transition from the previous candidate set to the current one is `-inf`, Viterbi has no valid predecessor. This happens when no directed path exists (a disconnected network component) or when none falls within the bounded search. The decoder then closes and traces back the previous chain, and restarts at the current fix with emission scores only. The output is a sequence of internally consistent sub-routes rather than a crash or a forced impossible jump. `tests/test_hmm.py` covers this case, along with gaps and fixes without candidates.

### 5.6 Did the HMM help? (classifier-only vs HMM)

Decoder tuning used TRAIN trips only, never student_test or evaluator data (`results/hmm/decoder_grid.csv`, `tuning_summary.json`):
- Two size-stratified train trips (00000026, 00000006), with MLP emissions (the grid was run before the tuned XGBoost was exported).
- Eight configurations: beta ∈ {25, 100}, gap threshold ∈ {30, 60} s, emission weight ∈ {0.5, 1}, sigma = 8 m.
- Selected: `beta = 100`, `gap_threshold_s = 30`, `emission_weight = 1`, kept in `artifacts/hmm.json`.

The 30 s and 60 s gap thresholds gave identical results on these trips. This is a small smoke grid, not a full search: the parameters are a reasonable setting, not a proven optimum. Re-tuning with XGBoost emissions on TRAIN trips (`python -m src.tune_hmm`) is optional.

Results on the 10 student-test trips with the tuned XGBoost (`results/hmm/xgboost/classifier_vs_hmm.csv` and `integration_summary.json`, mean over trips):

| Metric | Classifier only | Classifier + HMM | Trips improved |
|---|---|---|---|
| Top-1 accuracy | 0.8845 | 0.8665 | 2 / 10 |
| Route continuity | 0.9443 | 0.9633 | 10 / 10 |
| Route F1 (arc sequence) | 0.7183 | 0.7434 | 8 / 10 |

**Answer: yes for route-level quality, no for pointwise accuracy.**

- **Route level.** The HMM's job is to output a drivable route, and it does that better. Continuity rises on every trip, and route F1 rises on 8 of 10. The largest F1 gains are on trips 00000052 (0.857 → 0.933) and 00000007 (0.714 → 0.778); route precision rises on 8 of 10 trips because the HMM removes isolated off-route arcs.
- **Pointwise accuracy.** Top-1 rises on 00000052 (0.954 → 0.989) and 00000056 (0.900 → 0.909), is unchanged on 00000009 (0.872), and falls on the other 7, by up to 6 percentage points (00000077: 0.722 → 0.658; 00000065: 0.838 → 0.781; 00000007: 0.932 → 0.887).
- **Why Top-1 falls.** The transition term favours connected sequences. Where the classifier is confidently right on an isolated fix that the route model finds implausible, the HMM overrides it. This happens most at junctions, U-turns and revisited arcs, where the reference route is itself imperfect (`PROBLEM_STATEMENT.md`: 14 disconnected consecutive arcs across 90 trips, plus U-turns).
- **Trip 00000077.** It has low route F1 for both methods (about 0.30): about 1,400–1,450 predicted arcs against 268 reference arcs. Recall is high (0.96–0.97) but precision is about 0.18, which suggests the prediction covers a much larger part of the network than the reference route does.
- **Maps.** `maps/results/xgboost_hmm/` has the two trips where the HMM helped least (00000077 and 00000065), with per-fix error types (correct, wrong direction, wrong road) as in the other result maps. All non-improvement trips are listed in `results/hmm/xgboost/hmm_non_improvements.csv`.

A larger tuning sweep could shift this trade-off: a higher emission weight leans towards Top-1, a lower one towards continuity. Any such sweep must stay on TRAIN trips.

## Integration notes

- **Inference has no access to the reference route.** `src.pipeline.load_artifacts()` loads the model, the ordered feature list and the decoder settings once. `predict_trip()` generates candidates and features, scores them with the classifier and runs Viterbi. It returns per-fix predictions, a collapsed route and a route polyline. The evaluation harness (`src/evaluate_hmm.py`) sets `inference_trip.route = None` before calling `predict_trip()`. Only `score_trip()` receives the reference route.
- **Robustness.** All 10 student-test trips completed with 0 failures (`results/hmm/xgboost/integration_failures.csv` is empty).
- **Latency.** Per-trip timings are in `results/hmm/xgboost/classifier_vs_hmm.csv` (`inference_ms`, `candidate_generation_ms`, `model_ms`, `hmm_ms`).
  - Mean end-to-end time is 3.4 s per trip. It ranges from 0.54 s (00000052, 262 fixes) to 6.9 s (00000056, 3,958 fixes).
  - Viterbi decoding is most of the cost, 61–78% of each trip's time (`hmm_ms`). Candidate generation (about 0.04–0.21 s) and the XGBoost model (about 0.01–0.13 s) are small.
  - Long trips therefore exceed the "few seconds" demo target. Caching Dijkstra rows across segments, or tightening the search limit, are the first things to try.
  - Latency depends on the machine.
- **Reproducing.** `python -m src.evaluate_hmm` (or `notebooks/julia_hmm_viterbi_colab.ipynb`) regenerates every number in §5.6 and in this section; `integration_summary.json` is computed from the CSV of the same run.
