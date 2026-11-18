# Julia — report contribution draft

## §3.2 Model 6: Multilayer Perceptron (MLP)

The MLP predicts whether a candidate directed road arc is the correct road segment for a GPS fix. It uses the 19 engineered candidate features saved in `artifacts/features.json`. The network has two fully connected hidden layers with 128 and 64 neurons, ReLU activation, and a sigmoid probability output. It is trained with Adam (learning rate 0.001), batch size 8,192, L2 regularisation parameter `alpha = 0.0001`, and 20 epochs. The random seed is 42.

Evaluation uses five-fold `GroupKFold` with `trip_id` as the grouping variable, so fixes from one trip cannot be split between training and validation. Median imputation and standardisation are fitted on training data only. The training loop uses `partial_fit` epoch by epoch and records train and validation log loss at each epoch; the resulting loss histories are saved as CSV files and plotted in `results/models/mlp/`.

Across the five folds, mean macro F1 was **0.9075** (standard deviation **0.0347**), mean fix-level Top-1 accuracy was **0.8400**, and mean log loss was **0.0845**. On the student-test candidate rows, candidate-level accuracy was **0.9728**, precision **0.8649**, recall **0.8322**, macro F1 **0.9166**, and log loss **0.06885**. Fix-level Top-1 accuracy was **0.8570** over 19,143 valid fixes. The final training loss decreased from **0.12694** at epoch 1 to **0.07763** at epoch 20. These values are reproduced from `results/models/mlp/`; the CV and final training loss curves are `cv_loss_curves.png` and `final_training_loss.png`.

The MLP is registered as `mlp` in `src/model_zoo.py` and trained through `src.train`. Its standalone model is saved as `artifacts/mlp_model.joblib`, with the ordered feature list and metadata. This is Julia's model artifact and does not overwrite Joshua's shared `artifacts/model.joblib`.

## §5 HMM and Viterbi map matching

### 5.1 Hidden states

For each GPS fix, the hidden-state set is the set of candidate directed road arcs found within the candidate search radius. A state therefore represents a plausible road segment for the vehicle at that timestamp. Arc IDs are local to a trip; the decoder never compares arc IDs across trips.

### 5.2 Emission model

When a classifier is available, the emission score for candidate arc `a` at fix `t` is the logarithm of its predicted probability, `log(max(p(a|fix_t), 1e-6))`. The clipping avoids taking the logarithm of zero. When no classifier is available, the decoder uses the distance-based Gaussian log score `-0.5 * (dist_m / sigma)^2`, with `sigma` configured in `artifacts/hmm.json`. The current integration uses MLP classifier probabilities.

### 5.3 Transition model and directed shortest paths

For consecutive fixes, the transition score compares network route distance `d_route` with observed GPS displacement `d_gps`. The log transition score is `-|d_route - d_gps| / beta`. For the same directed arc when the second projection is further forward, the route distance is `(t_along_b - t_along_a) * arc_length`. Otherwise it is the remaining distance on the first arc, plus the shortest directed path from the first arc's end node to the second arc's start node, plus the distance along the second arc to the second projection.

Shortest paths are computed with `scipy.sparse.csgraph.dijkstra` on a directed sparse graph. The search limit is `max(200 m, 50 m/s × dt)`. Dijkstra is run from unique end nodes of the previous candidate arcs, and results are cached during a trip. Directed edges preserve one-way restrictions. Duplicate directed edges are collapsed to the shortest edge weight before graph construction.

### 5.4 Viterbi recurrence

The decoder works in log space. Let `E_t(j)` be the emission log score for state `j` at time `t`, and `T(i,j)` be the transition log score from previous state `i` to current state `j`. The recurrence is:

`delta_t(j) = E_t(j) + max_i [delta_(t-1)(i) + T(i,j)]`.

The maximizing previous state is stored in a back-pointer array. After the final fix in a connected segment, the best final state is selected and the back-pointers are followed in reverse to recover the most likely sequence. Log-space addition avoids numerical underflow from multiplying many small probabilities.

### 5.5 Missing candidates, gaps, disconnected networks and long trips

A fix with no candidates is left unmatched (`arc_hmm = -1`). It is omitted from the hidden-state sequence, while the next candidate-bearing fix is connected to the previous candidate-bearing fix using the elapsed time and GPS displacement across the gap. A time difference greater than the configured gap threshold starts a new independent Viterbi segment. If every transition between two candidate sets is impossible because the directed network is disconnected or the bounded search cannot reach the next arc, the previous segment is backtracked and decoding restarts at the current fix. The implementation retains back-pointers only for the current segment and closes segments as it goes, avoiding one full-trip dynamic-programming matrix for long trips. These behaviours are covered by `tests/test_hmm.py`.

### 5.6 Metrics, tuning and observed results

Pointwise Top-1 accuracy is measured only on fixes with reliable reference labels; a missing prediction counts as incorrect. Route continuity is the proportion of consecutive temporally valid fix pairs whose predicted arcs are identical or directly connected (`v[a] == u[b]`). Sequence-level route precision, recall and F1 compare the predicted route-arc sequence with the reference route after consecutive duplicate arcs are collapsed.

The decoder grid tests beta, distance sigma, GPS gap threshold and emission weight. Tuning uses TRAIN trips only; student-test is reserved for the final comparison, and evaluator data is not used. The current fast grid uses two TRAIN trips, eight parameter combinations and MLP probability emissions. It selects `beta = 100`, `sigma = 8`, `gap_threshold_s = 30`, and `emission_weight = 1`. This is a preliminary tuning subset; a larger training-trip sweep should be run if compute time permits before describing the parameters as globally optimal.

On all 10 student-test trips, the classifier-only mean Top-1 was **0.8732**, compared with **0.8560** after HMM decoding. Thus HMM Top-1 improved on 2 trips and did not improve on 8 trips. However, mean route continuity increased from **0.9437** to **0.9619**, and mean route-level F1 increased from **0.7084** to **0.7374**. Mean end-to-end inference time was approximately **1.87 seconds per trip**, with the slowest trip approximately **3.92 seconds**. All 10 trips completed without integration failures. The per-trip results are in `results/hmm/classifier_vs_hmm.csv`; examples where HMM did not improve Top-1 are in `maps/results/`.

## Integration notes

`src.pipeline.load_artifacts()` loads the model, ordered feature list and decoder settings. `predict_trip()` generates candidates and features, predicts candidate probabilities, runs Viterbi and returns per-fix predictions plus a collapsed route and route polyline. The evaluation harness explicitly sets `inference_trip.route = None` before calling `predict_trip()`. Only the separate `score_trip()` call receives the original trip and reference route. This keeps reference-route information out of inference.

The 10-trip integration run produced zero failures. The measured mean runtime was around 1.9 seconds and maximum runtime around 3.9 seconds in the execution environment used for this run. Actual latency will vary with CPU, memory and network size.

> Note: TEAM.md refers to six required HMM questions but does not enumerate them. This section is organised around six technical areas explicitly specified in the Julia task list: states, emissions, transitions, Viterbi, edge cases, and metrics/tuning/results. Align the subsection headings to the exact review-question wording if the instructor supplies a separate list.
