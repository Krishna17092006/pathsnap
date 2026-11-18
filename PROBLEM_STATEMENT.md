# PathSnap: Problem Statement

**UE24CS352A Machine Learning hackathon, PES University (RR Campus)**

This file explains what we are building and why. It condenses the five official PDFs (Main Guidelines, Models and Training Guidelines, Deliverables, Assessment and Plagiarism Policy, Dataset Exploration) and adds what our EDA found in the real data. Read it before touching any code.

---

## 1. The problem in one paragraph

A car records its GPS position roughly once a second, but GPS is noisy. Tall buildings, signal reflections (multipath), interference and dropouts make points drift onto a parallel street, jump across buildings, or vanish for a while. **Our job is to take a trip's noisy GPS points and work out which road segment the car was really on at every moment, then rebuild the complete route it drove.** This is called **map matching**. We solve it with machine learning (a classifier that scores nearby road segments) plus a sequence model (an HMM decoded with Viterbi) that keeps the route physically connected. We then serve it through an API and a UI.

```
noisy GPS fixes  ──►  nearby road segments (candidates)  ──►  classifier scores each candidate
                                                                         │
            predicted route  ◄──  Viterbi picks the best connected chain ◄──  HMM over the sequence
```

---

## 2. What the system must output

For a given trip:

1. **For every GPS fix:** the id of the road segment (arc) the vehicle was on.
2. **For the whole trip:** the reconstructed route, a connected sequence of arcs.
3. **Evaluation numbers:** the API compares its prediction with the reference route it holds privately. Examples are pointwise accuracy and route continuity.

---

## 3. The data

### 3.1 Files

Every trip is a folder `<partition>/<trip_id>/` with four **headerless, whitespace-separated** files:

| File | Columns | Meaning |
|---|---|---|
| `.track` | `lon lat t` | GPS fixes. `t` is in seconds, nominally 1 Hz. |
| `.nodes` | `lon lat` | Points of the road map. **Row number (0-based) = node id.** |
| `.arcs` | `u v` | Directed road segments from node `u` to node `v`. **Row number (0-based) = arc id.** |
| `.route` | `arc_id` | The reference (ground-truth) route: arc ids the vehicle drove, **in order**. |

How they connect:

```
.track rows  ──(we must infer)──►  arc ids  ──►  .arcs row (u, v)  ──►  .nodes rows (lon, lat)
                                       ▲
                       .route lists these arc ids in driving order
```

### 3.2 Partitions

| Partition | Trips | Allowed use |
|---|---|---|
| `train` | 80 | Training, cross-validation, hyperparameter tuning |
| `student_test` | 10 | Development-time comparison of models and decoder settings |
| evaluator test | 10 | **Final demo only.** Not given to us during development. Trip ids 15, 17, 23, 30, 49, 59, 67, 73, 87, 94 are missing from our data. |

A trip never appears in two partitions.

### 3.3 What the EDA found (numbers from `results/eda/`)

| Fact | Consequence for us |
|---|---|
| **No per-fix labels.** `.route` lists arcs, not "fix i → arc j". | We must **derive the label** for each fix by matching it to the reference route in order. The whole training set depends on this step. |
| **Every trip has its own road network.** Trips are spread across many countries (Germany, Russia, Kazakhstan, USA, France, Italy, Argentina, Brazil, South Africa, ...). | Arc 500 in one trip is unrelated to arc 500 in another. **Features must be geometric** (distances, angles, speeds), **never arc ids.** |
| Coordinates are degrees; trips span up to ~75 km. | Never use Euclidean distance on degrees. We use haversine for lengths and a per-trip local metric projection for geometry (max scale error 0.78%, checked in EDA). |
| 208,417 train fixes and 19,160 student-test fixes. 96.5% / 98.6% of intervals are exactly 1 s. | Data is dense. Gaps are rare but real. |
| 11 train trips have gaps over 300 s. The largest is 153,064 s (~42 h) in trip 00000090. | The HMM must split or relax at large gaps. |
| Median fix-to-route distance is about 5 m. Student test: 98.6% of fixes within 25 m, 99.9% within 50 m. | A candidate radius of about 50 m captures nearly every true arc. |
| Two train trips (00000054, 00000033) have GPS paths 5–7× longer than their reference route; most of their fixes are more than 100 m from it. | Their reference route covers only part of the drive. Fixes far from the route must be **excluded from labels** (no reliable ground truth). |
| A GPS fix has a median of 4 arcs within 25 m, 8 within 50 m and about 20 within 100 m. Two-way roads appear as two arcs. | Each fix has several plausible candidates. Direction (heading) matters for choosing between the two directions of a road. |
| Picking the geometrically nearest arc is right in about 89% of fixes (median per trip). | This is the naive baseline that our classifier + HMM must beat. |
| Reference routes are imperfect: 14 disconnected consecutive arcs across 90 trips, plus U-turns and revisited arcs. | Treat the reference as "reference", not perfect truth, and say so in the report. |
| Networks contain duplicate arcs, self-loops, zero-length arcs and disconnected pieces. Arc length median is about 25–32 m, max about 9 km. | Code must be robust to degenerate geometry. |

---

## 4. The pipeline we must build (official 12 steps)

1. **Understand** the problem.
2. **Explore** the data (EDA, done; see `results/eda/eda_summary.md`).
3. **Process data and engineer features:** labels, candidate generation, features.
4. **Train and compare at least 5 models** from different algorithm families.
5. **Select** the best model using measured evidence, not training accuracy.
6. **Bayesian hyperparameter optimisation** of the chosen model.
7. **HMM + Viterbi** route reconstruction.
8. **Export** the model and every artifact needed for inference.
9. **FastAPI backend** (Laptop 1).
10. **Evaluator UI** (Laptop 2, Streamlit).
11. **Integrate and test** the whole system.
12. **Technical report** (PDF).

### The core ML formulation

- **Candidate generation:** for each GPS fix, take every arc within R metres (R ≈ 50 m).
- **Classification:** each (fix, candidate arc) pair is one row, with label 1 if that arc is the fix's true arc and 0 otherwise. A classifier outputs P(correct) per candidate.
- **Classifier-only prediction:** for each fix, take the candidate with the highest probability.
- **HMM:**
  - *Hidden states:* the candidate arcs at each fix.
  - *Emission:* classifier probability.
  - *Transition:* how plausible it is to move from arc A to arc B, given road connectivity and the distance travelled.
  - *Viterbi* finds the single most likely connected chain. It fixes isolated wrong picks that jump between unconnected roads.

---

## 5. How we are evaluated

### 5.1 Final demo (two laptops)

1. The API runs on **Laptop 1** with the model, HMM, road networks and the private reference routes.
2. The UI runs on **Laptop 2**.
3. The evaluator opens the UI and sees the **10 evaluator trips**.
4. They pick any trip. The UI sends **only the trip id** to the API.
5. The API runs the **full pipeline**, computes accuracy against its private reference, and returns the result.
6. The UI shows the predicted route and the results.
7. We show the evaluator the **API logs** in the terminal of Laptop 1.

It must work for all 10 trips. Ground truth never leaves the API: the UI and the model never see it. No hardcoded outputs.

### 5.2 Metrics we must report

| Metric | Definition |
|---|---|
| Accuracy, precision, recall, F1 (macro), confusion matrix | On candidate rows (binary: correct arc or not) |
| Log loss | Quality of predicted probabilities |
| **Observation-level Top-1 accuracy** | Per fix: is the highest-scored candidate the reference arc? |
| **Candidate miss rate** | Share of fixes whose reference arc is not among the candidates at all |
| **Route continuity** | Not defined by the organisers; we define it. Example: share of consecutive predictions that are the same arc or directly connected arcs. |
| **Sequence-level performance** | Not defined by the organisers; we define it. Example: precision/recall of predicted route arcs vs reference route arcs. |
| Training time, inference time | Computational cost |

We must state which metric picked the final model and why.

### 5.3 Viva

Any team member can be asked about any part: dataset, preprocessing, features, model comparison, model selection, Bayesian optimisation, HMM, Viterbi, API, UI, results. **Everyone must understand everything.**

---

## 6. Deliverables

1. **Public GitHub repository**, organised into clear folders, containing:
   - code for data processing, features, every model trained, evaluation, Bayesian optimisation, HMM/Viterbi, API and UI
   - deployment config, Folium map code, result CSVs and logs
   - saved model weights (if size allows)
   - a README with install, run and deploy instructions
   - no secrets
2. **One PDF report** with these sections:
   - **§2 EDA:** preprocessing, patterns, chosen features and why, missing values, trajectory and network plots, at least one Folium map, and which partition each analysis used.
   - **§3 Models:** a comparison table (Model, Family, Key config, CV Macro F1, Top-1, Route continuity, Student-test Macro F1, Log loss, Inference time). Then a subsection per model: why suitable, strengths and limits, hyperparameters, loss, results, over/underfitting evidence.
     - Plots: CV F1 comparison, student-test Top-1/F1, log loss.
     - Loss curves only for models that really train in epochs.
   - **§4 Bayesian optimisation:** parameters searched, ranges, objective, number of trials, best values, tuned vs untuned.
   - **§5 HMM:** answers to:
     - What are the hidden states?
     - How are emissions computed?
     - How are transitions computed?
     - How are missing candidates and gaps handled?
     - What happens with disconnected segments?
     - Did the HMM help? (classifier-only vs HMM)
   - **§6 Folium maps:** GPS, road network, reference route and predicted route, with a legend and layer control.
   - **§7 API/UI:** screenshots of the UI and of API responses in the receiving laptop's terminal.

Every number in the report must be reproducible from the repository.

---

## 7. Rules that cost marks

- **No leakage:**
  - Evaluator trips are never used for training, feature selection, tuning or thresholds.
  - Fit scalers and other preprocessing on training data only.
  - Cross-validation must be **group-aware**: a whole trip stays in one fold.
- **No hardcoding** of predictions. Repositories are inspected.
- **No sharing** of code, models, configs, derived data or results with other teams. Similarity is checked across teams and sections.
- **No fabricated** metrics, graphs or curves.
- **AI-use disclosure** as required by the event policy. AI cannot replace real experiments.
- Every member must be able to explain the code and results.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| Fix | One GPS observation (a row of `.track`) |
| Node | A point of the road map (a row of `.nodes`) |
| Arc | A directed road segment between two nodes (a row of `.arcs`). A two-way road is two arcs. |
| Reference route | The `.route` file: ordered arcs the vehicle drove |
| Candidate | An arc within R metres of a fix that might be the true one |
| Label (per fix) | The reference arc assigned to a fix by matching the fix to the reference route in order |
| Emission probability | How likely a fix is, given the vehicle is on a candidate arc (our classifier's probability) |
| Transition probability | How likely the vehicle moves from arc A (at fix t) to arc B (at fix t+1) |
| Viterbi | Dynamic programming that finds the most probable sequence of hidden states |
| GroupKFold | Cross-validation where all rows of one trip stay in the same fold |
| Macro F1 | F1 averaged equally over classes (handles the imbalance of 1 true candidate vs several false ones) |
| Log loss | Penalises confident wrong probabilities; lower is better |
| Top-1 accuracy | Share of fixes where the top-scored candidate is the reference arc |
