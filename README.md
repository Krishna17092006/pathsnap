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
├── data/                  # raw dataset (not committed; see "Data setup")
│   ├── train/             # 80 trips
│   └── student_test/      # 10 trips
├── src/
│   ├── data_io.py         # raw file loaders (.track / .nodes / .arcs / .route)
│   ├── geo.py             # haversine, local metric projection, point-to-segment distance
│   ├── network.py         # road-network geometry, spatial index, connectivity, route helpers
│   ├── plot_style.py      # shared matplotlib style
│   ├── maps.py            # Folium map builders
│   └── eda.py             # exploratory data analysis
├── results/eda/           # EDA tables (CSV), figures (PNG), summary
└── maps/eda/              # interactive Folium maps (HTML)
```

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

### Data setup

Unzip the provided `Student folder.zip` so the trip folders sit directly under `data/`:

```
data/train/00000001/00000001.track
data/train/00000001/00000001.nodes
data/train/00000001/00000001.arcs
data/train/00000001/00000001.route
...
data/student_test/00000000/...
```

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

## Data-partition rules

- `train` (80 trips) is used for fitting, cross-validation and tuning.
- `student_test` (10 trips) is used for development-stage model comparison.
- Evaluator trips are used only by the deployed API during the final evaluation. They are never used for training, feature selection or tuning.
