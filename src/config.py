"""Shared constants for the whole pipeline. Change only after agreeing with the team."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = 42
N_FOLDS = 5                       # GroupKFold by trip on the 80 train trips

# Candidate generation: every arc within this distance of a GPS fix is a candidate.
# Chosen from the miss-rate analysis in results/candidates/ (python -m src.build_dataset).
CANDIDATE_RADIUS_M = 50.0

# Labelling: a fix whose route-matched arc is farther than this is treated as not
# covered by the reference route (no reliable ground truth). Kept larger than the
# candidate radius so the candidate miss rate stays a meaningful measurement.
LABEL_MAX_DIST_M = 100.0
# Distance cap inside the route-matching cost, so long uncovered stretches cannot
# pull the alignment of the rest of the trip.
LABEL_COST_CAP_M = 100.0

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
