"""Run Julia's PathSnap HMM/Viterbi decoder tuning and 10-trip integration evaluation.

Examples
--------
# Recommended first run: tune on a small TRAIN subset, evaluate all 10 student-test trips
python scripts/run_julia_hmm.py

# Skip tuning and use the current artifacts/hmm.json
python scripts/run_julia_hmm.py --skip-tune

# Tune on more training trips (can take substantially longer)
python scripts/run_julia_hmm.py --tune-trips 5

# Quick smoke test: tune on 2 train trips and evaluate 2 student-test trips
python scripts/run_julia_hmm.py --quick-eval
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # scripts/ -> project root


def run_step(command: list[str], label: str) -> None:
    print(f"\n{'=' * 72}\n{label}\n$ {' '.join(command)}\n{'=' * 72}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tune", action="store_true", help="Use the current artifacts/hmm.json")
    parser.add_argument("--tune-trips", type=int, default=2,
                        help="Number of TRAIN trips for decoder tuning (default: 2; larger is slower)")
    parser.add_argument("--quick-eval", action="store_true", help="Evaluate only 2 student-test trips")
    args = parser.parse_args()

    required = [ROOT / "src" / "hmm.py", ROOT / "src" / "pipeline.py",
                ROOT / "src" / "tune_hmm.py", ROOT / "src" / "evaluate_hmm.py",
                ROOT / "data" / "train", ROOT / "data" / "student_test"]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing required project files/directories: " + ", ".join(missing))

    if not args.skip_tune:
        if args.tune_trips < 2:
            raise SystemExit("--tune-trips must be at least 2")
        run_step([sys.executable, "-m", "src.tune_hmm", "--max-trips", str(args.tune_trips)],
                 "STEP 1 — Tune HMM decoder on TRAIN trips only")
    else:
        if not (ROOT / "artifacts" / "hmm.json").exists():
            raise SystemExit("artifacts/hmm.json is missing; remove --skip-tune to create it")
        print("Skipping tuning; using artifacts/hmm.json", flush=True)

    eval_cmd = [sys.executable, "-m", "src.evaluate_hmm"]
    if args.quick_eval:
        eval_cmd.append("--quick")
    run_step(eval_cmd, "STEP 2 — Compare classifier-only vs HMM on student_test")

    # Outputs are written per emission model: results/hmm/<model>/ and maps/results/<model>_hmm/.
    sys.path.insert(0, str(ROOT))
    from src.pipeline import load_artifacts
    name = load_artifacts(ROOT / "artifacts")["model_name"]
    print(f"\nCompleted (emission model: {name}). Key outputs:")
    for rel in ["artifacts/hmm.json", f"results/hmm/{name}/decoder_grid.csv",
                f"results/hmm/{name}/classifier_vs_hmm.csv", f"results/hmm/{name}/hmm_non_improvements.csv",
                f"results/hmm/{name}/integration_summary.json", f"results/hmm/{name}/predictions.parquet",
                f"results/hmm/{name}/classifier_vs_hmm.png", f"maps/results/{name}_hmm"]:
        p = ROOT / rel
        print(f"  {'OK' if p.exists() else 'MISSING'}  {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
