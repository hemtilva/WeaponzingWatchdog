#!/usr/bin/env bash
# Run the Weaponizing-the-Watchdog studies on this machine and push the results.
#
# Usage (from anywhere inside the repo, or after cloning):
#   bash scripts/run_study.sh                 # all studies, then commit + push results
#   STUDY=n4long bash scripts/run_study.sh    # one study (single|h2|h3|robust|n4long|all)
#   QUICK=1 NO_PUSH=1 bash scripts/run_study.sh   # smoke test, nothing pushed
#
# What it does:
#   1. git pull (fast-forward only) on branch fix/learnable-framing
#   2. creates .venv and installs dependencies (first run only)
#   3. builds the C++ engine and checks that the native simulation loop loads
#   4. runs the test suite (stops if anything fails)
#   5. runs the study, keeping the machine awake (caffeinate on macOS)
#   6. copies CSVs, plots, the log and machine info to results/<machine>/<study>/
#   7. commits those files and pushes them to the branch
#
# It never changes source code and never pushes anything except results/.

set -euo pipefail

BRANCH="${BRANCH:-fix/learnable-framing}"
STUDY="${STUDY:-all}"
QUICK="${QUICK:-0}"
NO_PUSH="${NO_PUSH:-0}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

cd "$(git rev-parse --show-toplevel 2>/dev/null)" || die "run this inside the WeaponzingWatchdog repo"

# ---------------------------------------------------------------- 1. code
say "Updating code ($BRANCH)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    die "you have uncommitted changes to tracked files; commit or stash them first"
fi
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"
COMMIT="$(git rev-parse --short HEAD)"
echo "code at $COMMIT"

# ---------------------------------------------------------------- 2. python
say "Python environment"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' || die "need Python >= 3.10"
if [ ! -d .venv ]; then
    "$PY" -m venv .venv
fi
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    # shellcheck disable=SC1091
    source .venv/Scripts/activate   # Windows (Git Bash)
fi
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy scipy matplotlib pandas pytest

# ---------------------------------------------------------------- 3. engine
say "Building the C++ engine"
python scripts/build_engine.py
python - <<'PY' || die "native simulation loop did not load (see build output above)"
from src.engine.native_sim import is_native_sim_available
assert is_native_sim_available()
print("native engine: OK")
PY

# ---------------------------------------------------------------- 4. tests
say "Running tests"
python -m pytest tests -q -p no:cacheprovider || die "tests failed; not running studies"

# ---------------------------------------------------------------- 5. study
say "Running study: $STUDY"
mkdir -p data
MACHINE="$(hostname -s 2>/dev/null || hostname)"
MACHINE="${MACHINE//[^A-Za-z0-9_-]/_}"
LOG="data/study_${STUDY}_${MACHINE}.log"
QUICK_FLAG=""
[ "$QUICK" = "1" ] && QUICK_FLAG="--quick"
# macOS: keep the machine awake while the study runs (bash 3.2-safe, no empty arrays)
AWAKE=""
command -v caffeinate >/dev/null 2>&1 && AWAKE="caffeinate -i"
START="$(date '+%Y-%m-%d %H:%M:%S')"
if ! $AWAKE python experiments/run_study.py "$STUDY" $QUICK_FLAG 2>&1 | tee "$LOG"; then
    die "study failed; see $LOG"
fi

# ---------------------------------------------------------------- 6. collect
if [ "$QUICK" = "1" ]; then SUF="_quick"; else SUF=""; fi
if [ "$STUDY" = "all" ]; then STUDIES=(single h2 h3 robust n4long); else STUDIES=("$STUDY"); fi
OUT="results/${MACHINE}/${STUDY}${SUF}"
say "Collecting results into $OUT"
mkdir -p "$OUT"
for s in "${STUDIES[@]}"; do
    cp "data/study_${s}${SUF}.csv" "$OUT/" 2>/dev/null || echo "  (no CSV for $s)"
    cp "plots/study_${s}${SUF}.png" "$OUT/" 2>/dev/null || echo "  (no plot for $s)"
done
cp "$LOG" "$OUT/run.log"
{
    echo "study:    $STUDY (quick=$QUICK)"
    echo "commit:   $COMMIT"
    echo "branch:   $BRANCH"
    echo "machine:  $MACHINE"
    echo "started:  $START"
    echo "finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "python:   $(python --version 2>&1)"
    echo "os:       $(uname -srm 2>/dev/null || echo unknown)"
    if command -v sysctl >/dev/null 2>&1; then
        echo "cpu:      $(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
        echo "cores:    $(sysctl -n hw.ncpu 2>/dev/null || true)"
    fi
} > "$OUT/machine.txt"
ls -la "$OUT"

# ---------------------------------------------------------------- 7. push
if [ "$NO_PUSH" = "1" ]; then
    say "NO_PUSH=1: results left in $OUT (not committed)"
    exit 0
fi
say "Committing and pushing results"
git add -f "$OUT"
if git diff --cached --quiet; then
    echo "nothing new to commit"
    exit 0
fi
git commit -m "Results: $STUDY on $MACHINE (code $COMMIT)"
for attempt in 1 2 3; do
    if git push origin "$BRANCH"; then
        say "Done. Results pushed to $BRANCH under $OUT"
        exit 0
    fi
    echo "push rejected (someone pushed meanwhile); rebasing and retrying ($attempt/3)"
    git pull --rebase origin "$BRANCH"
done
die "could not push; results are committed locally, run 'git push origin $BRANCH' later"
