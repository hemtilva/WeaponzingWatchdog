#!/usr/bin/env bash
# ==============================================================================
# Launch Script for Full 2-3 Day Production Experiment Suite
# Weaponizing the Watchdog: Regulatory Exploitation in Learning Markets
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_FILE="$PROJECT_ROOT/production_sweep.log"
PID_FILE="$PROJECT_ROOT/production_sweep.pid"

echo "======================================================="
echo "Weaponizing the Watchdog: Production Suite Launcher"
echo "======================================================="
echo "Project Directory: $PROJECT_ROOT"
echo "Log File:          $LOG_FILE"
echo "PID File:          $PID_FILE"
echo "======================================================="

# Verify virtual environment
if [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_EXEC="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

echo "Using Python: $PYTHON_EXEC"

# Launch in background with nohup
nohup "$PYTHON_EXEC" "$PROJECT_ROOT/experiments/run_full_production_suite.py" --production > "$LOG_FILE" 2>&1 &
SWEEP_PID=$!
echo "$SWEEP_PID" > "$PID_FILE"

echo ""
echo "[SUCCESS] Production sweep started in background!"
echo "Process ID (PID): $SWEEP_PID"
echo ""
echo "To monitor live progress, run:"
echo "  tail -f production_sweep.log"
echo ""
echo "To check if it is still running:"
echo "  kill -0 $SWEEP_PID 2>/dev/null && echo 'Running' || echo 'Completed'"
echo "======================================================="

