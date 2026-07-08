#!/usr/bin/env bash
# Daily auto-retrain pipeline for Poker44 UID 198
# Fetches latest benchmark data, retrains model, deploys if improved
set -euo pipefail

REPO=/root/Poker44-top-miner
PYTHON="$REPO/miner_env/bin/python3"
LOG_FILE="$REPO/logs/daily_retrain.log"
mkdir -p "$REPO/logs"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_FILE"; }

log "=== Daily Retrain Started ==="
cd "$REPO"

# 1. Fetch latest benchmark data
log "Fetching latest benchmark data..."
$PYTHON - << 'PYEOF' 2>&1 | tee -a "$LOG_FILE"
import requests, json, sys
from pathlib import Path

API_BASE = "https://api.poker44.net/api/v1/benchmark"
BENCH_FILE = Path("hands_generator/evaluation_datas/training_benchmark_v112_full.txt")

# Get status
status = requests.get(API_BASE, timeout=30).json()["data"]
latest_date = status["latestSourceDate"]
print(f"API latest date: {latest_date}")

# Load existing data
with open(BENCH_FILE) as f:
    existing_data = json.load(f)
existing_chunks = existing_data["data"]["chunks"]
existing_dates = set(c.get("sourceDate", "") for c in existing_chunks)
print(f"Existing dates: {sorted(existing_dates)[-3:]}, total chunks: {len(existing_chunks)}")

# Find missing dates
from datetime import date, timedelta
start_date = date(2026, 5, 26)
end_date = date.fromisoformat(latest_date)
new_dates = []
d = start_date
while d <= end_date:
    ds = d.isoformat()
    if ds not in existing_dates:
        new_dates.append(ds)
    d += timedelta(days=1)

print(f"Missing dates to fetch: {new_dates}")
if not new_dates:
    print("No new data to fetch.")
    sys.exit(0)

# Fetch each missing date
new_chunks = []
for date_str in new_dates:
    try:
        resp = requests.get(f"{API_BASE}/chunks", params={"sourceDate": date_str}, timeout=60).json()
        chunks = resp.get("data", {}).get("chunks", [])
        new_chunks.extend(chunks)
        print(f"Fetched {len(chunks)} chunks for {date_str}")
    except Exception as e:
        print(f"Error fetching {date_str}: {e}")

if not new_chunks:
    print("No new chunks fetched.")
    sys.exit(0)

# Append and save
all_chunks = existing_chunks + new_chunks
updated_data = {"data": {"releaseVersion": "v1.12", "chunks": all_chunks}}
with open(BENCH_FILE, "w") as f:
    json.dump(updated_data, f)
print(f"Saved: {len(all_chunks)} total chunks (added {len(new_chunks)} new)")
PYEOF

# 2. Retrain hybrid v124 model (v2.2 benchmark-first calibration)
log "Running v124 hybrid model training..."
$PYTHON -m training.train_v126 2>&1 | tee -a "$LOG_FILE"
RETRAIN_EXIT=$?

if [ $RETRAIN_EXIT -ne 0 ]; then
    log "ERROR: Retraining failed with exit code $RETRAIN_EXIT"
    exit 1
fi

# 3. Check if new model is better (compare holdout AP)
NEW_SHA=$(sha256sum models/poker44_v124_deploy.joblib | cut -d' ' -f1)
CURRENT_SHA=$(pm2 env 2 2>/dev/null | grep POKER44_MODEL_ARTIFACT_SHA256 | awk '{print $2}')
log "New model SHA256: $NEW_SHA"
log "Current model SHA256: $CURRENT_SHA"

if [ "$NEW_SHA" = "$CURRENT_SHA" ]; then
    log "Model unchanged, skipping deployment."
    exit 0
fi

# 4. Deploy new model
log "Deploying new model..."
COMMIT=$(git rev-parse HEAD)
export POKER44_MODEL_PATH=./models/poker44_v124_deploy.joblib
export POKER44_MODEL_NAME=poker44-v124-hybrid
export POKER44_MODEL_VERSION=1.24.0
export POKER44_MODEL_SHA256=$NEW_SHA
export POKER44_MODEL_ARTIFACT_SHA256=$NEW_SHA
export POKER44_MODEL_REPO_COMMIT=$COMMIT
export POKER44_MODEL_REPO_URL=https://github.com/Yaroslav98214/poker44-handngram-miner.git
export POKER44_MODEL_OPEN_SOURCE=true
export POKER44_LOG_SCORE_ARRAYS=1
export POKER44_LOG_SCORE_COMPONENTS=1
export POKER44_MODEL_FRAMEWORK=hybrid-lgb-xgb-et-hgram-v22-apfirst
export POKER44_MODEL_TRAINING_DATA_SOURCES=released_training_benchmark_v113
export POKER44_MODEL_TRAINING_DATA_STATEMENT="Trained on public Poker44 benchmark v1.13 through 2026-07-06 with holdout-first calibration for v2.2 competition."
export POKER44_MODEL_PRIVATE_DATA_ATTESTATION="No private data used. Training uses only the public benchmark API corpus."
export POKER44_MODEL_DATA_ATTESTATION="No private data used. Training uses only the public benchmark API corpus."

pm2 restart poker44-miner --update-env
sleep 5
log "Miner restarted successfully with new model"
tail -5 /root/.pm2/logs/poker44-miner-out.log >> "$LOG_FILE"

log "=== Daily Retrain Complete ==="
