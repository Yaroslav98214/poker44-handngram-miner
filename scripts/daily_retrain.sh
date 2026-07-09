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

# 2. Retrain hybrid v127 model (Jul 9 benchmark + live bot-rate guard)
log "Running v127 hybrid model training..."
$PYTHON -m training.train_v127 2>&1 | tee -a "$LOG_FILE"
RETRAIN_EXIT=$?

if [ $RETRAIN_EXIT -ne 0 ]; then
    log "ERROR: Retraining failed with exit code $RETRAIN_EXIT"
    exit 1
fi

# 3. Deploy if model artifact changed
NEW_SHA=$(sha256sum models/poker44_v127_deploy.joblib | cut -d' ' -f1)
CURRENT_SHA=$(grep POKER44_MODEL_ARTIFACT_SHA256 scripts/miner/ecosystem.config.cjs | head -1 | tr -d ' ",')
log "New model SHA256: $NEW_SHA"
log "Ecosystem SHA256: $CURRENT_SHA"

if [ "$NEW_SHA" = "$CURRENT_SHA" ]; then
    log "Model unchanged, skipping deployment."
    exit 0
fi

# 4. Update ecosystem config and restart miner
log "Deploying new model..."
COMMIT=$(git rev-parse HEAD)
sed -i "s|POKER44_MODEL_PATH:.*|POKER44_MODEL_PATH: \"/root/Poker44-top-miner/models/poker44_v127_deploy.joblib\",|" scripts/miner/ecosystem.config.cjs
sed -i "s|POKER44_MODEL_SHA256:.*|POKER44_MODEL_SHA256:|" scripts/miner/ecosystem.config.cjs
sed -i "0,/POKER44_MODEL_SHA256:/!b;//n;c\\          \"$NEW_SHA\"," scripts/miner/ecosystem.config.cjs 2>/dev/null || true

pm2 delete poker44-miner 2>/dev/null || true
pm2 start scripts/miner/ecosystem.config.cjs --update-env
pm2 save
sleep 5
log "Miner restarted successfully with new model"
tail -5 /root/.pm2/logs/poker44-miner-out.log >> "$LOG_FILE"

log "=== Daily Retrain Complete ==="
