#!/usr/bin/env bash
# Daily pipeline for Poker44 UID 208: fetch benchmark + retrain v123 (R1 threshold_logit).
set -euo pipefail

REPO=/root/Poker44-top-miner
PYTHON="$REPO/miner_env/bin/python3"
LOG_FILE="$REPO/logs/daily_retrain.log"
ECOSYSTEM="$REPO/scripts/miner/ecosystem.config.cjs"
MODEL_PATH="$REPO/models/poker44_v123_deploy.joblib"
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

status = requests.get(API_BASE, timeout=30).json()["data"]
latest_date = status["latestSourceDate"]
print(f"API latest date: {latest_date}")

with open(BENCH_FILE) as f:
    existing_data = json.load(f)
existing_chunks = existing_data["data"]["chunks"]
existing_dates = set(c.get("sourceDate", "") for c in existing_chunks)
print(f"Existing dates: {sorted(existing_dates)[-3:]}, total chunks: {len(existing_chunks)}")

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

all_chunks = existing_chunks + new_chunks
updated_data = {"data": {"releaseVersion": "v1.12", "chunks": all_chunks}}
with open(BENCH_FILE, "w") as f:
    json.dump(updated_data, f)
print(f"Saved: {len(all_chunks)} total chunks (added {len(new_chunks)} new)")
PYEOF

# 2. Retrain v123 (R1-era threshold_logit pipeline — no batch_rank)
log "Running v123 hybrid model training..."
$PYTHON -m training.train_v123 2>&1 | tee -a "$LOG_FILE"
RETRAIN_EXIT=$?

if [ $RETRAIN_EXIT -ne 0 ]; then
    log "ERROR: Retraining failed with exit code $RETRAIN_EXIT"
    exit 1
fi

# 3. Quality gate: threshold_logit only, holdout FPR below validator cliff
read -r NEW_SHA HOLDOUT_FPR HOLDOUT_REWARD REMAP_KIND <<< "$($PYTHON - << PY
import hashlib, joblib, sys
path = "$MODEL_PATH"
with open(path, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()
art = joblib.load(path)
meta = art.get("metadata") or {}
remap = art.get("score_remap") or meta.get("score_remap") or {}
kind = str(remap.get("kind", ""))
fpr = float(meta.get("holdout_fpr", 1.0))
reward = float(meta.get("holdout_reward", 0.0))
print(sha, f"{fpr:.4f}", f"{reward:.4f}", kind)
PY
)"

log "New model SHA256: $NEW_SHA"
log "Holdout FPR=$HOLDOUT_FPR reward=$HOLDOUT_REWARD remap=$REMAP_KIND"

if [ "$REMAP_KIND" != "threshold_logit_v1" ]; then
    log "ERROR: Refusing deploy — expected threshold_logit_v1, got $REMAP_KIND"
    exit 1
fi

$PYTHON - << PY
fpr = float("$HOLDOUT_FPR")
if fpr > 0.10:
    raise SystemExit(f"Holdout FPR {fpr:.3f} exceeds 10% validator cliff")
PY

CURRENT_SHA=$($PYTHON - << PY
import re, pathlib
text = pathlib.Path("$ECOSYSTEM").read_text()
m = re.search(r'POKER44_MODEL_ARTIFACT_SHA256:\s*"([a-f0-9]{64})"', text)
print(m.group(1) if m else "")
PY
)
log "Current deployed SHA256: $CURRENT_SHA"

if [ "$NEW_SHA" = "$CURRENT_SHA" ]; then
    log "Model unchanged, skipping deployment."
    log "=== Daily Retrain Complete ==="
    exit 0
fi

# 4. Update ecosystem SHA fields safely and restart
log "Deploying new v123 model..."
COMMIT=$(git rev-parse HEAD)
$PYTHON - << PY
import pathlib, re
path = pathlib.Path("$ECOSYSTEM")
text = path.read_text()
text = re.sub(
    r'POKER44_MODEL_SHA256:\s*"[^"]*"',
    f'POKER44_MODEL_SHA256: "{("$NEW_SHA")}"',
    text,
)
text = re.sub(
    r'POKER44_MODEL_ARTIFACT_SHA256:\s*"[^"]*"',
    f'POKER44_MODEL_ARTIFACT_SHA256: "{("$NEW_SHA")}"',
    text,
)
text = re.sub(
    r'POKER44_MODEL_REPO_COMMIT:\s*"[^"]*"',
    f'POKER44_MODEL_REPO_COMMIT: "{("$COMMIT")}"',
    text,
)
path.write_text(text)
PY

pm2 delete poker44-miner 2>/dev/null || true
pm2 start "$ECOSYSTEM" --update-env
pm2 save
sleep 5
log "Miner restarted successfully with new v123 model"
tail -5 /root/.pm2/logs/poker44-miner-out.log >> "$LOG_FILE"

log "=== Daily Retrain Complete ==="
