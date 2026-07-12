"""Patch v123 calibration for current live arena raw score bands (Jul 2026 R2)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import scipy.special as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from poker44.score.scoring import reward
from poker44_ml.inference import Poker44Model
from training.build_dataset import load_benchmark_examples
from training.train_model_v2 import _apply_score_remap_np

MODEL_PATH = REPO / "models" / "poker44_v123_deploy.joblib"
BENCH_PATH = REPO / "hands_generator/evaluation_datas/training_benchmark_v112_full.txt"
HOLDOUT_DATES = {"2026-07-02", "2026-07-03"}
MAX_FPR = 0.10

# Observed live PM2 raw bands (Jul 11-12 homogeneous batches)
LIVE_HUMAN_RAW_LO, LIVE_HUMAN_RAW_HI = 0.065, 0.20
LIVE_BOT_RAW_LO, LIVE_BOT_RAW_HI = 0.22, 0.50
BATCH = 50


def apply_logit_bias(scores: np.ndarray, bias: float) -> np.ndarray:
    s = np.clip(scores.astype(float), 1e-6, 1 - 1e-6)
    if abs(bias) < 1e-12:
        return s
    return sp.expit(sp.logit(s) + bias)


def pipeline(raw: np.ndarray, remap: dict, bias: float) -> np.ndarray:
    remapped = _apply_score_remap_np(raw, remap)
    return apply_logit_bias(remapped, bias)


def metrics(raw: np.ndarray, labels: np.ndarray, remap: dict, bias: float) -> dict:
    final = pipeline(raw, remap, bias)
    rew, met = reward(final, labels)
    return {
        "reward": float(rew),
        "recall": float(met["bot_recall"]),
        "fpr": float(met["fpr"]),
        "human_max": float(final[~labels].max()) if (~labels).any() else 0.0,
        "bot_min": float(final[labels].min()) if labels.any() else 0.0,
    }


def main() -> None:
    live_raw = np.concatenate([
        np.linspace(LIVE_HUMAN_RAW_LO, LIVE_HUMAN_RAW_HI, BATCH),
        np.linspace(LIVE_BOT_RAW_LO, LIVE_BOT_RAW_HI, BATCH),
    ])
    live_labels = np.array([0] * BATCH + [1] * BATCH, dtype=bool)

    examples = load_benchmark_examples(str(BENCH_PATH), miner_visible=True)
    holdout = [e for e in examples if e.get("source_date") in HOLDOUT_DATES]
    model = Poker44Model(MODEL_PATH)
    hold_raw = np.array(
        model.debug_score_components([e["chunk"] for e in holdout])["raw_scores"],
        dtype=float,
    )
    hold_labels = np.array([int(e["label"]) for e in holdout], dtype=bool)

    best = None
    for threshold in [float(x) for x in np.linspace(0.08, 0.28, 21)]:
        for temperature in [0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.25]:
            remap = {
                "kind": "threshold_logit_v1",
                "threshold": threshold,
                "temperature": temperature,
            }
            for bias in [float(x) for x in np.arange(-0.10, 0.81, 0.05)]:
                live_met = metrics(live_raw, live_labels, remap, bias)
                if live_met["fpr"] >= MAX_FPR or live_met["recall"] < 0.50:
                    continue
                if live_met["human_max"] >= 0.50 or live_met["bot_min"] < 0.50:
                    continue
                hold_met = metrics(hold_raw, hold_labels, remap, bias)
                if hold_met["fpr"] >= MAX_FPR:
                    continue
                key = (live_met["reward"], hold_met["reward"], live_met["recall"], -live_met["human_max"])
                if best is None or key > best[0]:
                    best = (key, remap, bias, live_met, hold_met)

    if best is None:
        raise SystemExit("No calibration passed live + holdout gates")

    _, score_remap, optimal_bias, live_met, hold_met = best
    print("Selected calibration:")
    print(f"  score_remap={score_remap}")
    print(f"  score_logit_bias={optimal_bias:.3f}")
    print(
        f"  live reward={live_met['reward']:.4f} recall={live_met['recall']:.3f} "
        f"human_max={live_met['human_max']:.3f} bot_min={live_met['bot_min']:.3f}"
    )
    print(f"  holdout reward={hold_met['reward']:.4f} recall={hold_met['recall']:.3f} fpr={hold_met['fpr']:.3f}")

    artifact = joblib.load(MODEL_PATH)
    artifact["score_remap"] = dict(score_remap)
    artifact["score_logit_bias"] = float(optimal_bias)
    artifact["score_logit_temperature"] = 1.0
    meta = dict(artifact.get("metadata") or {})
    meta.update({
        "score_remap": dict(score_remap),
        "score_logit_bias": float(optimal_bias),
        "score_logit_temperature": 1.0,
        "calibration_notes": (
            "Retuned Jul 12 live arena raw bands "
            f"human_raw={LIVE_HUMAN_RAW_LO:.3f}-{LIVE_HUMAN_RAW_HI:.3f} "
            f"bot_raw={LIVE_BOT_RAW_LO:.3f}-{LIVE_BOT_RAW_HI:.3f}"
        ),
        "holdout_reward": float(hold_met["reward"]),
        "live_mixed_reward": float(live_met["reward"]),
    })
    artifact["metadata"] = meta
    joblib.dump(artifact, MODEL_PATH, compress=3)
    sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    print(f"Saved {MODEL_PATH}")
    print(f"SHA256: {sha}")


if __name__ == "__main__":
    main()
