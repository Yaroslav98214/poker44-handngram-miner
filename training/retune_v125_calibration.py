"""Apply v125 within-batch calibration to v124 production artifact."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from poker44.score.scoring import reward
from poker44_ml.inference import Poker44Model
from training.build_dataset import load_benchmark_examples
from training.train_model_v2 import _logit_shift

SRC = REPO / "models/poker44_v124_deploy.joblib"
OUT = REPO / "models/poker44_v125_deploy.joblib"
BENCH = REPO / "hands_generator/evaluation_datas/training_benchmark_v112_full.txt"

LO = 0.14
SPAN = 0.82
BIAS = 0.70


def batch_rank(raw: np.ndarray, lo: float, span: float) -> np.ndarray:
    mn, mx = float(raw.min()), float(raw.max())
    if mx - mn < 1e-9:
        return np.full_like(raw, lo + span / 2.0)
    ranks = (raw - mn) / (mx - mn)
    return lo + ranks * span


def main() -> None:
    artifact = joblib.load(SRC)
    score_remap = {"kind": "batch_rank_v1", "lo": LO, "span": SPAN}

    meta = dict(artifact.get("metadata") or {})
    meta.update(
        {
            "score_logit_bias": BIAS,
            "score_logit_temperature": 1.0,
            "score_remap": score_remap,
            "model_name": "poker44-v125-hybrid",
            "model_version": "1.25.0",
            "framework": "hybrid-lgb-xgb-et-hgram-v125-mixed-batch",
            "calibration_notes": (
                f"batch_rank_v1 lo={LO} span={SPAN} bias={BIAS}; "
                "fixes R2 mixed-batch FPR on require_mixed validator evals"
            ),
            "bias_source": "arena_mixed_batch_rank_retune",
        }
    )
    artifact["metadata"] = meta
    artifact["score_logit_bias"] = BIAS
    artifact["score_logit_temperature"] = 1.0
    artifact["score_remap"] = score_remap
    artifact["model_name"] = "poker44-v125-hybrid"
    artifact["model_version"] = "1.25.0"

    joblib.dump(artifact, OUT, compress=3)
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()

    model = Poker44Model(OUT)
    examples = load_benchmark_examples(str(BENCH), miner_visible=True)
    holdout = [e for e in examples if e.get("source_date") in {"2026-07-06", "2026-07-07"}]
    chunks = [e["chunk"] for e in holdout]
    labels = np.array([e["label"] for e in holdout], dtype=int)
    raw = np.array([float(x) for x in model.debug_score_components(chunks)["raw_scores"]])
    final = _logit_shift(batch_rank(raw, LO, SPAN), BIAS, 1.0)
    rew, det = reward(final, labels)
    print(f"Holdout verify: reward={rew:.4f} fpr={det['fpr']:.3f} recall={det['bot_recall']:.3f}")
    print(f"Saved: {OUT}")
    print(f"SHA256: {sha}")


if __name__ == "__main__":
    main()
