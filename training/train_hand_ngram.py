"""Train the Poker44 hand-level n-gram ensemble (poker44-handngram-supervised).

Trains on the public Poker44 training benchmark fetched from
https://api.poker44.net/api/v1/benchmark. Each hand becomes a bag-of-ngrams
document. A LightGBM + logistic-regression ensemble scores hands; chunk risk
is the mean hand probability passed through absolute sigmoid calibration
(fitted on a held-out date slice) and mapped into [0.04, 0.49].

Usage:
    python -m training.train_hand_ngram \
        --benchmark-path hands_generator/evaluation_datas/training_benchmark_v112_full.txt \
        --output models/poker44_handngram_v2.joblib
"""

from __future__ import annotations

import argparse
import collections
import json
import warnings

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from poker44_ml.hand_ngram import HandNgramEnsemble, hand_ngram_doc

warnings.filterwarnings("ignore")


def _fit_calibration(
    raw_means: np.ndarray,
    labels: np.ndarray,
    *,
    score_low: float = 0.04,
    score_high: float = 0.49,
) -> tuple[float, float]:
    """Pick global center/scale maximizing holdout AP with human scores < 0.49."""
    center = float(np.median(raw_means))
    base_scale = float(max(np.std(raw_means), 1e-4))
    best_ap = -1.0
    best_center, best_scale = center, base_scale
    for scale_mult in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        scale = base_scale * scale_mult
        probe = HandNgramEnsemble(
            {},
            None,
            None,
            stretch_center=center,
            stretch_scale=scale,
            score_low=score_low,
            score_high=score_high,
        )
        scores = np.array([probe._map_absolute(r) for r in raw_means])
        human_max = float(scores[labels == 0].max()) if np.any(labels == 0) else 0.0
        if human_max >= 0.49:
            continue
        ap = average_precision_score(labels, scores)
        if ap > best_ap:
            best_ap = ap
            best_center, best_scale = center, scale
    return best_center, best_scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-token-count", type=int, default=40)
    parser.add_argument("--lgb-weight", type=float, default=0.6)
    parser.add_argument("--recency-ramp", type=float, default=1.5)
    parser.add_argument("--holdout-days", type=int, default=5)
    args = parser.parse_args()

    data = json.load(open(args.benchmark_path))
    groups = data["data"]["chunks"]
    release_version = data.get("data", {}).get("releaseVersion", "unknown")

    hand_docs: list[collections.Counter] = []
    hand_labels: list[int] = []
    hand_dates: list[str] = []
    chunk_lists: list[list] = []
    chunk_labels: list[int] = []
    chunk_dates: list[str] = []
    for group in groups:
        for idx, chunk in enumerate(group["chunks"]):
            label = int(group["groundTruth"][idx])
            chunk_lists.append(chunk)
            chunk_labels.append(label)
            chunk_dates.append(group["sourceDate"])
            for hand in chunk:
                hand_docs.append(hand_ngram_doc(hand))
                hand_labels.append(label)
                hand_dates.append(group["sourceDate"])

    y = np.array(hand_labels)
    dates_arr = np.array(hand_dates)
    chunk_y = np.array(chunk_labels)
    chunk_dates_arr = np.array(chunk_dates)
    dates = sorted(set(hand_dates))
    holdout_dates = set(dates[-args.holdout_days :])
    train_mask = np.array([d not in holdout_dates for d in dates_arr])
    holdout_chunk_mask = np.array([d in holdout_dates for d in chunk_dates_arr])

    counts = collections.Counter(k for doc in hand_docs for k in doc)
    vocabulary = {
        key: i
        for i, key in enumerate(sorted(k for k, c in counts.items() if c >= args.min_token_count))
    }
    x = np.zeros((len(hand_docs), len(vocabulary)), dtype=np.float32)
    for i, doc in enumerate(hand_docs):
        for key, value in doc.items():
            j = vocabulary.get(key)
            if j is not None:
                x[i, j] = value

    date_index = {d: i for i, d in enumerate(dates)}
    weights = np.array(
        [1.0 + args.recency_ramp * date_index[d] / max(len(dates) - 1, 1) for d in dates_arr]
    )

    lgb_model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.7,
        verbose=-1,
        n_jobs=6,
    )
    lgb_model.fit(x[train_mask], y[train_mask], sample_weight=weights[train_mask])
    lr_model = make_pipeline(
        StandardScaler(with_mean=False), LogisticRegression(C=0.3, max_iter=3000)
    )
    lr_model.fit(
        x[train_mask],
        y[train_mask],
        logisticregression__sample_weight=weights[train_mask],
    )

    probe = HandNgramEnsemble(
        vocabulary, lgb_model, lr_model, lgb_weight=args.lgb_weight, score_low=0.0, score_high=1.0
    )
    holdout_raw = []
    for chunk in np.array(chunk_lists, dtype=object)[holdout_chunk_mask]:
        holdout_raw.append(probe._raw_chunk_score(chunk))
    holdout_raw_arr = np.array(holdout_raw)
    holdout_labels = chunk_y[holdout_chunk_mask]
    center, scale = _fit_calibration(holdout_raw_arr, holdout_labels)

    ensemble = HandNgramEnsemble(
        vocabulary,
        lgb_model,
        lr_model,
        lgb_weight=args.lgb_weight,
        aggregation="mean",
        score_low=0.04,
        score_high=0.49,
        stretch_center=center,
        stretch_scale=scale,
    )
    artifact = {
        "models": [ensemble],
        "feature_names": [],
        "metadata": {
            "model_name": "poker44-handngram-supervised",
            "model_version": "1.2.0",
            "training_data": (
                f"Poker44 public training benchmark {release_version} "
                f"({dates[0]}..{dates[-1]}, {len(hand_docs)} hands)"
            ),
            "architecture": (
                "hand-level ngram LightGBM+LogisticRegression ensemble, "
                "chunk mean aggregation, absolute sigmoid calibration"
            ),
            "score_invert": False,
            "score_logit_bias": 0.0,
            "score_logit_temperature": 1.0,
        },
        "model_weights": [1.0],
    }
    joblib.dump(artifact, args.output)

    all_scores = np.array(ensemble.predict_chunk_scores(chunk_lists))
    holdout_scores = all_scores[holdout_chunk_mask]
    in_ap = average_precision_score(chunk_y, all_scores)
    ho_ap = average_precision_score(holdout_labels, holdout_scores)
    human_max = float(all_scores[chunk_y == 0].max())
    print(
        f"saved {args.output} | chunks={len(all_scores)} hands={len(hand_docs)} "
        f"vocab={len(vocabulary)} cal_center={center:.5f} cal_scale={scale:.6f}"
    )
    print(
        f"  in-sample AP={in_ap:.3f} holdout AP={ho_ap:.3f} "
        f"range=[{all_scores.min():.4f},{all_scores.max():.4f}] human_max={human_max:.4f}"
    )


if __name__ == "__main__":
    main()
