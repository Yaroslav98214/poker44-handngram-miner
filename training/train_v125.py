"""
Train v125: fix R2 mixed-batch FPR collapse.

Root cause (Jul 7 dashboard): R1=0.581 -> R2=0.500 while top miners held ~0.55.
Validators send require_mixed=True batches. v124 global threshold_logit maps
overlapping live raw scores so arena humans (raw up to 0.23) land above 0.5
alongside bots -> 96-98% FPR on simulated R2 mixed batches.

v125: stronger human-weighted retrain + batch_rank_v1 calibration tuned on
holdout and arena-shifted mixed simulations from observed live raw ranges.
"""
import hashlib
import sys
import time
from collections import Counter

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

print("=== v125 Hybrid Training (mixed-batch FPR fix) ===", flush=True)
t0 = time.time()

sys.path.insert(0, ".")
from poker44.score.scoring import reward as subnet_reward
from poker44_ml.calibration import BlendedQuantileCalibrator
from poker44_ml.hand_ngram import HandNgramEnsemble, hand_ngram_doc
from poker44_ml.stacked import StackedEnsemble
from training.build_dataset import load_benchmark_examples
from training.train_hand_ngram import _fit_calibration as fit_hgram_calibration
from training.train_model_v2 import _enrich_metrics, _logit_shift

BENCH_PATH = "hands_generator/evaluation_datas/training_benchmark_v112_full.txt"
HOLDOUT_DATES = {"2026-07-06", "2026-07-07"}
RECENT_DATES = {
    "2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02",
    "2026-07-03", "2026-07-04", "2026-07-05", "2026-07-06", "2026-07-07",
}
HUMAN_W = 45.0
RECENCY_BOOST = 10.0
MAX_FPR = 0.10
MIN_TOKEN = 40
QUANTILE_BLEND = 0.0


def build_hand_ngram_model(train_ex, test_ex):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    counter = Counter()
    for ex in train_ex + test_ex:
        for hand in ex.get("chunk") or []:
            if isinstance(hand, dict):
                counter.update(hand_ngram_doc(hand))
    vocab = {k: i for i, k in enumerate(k for k, c in counter.items() if c >= MIN_TOKEN)}

    def ex_to_row(doc_items):
        row = np.zeros(len(vocab), dtype=np.float32)
        for key, val in doc_items:
            j = vocab.get(key)
            if j is not None:
                row[j] += val
        return row

    train_rows, train_labels = [], []
    for ex in train_ex:
        items = []
        for hand in ex.get("chunk") or []:
            if isinstance(hand, dict):
                items.extend(hand_ngram_doc(hand).items())
        if items:
            train_rows.append(ex_to_row(items))
            train_labels.append(int(ex.get("label", 0)))

    x_tr = np.array(train_rows)
    y_tr = np.array(train_labels, dtype=int)
    sw = np.where(y_tr == 0, HUMAN_W, 1.0).astype(np.float64)
    sw /= sw.mean()

    lgb_h = lgb.LGBMClassifier(
        n_estimators=500, num_leaves=31, learning_rate=0.05,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=4, verbosity=-1,
    )
    lgb_h.fit(x_tr, y_tr, sample_weight=sw)
    lr_h = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=500, C=0.5, random_state=42),
    )
    lr_h.fit(x_tr, y_tr, logisticregression__sample_weight=sw)

    hold_raw, hold_labels = [], []
    for ex in test_ex:
        items = []
        for hand in ex.get("chunk") or []:
            if isinstance(hand, dict):
                items.extend(hand_ngram_doc(hand).items())
        if not items:
            continue
        x = ex_to_row(items).reshape(1, -1)
        probs = 0.6 * lgb_h.predict_proba(x)[:, 1] + 0.4 * lr_h.predict_proba(x)[:, 1]
        hold_raw.append(float(probs[0]))
        hold_labels.append(int(ex.get("label", 0)))

    center, scale = fit_hgram_calibration(
        np.array(hold_raw), np.array(hold_labels), score_low=0.04, score_high=0.49,
    )
    return HandNgramEnsemble(
        vocab, lgb_h, lr_h, lgb_weight=0.6,
        stretch_center=center, stretch_scale=scale,
    )


def hgram_chunk_features(hgram_model, chunk):
    hands = [h for h in (chunk or []) if isinstance(h, dict)]
    if not hands:
        return {"hgram_mean": 0.0, "hgram_max": 0.0, "hgram_std": 0.0}
    probs = hgram_model._hand_probs(hands)
    return {
        "hgram_mean": float(np.mean(probs)),
        "hgram_max": float(np.max(probs)),
        "hgram_std": float(np.std(probs)) if len(probs) > 1 else 0.0,
    }


def apply_batch_rank_np(raw: np.ndarray, lo: float, span: float) -> np.ndarray:
    arr = np.asarray(raw, dtype=float)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return np.full_like(arr, lo + span / 2.0)
    ranks = (arr - mn) / (mx - mn)
    return lo + ranks * span


def pipeline_scores(raw: np.ndarray, lo: float, span: float, bias: float) -> np.ndarray:
    ranked = apply_batch_rank_np(raw, lo, span)
    return _logit_shift(ranked, bias, 1.0)


def validator_reward(y_true, y_pred_scores):
    labels = np.asarray(y_true, dtype=int).tolist()
    scores = np.asarray(y_pred_scores, dtype=float).tolist()
    metrics = _enrich_metrics(labels, scores)
    return (
        float(metrics.get("validator_reward", 0.0)),
        float(metrics.get("pr_auc", 0.0)),
        float(metrics.get("validator_bot_recall", 0.0)),
        float(metrics.get("validator_fpr", 1.0)),
    )


def arena_shifted_mixed(raw_holdout: np.ndarray, labels: np.ndarray, rng: np.random.Generator):
    """Simulate arena human raw inflation seen in live Jul 6-7 logs."""
    hum_idx = np.where(labels == 0)[0]
    bot_idx = np.where(labels == 1)[0]
    n = min(50, len(hum_idx), len(bot_idx))
    if n < 10:
        return raw_holdout, labels
    h = rng.choice(hum_idx, n, replace=False)
    b = rng.choice(bot_idx, n, replace=False)
    mixed_raw = raw_holdout[np.concatenate([h, b])].copy()
    mixed_labels = labels[np.concatenate([h, b])]
    # Push a subset of humans toward live arena raw band (0.14-0.22).
    hum_part = mixed_raw[:n]
    boost = rng.uniform(0.03, 0.10, size=n)
    order = np.argsort(hum_part)
    for rank, idx in enumerate(order):
        if rank >= n // 2:
            hum_part[idx] = min(0.24, hum_part[idx] + boost[idx])
    mixed_raw[:n] = hum_part
    return mixed_raw, mixed_labels


examples = load_benchmark_examples(BENCH_PATH, miner_visible=True)
print(f"Loaded {len(examples)} examples in {time.time() - t0:.1f}s", flush=True)

train_ex = [e for e in examples if e["source_date"] not in HOLDOUT_DATES]
test_ex = [e for e in examples if e["source_date"] in HOLDOUT_DATES]
print(f"Train={len(train_ex)} Holdout={len(test_ex)}", flush=True)

hgram_model = build_hand_ngram_model(train_ex, test_ex)
for ex in examples:
    ex["features"].update(hgram_chunk_features(hgram_model, ex.get("chunk")))

feat_names = sorted(examples[0]["features"].keys())


def featurize(exs):
    return np.array(
        [[float(e["features"].get(n, 0)) for n in feat_names] for e in exs],
        dtype=np.float32,
    )


def sample_weights_for(exs):
    sw = np.ones(len(exs), dtype=np.float64)
    for i, e in enumerate(exs):
        if e.get("label") == 0:
            sw[i] = HUMAN_W
        if e.get("source_date") in RECENT_DATES:
            sw[i] *= RECENCY_BOOST
    return sw / sw.mean()


X_train = featurize(train_ex)
y_train = np.array([e["label"] for e in train_ex])
X_test = featurize(test_ex)
y_test = np.array([e["label"] for e in test_ex])
sw_train = sample_weights_for(train_ex)

lgb_params = {
    "objective": "binary", "n_estimators": 1200, "num_leaves": 63,
    "learning_rate": 0.02, "min_child_samples": 3, "subsample": 0.85,
    "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": 42, "n_jobs": 4, "verbosity": -1,
}
xgb_params = {
    "n_estimators": 1000, "max_depth": 7, "learning_rate": 0.025,
    "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1,
    "reg_lambda": 1.0, "random_state": 42, "n_jobs": 4,
    "objective": "binary:logistic", "eval_metric": "logloss",
    "verbosity": 0, "tree_method": "hist",
}
et_params = {
    "n_estimators": 400, "max_depth": 12, "min_samples_leaf": 2,
    "max_features": "sqrt", "random_state": 42, "n_jobs": 4,
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_cols = np.zeros((len(y_train), 3))
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
    w = sw_train[tr_idx]
    m_lgb = lgb.LGBMClassifier(**lgb_params)
    m_lgb.fit(X_train[tr_idx], y_train[tr_idx], sample_weight=w)
    m_xgb = xgb.XGBClassifier(**xgb_params)
    m_xgb.fit(X_train[tr_idx], y_train[tr_idx], sample_weight=w)
    m_et = ExtraTreesClassifier(**et_params)
    m_et.fit(X_train[tr_idx], y_train[tr_idx], sample_weight=w)
    oof_cols[va_idx, 0] = m_lgb.predict_proba(X_train[va_idx])[:, 1]
    oof_cols[va_idx, 1] = m_xgb.predict_proba(X_train[va_idx])[:, 1]
    oof_cols[va_idx, 2] = m_et.predict_proba(X_train[va_idx])[:, 1]

meta = LogisticRegression(C=0.8, max_iter=1000, random_state=42)
meta.fit(oof_cols, y_train, sample_weight=sw_train)
stack_calibrator = BlendedQuantileCalibrator(blend=QUANTILE_BLEND).fit(
    meta.predict_proba(oof_cols)[:, 1]
)

X_all = featurize(examples)
y_all = np.array([e["label"] for e in examples])
sw_all = sample_weights_for(examples)

prod_lgb = lgb.LGBMClassifier(**lgb_params)
prod_lgb.fit(X_all, y_all, sample_weight=sw_all)
prod_xgb = xgb.XGBClassifier(**xgb_params)
prod_xgb.fit(X_all, y_all, sample_weight=sw_all)
prod_et = ExtraTreesClassifier(**et_params)
prod_et.fit(X_all, y_all, sample_weight=sw_all)

raw_test = meta.predict_proba(np.column_stack([
    prod_lgb.predict_proba(X_test)[:, 1],
    prod_xgb.predict_proba(X_test)[:, 1],
    prod_et.predict_proba(X_test)[:, 1],
]))[:, 1]
raw_test = stack_calibrator.transform(raw_test)
print(f"Holdout raw AP={average_precision_score(y_test, raw_test):.4f}", flush=True)
print(
    f"  raw human max={raw_test[y_test==0].max():.4f} bot min={raw_test[y_test==1].min():.4f}",
    flush=True,
)

rng = np.random.default_rng(42)
arena_raw, arena_labels = arena_shifted_mixed(raw_test, y_test, rng)

print("\n=== batch_rank calibration grid ===", flush=True)
best = None
for lo in np.arange(0.04, 0.22, 0.02):
    for span in np.arange(0.50, 0.92, 0.04):
        for bias in np.arange(0.4, 1.4, 0.1):
            hold_final = pipeline_scores(raw_test, lo, span, bias)
            hold_rew, _, hold_recall, hold_fpr = validator_reward(y_test, hold_final)

            arena_final = pipeline_scores(arena_raw, lo, span, bias)
            arena_rew, _, arena_recall, arena_fpr = validator_reward(arena_labels, arena_final)

            if hold_fpr >= MAX_FPR or hold_recall < 0.88:
                continue
            if arena_fpr >= MAX_FPR or arena_recall < 0.80:
                continue

            bot_only = raw_test[y_test == 1]
            bot_final = pipeline_scores(bot_only, lo, span, bias)
            _, _, bot_recall, _ = validator_reward(np.ones(len(bot_only)), bot_final)
            if bot_recall < 0.92:
                continue

            score = arena_rew * 0.55 + hold_rew * 0.45
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "lo": float(lo),
                    "span": float(span),
                    "bias": float(bias),
                    "hold_rew": float(hold_rew),
                    "hold_fpr": float(hold_fpr),
                    "hold_recall": float(hold_recall),
                    "arena_rew": float(arena_rew),
                    "arena_fpr": float(arena_fpr),
                    "arena_recall": float(arena_recall),
                    "bot_recall": float(bot_recall),
                }

if best is None:
    raise RuntimeError("No calibration config passed mixed-batch constraints")

score_remap = {"kind": "batch_rank_v1", "lo": best["lo"], "span": best["span"]}
optimal_bias = best["bias"]

print(
    f"  lo={best['lo']:.2f} span={best['span']:.2f} bias={best['bias']:.1f}",
    flush=True,
)
print(
    f"  holdout rew={best['hold_rew']:.3f} fpr={best['hold_fpr']:.3f} recall={best['hold_recall']:.3f}",
    flush=True,
)
print(
    f"  arena rew={best['arena_rew']:.3f} fpr={best['arena_fpr']:.3f} recall={best['arena_recall']:.3f}",
    flush=True,
)
print(f"  bot-only recall={best['bot_recall']:.3f}", flush=True)

meta_info = {
    "score_logit_bias": float(optimal_bias),
    "score_logit_temperature": 1.0,
    "score_remap": score_remap,
    "human_weight_multiplier": float(HUMAN_W),
    "recency_boost_multiplier": float(RECENCY_BOOST),
    "model_name": "poker44-v125-hybrid",
    "model_version": "1.25.0",
    "framework": "hybrid-lgb-xgb-et-hgram-v125-mixed-batch",
    "train_latest_date": max(e.get("source_date", "") for e in examples),
    "train_total_examples": int(len(y_all)),
    "holdout_source_dates": sorted(HOLDOUT_DATES),
    "holdout_reward": float(best["hold_rew"]),
    "arena_mixed_reward": float(best["arena_rew"]),
    "calibration_notes": (
        f"batch_rank_v1 lo={best['lo']:.2f} span={best['span']:.2f} "
        f"for require_mixed validator batches; fixes R2 FPR collapse"
    ),
    "quantile_calibration_blend": float(QUANTILE_BLEND),
    "hgram_stretch_center": float(hgram_model.stretch_center or 0),
    "hgram_stretch_scale": float(hgram_model.stretch_scale or 0),
    "stack_meta_weights": meta.coef_.tolist(),
    "bias_source": "arena_mixed_batch_rank",
}

stacked = StackedEnsemble(
    base_models=[prod_lgb, prod_xgb, prod_et],
    meta_model=meta,
    calibrator=stack_calibrator,
)

artifact = {
    "models": [stacked],
    "model_weights": [1.0],
    "feature_names": feat_names,
    "metadata": meta_info,
    "score_logit_bias": float(optimal_bias),
    "score_logit_temperature": 1.0,
    "score_remap": score_remap,
    "model_name": "poker44-v125-hybrid",
    "model_version": "1.25.0",
    "hand_ngram_model": hgram_model,
}

out_path = "models/poker44_v125_deploy.joblib"
joblib.dump(artifact, out_path, compress=3)
sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
print(f"\nSaved: {out_path}")
print(f"SHA256: {sha}")
print(f"Total time: {time.time() - t0:.1f}s")
print("=== DONE ===", flush=True)
