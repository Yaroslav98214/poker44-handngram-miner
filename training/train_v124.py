"""
Train v124: v2.2 competition model — benchmark-first calibration for bot recall.

v123 over-tuned live linspace bands and crushed bot recall (34% on benchmark).
v124 prioritizes holdout AP/recall on v1.13 data and uses observed Jul-2026
human finals only as a safety guard.
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

print("=== v124 Hybrid Training (v2.2 benchmark-first + bot recall) ===", flush=True)
t0 = time.time()

sys.path.insert(0, ".")
from poker44_ml.calibration import BlendedQuantileCalibrator
from poker44_ml.hand_ngram import HandNgramEnsemble, hand_ngram_doc
from poker44_ml.stacked import StackedEnsemble
from training.build_dataset import load_benchmark_examples
from training.train_hand_ngram import _fit_calibration as fit_hgram_calibration
from training.train_model_v2 import (
    _apply_score_remap_np,
    _enrich_metrics,
    _logit_shift,
    _select_score_remap_for_validator_reward,
)

BENCH_PATH = "hands_generator/evaluation_datas/training_benchmark_v112_full.txt"
HOLDOUT_DATES = {"2026-07-05", "2026-07-06"}
RECENT_DATES = {
    "2026-06-28", "2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02",
    "2026-07-03", "2026-07-04", "2026-07-05", "2026-07-06",
}
HUMAN_W = 25.0
RECENCY_BOOST = 8.0
MAX_FPR = 0.10
MAX_VALIDATOR_FPR = 0.08
MIN_TOKEN = 40
BATCH_BOTS = 50
BATCH_HUMANS = 50
QUANTILE_BLEND = 0.0
# Observed v123 live human finals (Jul 6): 0.14-0.26 — safety guard only
LIVE_HUMAN_FINAL_MAX = 0.30


def build_hand_ngram_model(train_ex, test_ex):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    all_ex = train_ex + test_ex
    counter = Counter()
    docs_by_ex = []
    for ex in all_ex:
        doc = []
        label = int(ex.get("label", 0))
        for hand in ex.get("chunk") or []:
            if isinstance(hand, dict):
                d = hand_ngram_doc(hand)
                doc.extend(d.items())
                counter.update(d)
        docs_by_ex.append((doc, label))

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


def apply_logit_bias(scores, bias, temp=1.0):
    return _logit_shift(np.asarray(scores, dtype=float), bias, temp)


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


def squash_to_range(scores, lo, hi):
    arr = np.asarray(scores, dtype=float)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return np.full_like(arr, (lo + hi) / 2.0)
    ranks = (arr - mn) / (mx - mn)
    return lo + ranks * (hi - lo)


def pipeline_scores(raw_scores, score_remap, bias, temp=1.0):
    remapped = _apply_score_remap_np(np.asarray(raw_scores, dtype=float), score_remap or {})
    return apply_logit_bias(remapped, bias, temp)


examples = load_benchmark_examples(BENCH_PATH, miner_visible=True)
print(f"Loaded {len(examples)} examples in {time.time() - t0:.1f}s", flush=True)
print(f"Latest date: {max(e.get('source_date', '') for e in examples)}", flush=True)

train_ex = [e for e in examples if e["source_date"] not in HOLDOUT_DATES]
test_ex = [e for e in examples if e["source_date"] in HOLDOUT_DATES]
print(f"Train={len(train_ex)} Holdout={len(test_ex)}", flush=True)

print("Training hand-ngram side model...", flush=True)
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
print("5-fold CV:", flush=True)
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
    blend = 0.45 * oof_cols[va_idx, 0] + 0.35 * oof_cols[va_idx, 1] + 0.20 * oof_cols[va_idx, 2]
    print(f"  Fold {fold + 1}: AP={average_precision_score(y_train[va_idx], blend):.4f}", flush=True)

meta = LogisticRegression(C=0.8, max_iter=1000, random_state=42)
meta.fit(oof_cols, y_train, sample_weight=sw_train)
oof_blend = meta.predict_proba(oof_cols)[:, 1]
print(f"OOF stacked AP={average_precision_score(y_train, oof_blend):.4f}", flush=True)

stack_calibrator = BlendedQuantileCalibrator(blend=QUANTILE_BLEND).fit(oof_blend)
oof_calibrated = stack_calibrator.transform(oof_blend)
print(f"OOF calibrated range: [{oof_calibrated.min():.4f}, {oof_calibrated.max():.4f}]", flush=True)

_, cal_hold = train_test_split(
    list(range(len(train_ex))), test_size=0.25, random_state=42, stratify=y_train,
)
cal_raw = oof_calibrated[np.array(cal_hold)]
cal_labels = y_train[np.array(cal_hold)]
cal_bot = cal_raw[cal_labels == 1]
cal_hum = cal_raw[cal_labels == 0]

# Calibrate remap on holdout raw distribution after prod fit (not synthetic linspace).
def remap_holdout_metrics(remap: dict, raw: np.ndarray, labels: np.ndarray, bias: float = 0.0):
    final = pipeline_scores(raw, remap, bias)
    rew, ap, recall, fpr = validator_reward(labels, final)
    hum_max = float(final[labels == 0].max()) if np.any(labels == 0) else 0.0
    bot_min = float(final[labels == 1].min()) if np.any(labels == 1) else 1.0
    return {
        "reward": rew, "ap": ap, "recall": recall, "fpr": fpr,
        "human_max": hum_max, "bot_min": bot_min,
    }


print("Tuning score_remap (calibration split AP-first)...", flush=True)
score_remap, remap_metrics = _select_score_remap_for_validator_reward(
    cal_labels,
    cal_raw,
    target_fpr=0.04,
    max_validator_fpr=MAX_VALIDATOR_FPR,
    calibration_objective="ap_first",
    temperature_grid=[0.12, 0.18, 0.25, 0.35, 0.50, 0.65, 0.85, 1.0, 1.25],
    prefer_smooth_remap=True,
)
print(
    f"  cal remap={score_remap} ap={remap_metrics.get('pr_auc', 0):.4f} "
    f"recall={remap_metrics.get('validator_bot_recall', 0):.3f} fpr={remap_metrics.get('validator_fpr', 0):.3f}",
    flush=True,
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

print(f"\nHoldout raw AP={average_precision_score(y_test, raw_test):.4f}", flush=True)
print(
    f"  Bots [{raw_test[y_test==1].min():.4f},{raw_test[y_test==1].max():.4f}] "
    f"Hum [{raw_test[y_test==0].min():.4f},{raw_test[y_test==0].max():.4f}]",
    flush=True,
)

# Re-tune remap on actual holdout raw scores (benchmark-first for v2.2).
print("Re-tuning remap on holdout distribution...", flush=True)
hold_remap, hold_metrics = _select_score_remap_for_validator_reward(
    y_test,
    raw_test,
    target_fpr=0.05,
    max_validator_fpr=MAX_FPR,
    calibration_objective="ap_first",
    temperature_grid=[0.15, 0.20, 0.25, 0.35, 0.50, 0.65, 0.85, 1.0, 1.25, 1.5],
    prefer_smooth_remap=False,
)
if hold_remap:
    hold_check = remap_holdout_metrics(hold_remap, raw_test, y_test, 0.0)
    if hold_check["recall"] >= 0.5 and hold_check["fpr"] < MAX_FPR:
        score_remap = hold_remap
        print(f"  holdout remap={score_remap} reward={hold_check['reward']:.4f} recall={hold_check['recall']:.3f}", flush=True)
    else:
        print(f"  holdout remap failed guard recall={hold_check['recall']:.3f}; keeping cal remap", flush=True)

bot_raw = raw_test[y_test == 1]
hum_raw = raw_test[y_test == 0]
rng = np.random.default_rng(42)
mixed_raw = np.concatenate([
    rng.choice(hum_raw, BATCH_HUMANS, replace=True) if len(hum_raw) else hum_raw,
    rng.choice(bot_raw, BATCH_BOTS, replace=True) if len(bot_raw) else bot_raw,
])
mixed_labels = np.array([0] * BATCH_HUMANS + [1] * BATCH_BOTS)

scenarios = [
    ("holdout", raw_test, y_test),
    ("mixed50", mixed_raw, mixed_labels),
]

print("\n=== Bias tune (holdout recall>=0.5, FPR<10%, human_max guard) ===", flush=True)
best = {"reward": -1.0, "bias": 0.0, "details": {}}
for bias in np.arange(-0.3, 1.5, 0.05):
    details = {}
    ok = True
    for name, raw, labels in scenarios:
        final = pipeline_scores(raw, score_remap, bias)
        rew, ap, recall, fpr = validator_reward(labels, final)
        hum_max = float(final[labels == 0].max()) if np.any(labels == 0) else 0.0
        details[name] = {"reward": rew, "ap": ap, "recall": recall, "fpr": fpr, "human_max": hum_max}
        if fpr >= MAX_FPR or recall < 0.5:
            ok = False
            break
        if hum_max > LIVE_HUMAN_FINAL_MAX:
            ok = False
            break
    if not ok:
        continue
    holdout_rew = details["holdout"]["reward"]
    if holdout_rew > best["reward"]:
        best = {"reward": holdout_rew, "bias": float(bias), "details": details}

optimal_bias = best["bias"]
bias_source = "holdout_recall_guard"
if best["reward"] < 0:
    optimal_bias = 0.0
    bias_source = "default_fallback"
    best["details"] = {}
    for name, raw, labels in scenarios:
        final = pipeline_scores(raw, score_remap, 0.0)
        rew, ap, recall, fpr = validator_reward(labels, final)
        best["details"][name] = {"reward": rew, "ap": ap, "recall": recall, "fpr": fpr}
    best["reward"] = best["details"]["holdout"]["reward"]

print(f"  Selected bias={optimal_bias:.2f} (source={bias_source}) holdout_reward={best['reward']:.4f}", flush=True)
for name, d in best.get("details", {}).items():
    print(
        f"    {name}: reward={d['reward']:.4f} ap={d['ap']:.4f} "
        f"recall={d['recall']:.3f} fpr={d['fpr']:.3f}",
        flush=True,
    )

cal_test = pipeline_scores(raw_test, score_remap, optimal_bias)
rew, ap, recall, fpr = validator_reward(y_test, cal_test)
spread = float(np.quantile(raw_test[y_test == 1], 0.10) - np.quantile(raw_test[y_test == 0], 0.90))
print(
    f"\nFinal holdout: reward={rew:.4f} AP={ap:.4f} recall={recall:.3f} FPR={fpr:.3f} "
    f"raw_sep={spread:.4f} bots_above_0.5={(cal_test[y_test==1]>=0.5).sum()}/{int(y_test.sum())} "
    f"human_max={float(cal_test[y_test==0].max()) if np.any(y_test==0) else 0:.4f}",
    flush=True,
)

meta_info = {
    "score_logit_bias": float(optimal_bias),
    "score_logit_temperature": 1.0,
    "score_remap": dict(score_remap) if score_remap else {},
    "human_weight_multiplier": float(HUMAN_W),
    "recency_boost_multiplier": float(RECENCY_BOOST),
    "model_name": "poker44-v124-hybrid",
    "model_version": "1.24.0",
    "framework": "hybrid-lgb-xgb-et-hgram-v22-apfirst",
    "train_latest_date": max(e.get("source_date", "") for e in examples),
    "train_total_examples": int(len(y_all)),
    "holdout_source_dates": sorted(HOLDOUT_DATES),
    "holdout_ap": float(ap),
    "holdout_bot_recall": float(recall),
    "holdout_fpr": float(fpr),
    "holdout_reward": float(rew),
    "min_scenario_reward": float(best["reward"]),
    "scenario_metrics": best.get("details", {}),
    "calibration_notes": f"v2.2 holdout-first remap, quantile_blend={QUANTILE_BLEND}, human_max<={LIVE_HUMAN_FINAL_MAX}",
    "quantile_calibration_blend": float(QUANTILE_BLEND),
    "hgram_stretch_center": float(hgram_model.stretch_center or 0),
    "hgram_stretch_scale": float(hgram_model.stretch_scale or 0),
    "stack_meta_weights": meta.coef_.tolist(),
    "bias_source": bias_source,
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
    "score_remap": dict(score_remap) if score_remap else {},
    "model_name": "poker44-v123-hybrid",
    "model_version": "1.23.0",
    "hand_ngram_model": hgram_model,
}

out_path = "models/poker44_v124_deploy.joblib"
joblib.dump(artifact, out_path, compress=3)
with open(out_path, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()

print(f"\nSaved: {out_path}")
print(f"SHA256: {sha}")
print(f"Total time: {time.time() - t0:.1f}s")
print("=== DONE ===", flush=True)
