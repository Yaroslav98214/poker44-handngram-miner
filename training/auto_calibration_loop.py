from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import requests

from poker44.score.scoring import reward
from poker44_ml.inference import Poker44Model
from training.build_dataset import load_benchmark_examples, resolve_benchmark_paths
from training.evaluate_model import _evaluate_examples


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "hands_generator" / "evaluation_datas"
BENCHMARK_PATH = EVAL_DIR / "training_benchmark.txt"
AUTO_DIR = REPO_ROOT / "models" / "auto"
AUTO_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "https://api.poker44.net/api/v1/benchmark"


@dataclass
class Metrics:
    reward: float
    fpr: float
    ap: float
    recall: float


def _log(message: str) -> None:
    print(f"[auto-calibrate] {message}", flush=True)


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=env)


def _fetch_latest_benchmark() -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    status = requests.get(API_BASE, timeout=30).json()["data"]
    source_date = status["latestSourceDate"]
    release = status.get("releaseVersion", "latest")

    rows = requests.get(
        f"{API_BASE}/chunks",
        params={"sourceDate": source_date},
        timeout=60,
    ).json()["data"]["chunks"]

    payload = {
        "data": {
            "releaseVersion": release,
            "sourceDate": source_date,
            "chunks": rows,
        }
    }
    versioned = EVAL_DIR / f"training_benchmark_{release}_{source_date}.txt"
    versioned.write_text(json.dumps(payload), encoding="utf-8")
    BENCHMARK_PATH.write_text(json.dumps(payload), encoding="utf-8")
    _log(
        f"benchmark synced release={release} source_date={source_date} "
        f"groups={len(rows)}"
    )
    return BENCHMARK_PATH


def _load_examples(path: Path) -> list[dict[str, Any]]:
    paths = resolve_benchmark_paths(path)
    examples = load_benchmark_examples(paths)
    _log(f"loaded examples={len(examples)} from={path}")
    return examples


def _eval_model(model_path: Path, examples: list[dict[str, Any]]) -> Metrics:
    model = Poker44Model(model_path)
    data = _evaluate_examples(model, examples, reward_mode="live")
    return Metrics(
        reward=float(data["validator_reward"]),
        fpr=float(data["validator_fpr"]),
        ap=float(data["validator_ap_score"]),
        recall=float(data["validator_bot_recall"]),
    )


def _safe_prob(value: float) -> float:
    return max(1e-6, min(1.0 - 1e-6, float(value)))


def _score_transform(
    raw_scores: np.ndarray,
    *,
    invert: bool,
    bias: float,
    temperature: float,
) -> np.ndarray:
    probs = 1.0 - raw_scores if invert else raw_scores
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    logits = np.log(probs / (1.0 - probs))
    shifted = (logits + bias) / max(temperature, 1e-6)
    out = 1.0 / (1.0 + np.exp(-np.clip(shifted, -40.0, 40.0)))
    return np.clip(out, 1e-6, 1.0 - 1e-6)


def _best_calibration(model_path: Path, examples: list[dict[str, Any]]) -> tuple[bool, float, float, Metrics]:
    model = Poker44Model(model_path)
    chunks = [e["chunk"] for e in examples]
    labels = np.asarray([int(e["label"]) for e in examples], dtype=int)
    raw = np.asarray([_safe_prob(x) for x in model.predict_chunk_scores(chunks)], dtype=float)

    best: tuple[bool, float, float, Metrics] | None = None
    for invert in (False, True):
        for bias in np.linspace(-8.0, 2.0, 81):
            for temp in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
                tuned = _score_transform(raw, invert=invert, bias=float(bias), temperature=float(temp))
                rew, details = reward(tuned, labels)
                current = Metrics(
                    reward=float(rew),
                    fpr=float(details["fpr"]),
                    ap=float(details["ap_score"]),
                    recall=float(details["bot_recall"]),
                )
                if best is None or current.reward > best[3].reward:
                    best = (invert, float(bias), float(temp), current)
    assert best is not None
    return best


def _write_tuned_artifact(
    src_model: Path,
    out_model: Path,
    *,
    invert: bool,
    bias: float,
    temperature: float,
) -> None:
    artifact = joblib.load(src_model)
    metadata = dict(artifact.get("metadata") or {})
    base_name = str(metadata.get("model_name", "poker44_auto")).strip() or "poker44_auto"
    metadata["model_name"] = f"{base_name}_auto"
    metadata["model_version"] = "auto-calibrated"
    metadata["score_invert"] = bool(invert)
    metadata["score_logit_bias"] = float(bias)
    metadata["score_logit_temperature"] = float(temperature)
    notes = str(metadata.get("notes", "")).strip()
    suffix = f" auto-calibrated invert={invert} bias={bias:.3f} temp={temperature:.3f}"
    metadata["notes"] = (notes + suffix).strip()
    artifact["metadata"] = metadata
    joblib.dump(artifact, out_model)


def _train_candidate(benchmark_path: Path) -> Path:
    stamp = int(time.time())
    output = AUTO_DIR / f"candidate_{stamp}.joblib"
    cmd = [
        "python3",
        "-m",
        "training.train_model_v2",
        "--benchmark-path",
        str(benchmark_path),
        "--output",
        str(output),
        "--seed",
        "42",
        "--n-folds",
        "5",
        "--target-fpr",
        "0.06",
        "--max-validator-fpr",
        "0.09",
        "--calibration-objective",
        "ap_first",
        "--stack-calibrator",
        "quantile",
        "--quantile-calibration-blend",
        "0.92",
        "--human-weight-multiplier",
        "1.8",
        "--meta-c",
        "1.2",
        "--robust-features-only",
        "--no-score-logit-tune",
    ]
    _log(f"training candidate output={output}")
    _run(cmd, env=dict(os.environ))
    return output


def _deploy_model(active_model: Path) -> None:
    wallet = os.getenv("AUTO_WALLET_NAME", "justice-coldkey")
    hotkey = os.getenv("AUTO_HOTKEY_NAME", "justice-hotkey-poker44")
    axon_port = os.getenv("AUTO_AXON_PORT", "8091")
    pm2_name = os.getenv("AUTO_PM2_NAME", "poker44-miner")
    allowlist = os.getenv(
        "AUTO_ALLOWED_VALIDATOR_HOTKEYS",
        "5E2LP6EnZ54m3wS8s1yPvD5c3xo71kQroBw7aUVK32TKeZ5u "
        "5FxQcdsCXcNjWowQ63Y2oeMhN3JRQksejV3aHRr4XmtknM2k "
        "5FZD47WhA1UaVicYAr7pGnWb2YQLMD7uViipDYN2r1AJ5ggD "
        "5EP9fmtknrTnDhQmLRY9ciFYoM7YZM8rPWvQ9J7yywEsn126 "
        "5HWe7T96SrY4vRvaLmSoriUJ2CGvhRc559U1vZ1pNPuyz2VA "
        "5CsvRJXuR955WojnGMdok1hbhffZyB4N5ocrv82f3p5A2zVp "
        "5Hftk9jrMGSJtKBPWkkAkU53FUSr2BqHGPCThg7mbob3hEq1 "
        "5HmkWGB5PVzKCNLB4QxWWHFVEHPAbKKxGyoXW7Evs38gs126 "
        "5G9hfkx9wGB1CLMT9WXkpHSAiYzjZb5o1Boyq4KAdDhjwrc5 "
        "5FLoWCDovMPeH3Gv4syQSZ8TuKcMv6N27g8diDU8zJSeRv8m "
        "5DqrUa2z6E9taJdY8FGiPCrtCswsEjHjPbVo5xcTw2GqvKZm",
    )
    env = dict(os.environ)
    env.update(
        {
            "WALLET_NAME": wallet,
            "HOTKEY": hotkey,
            "AXON_PORT": axon_port,
            "PM2_NAME": pm2_name,
            "POKER44_MODEL_PATH": str(active_model.resolve()),
            "ALLOWED_VALIDATOR_HOTKEYS": allowlist,
        }
    )
    _run(["pm2", "restart", pm2_name, "--update-env"], env=env)
    _log(f"deployed model={active_model}")


def run_cycle() -> None:
    benchmark_path = _fetch_latest_benchmark()
    examples = _load_examples(benchmark_path)
    active_path = Path(
        os.getenv("AUTO_ACTIVE_MODEL_PATH", str(AUTO_DIR / "active.joblib"))
    )
    min_improve = float(os.getenv("AUTO_MIN_REWARD_IMPROVEMENT", "0.02"))
    max_fpr = float(os.getenv("AUTO_MAX_FPR", "0.09"))
    min_ap = float(os.getenv("AUTO_MIN_AP", "0.50"))

    baseline_path = active_path if active_path.exists() else Path(
        os.getenv("POKER44_MODEL_PATH", str(REPO_ROOT / "models" / "poker44_v17_tuned.joblib"))
    )
    baseline = _eval_model(baseline_path, examples)
    _log(
        "baseline "
        f"path={baseline_path} reward={baseline.reward:.6f} fpr={baseline.fpr:.6f} "
        f"ap={baseline.ap:.6f} recall={baseline.recall:.6f}"
    )

    candidate_raw = _train_candidate(benchmark_path)
    invert, bias, temp, tuned_est = _best_calibration(candidate_raw, examples)
    candidate_tuned = AUTO_DIR / f"{candidate_raw.stem}_tuned.joblib"
    _write_tuned_artifact(
        candidate_raw,
        candidate_tuned,
        invert=invert,
        bias=bias,
        temperature=temp,
    )
    candidate = _eval_model(candidate_tuned, examples)
    _log(
        "candidate "
        f"path={candidate_tuned} reward={candidate.reward:.6f} fpr={candidate.fpr:.6f} "
        f"ap={candidate.ap:.6f} recall={candidate.recall:.6f} "
        f"invert={invert} bias={bias:.3f} temp={temp:.3f}"
    )

    should_deploy = (
        candidate.reward >= baseline.reward + min_improve
        and candidate.fpr <= max_fpr
        and candidate.ap >= min_ap
    )
    if should_deploy:
        shutil.copy2(candidate_tuned, active_path)
        _deploy_model(active_path)
    else:
        _log(
            "skip deploy "
            f"(need reward >= {baseline.reward + min_improve:.6f}, fpr <= {max_fpr:.3f}, ap >= {min_ap:.3f})"
        )


def main() -> None:
    interval = int(os.getenv("AUTO_LOOP_INTERVAL_SECONDS", str(6 * 60 * 60)))
    run_once = os.getenv("AUTO_RUN_ONCE", "0").strip() in {"1", "true", "yes", "on"}
    _log(f"starting loop interval_seconds={interval} run_once={run_once}")
    while True:
        try:
            run_cycle()
        except Exception as exc:
            _log(f"cycle failed: {exc}")
        if run_once:
            break
        _log(f"sleeping {interval}s")
        time.sleep(interval)


if __name__ == "__main__":
    main()
