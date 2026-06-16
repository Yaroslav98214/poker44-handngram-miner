"""Hand-level n-gram ensemble for Poker44 chunk scoring.

Tokenizes each sanitized hand's action stream into street/action/size tokens,
builds bag-of-ngram counts, scores every hand with a LightGBM + logistic
ensemble, and aggregates hand probabilities into a single chunk risk score.

Chunk scores use **absolute** calibration (global sigmoid stretch fitted on
training data), not batch-relative ranking. That keeps scores comparable
across homogeneous all-human or all-bot validator batches.
"""

from __future__ import annotations

import collections
import math
from typing import Any, Dict, List, Sequence

import numpy as np

ACTION_CODES = {
    "fold": "F",
    "call": "C",
    "raise": "R",
    "check": "K",
    "bet": "B",
    "action": "A",
    "all_in": "I",
}


def hand_ngram_doc(hand: Dict[str, Any]) -> collections.Counter:
    """Bag-of-ngrams document for a single sanitized hand."""
    actions = hand.get("actions") or []
    metadata = hand.get("metadata") or {}
    button_seat = metadata.get("button_seat")
    max_seats = metadata.get("max_seats") or 6

    tokens: List[str] = []
    grams: collections.Counter = collections.Counter()
    acting_seats = set()
    for action in actions:
        street = (action.get("street") or "x")[:1]
        act = ACTION_CODES.get(action.get("action_type") or "x", "X")
        amount = float(action.get("amount") or 0.0)
        pot_before = float(action.get("pot_before") or 0.0)
        if amount <= 0:
            bucket = "0"
        elif pot_before <= 0:
            bucket = "?"
        else:
            ratio = amount / pot_before
            bucket = "s" if ratio < 0.4 else ("m" if ratio < 0.9 else ("p" if ratio < 1.5 else "o"))
        token = street + act + bucket
        tokens.append(token)
        grams[token] += 1
        try:
            rel = (int(action.get("actor_seat")) - int(button_seat)) % int(max_seats)
            grams["pos" + str(rel) + act] += 1
        except Exception:
            pass
        acting_seats.add(action.get("actor_seat"))
    for i in range(len(tokens) - 1):
        grams[tokens[i] + "|" + tokens[i + 1]] += 1
        if i + 2 < len(tokens):
            grams[tokens[i] + "|" + tokens[i + 1] + "|" + tokens[i + 2]] += 1
    grams["len"] = len(tokens)
    grams["nseats"] = len(acting_seats)
    return grams


class HandNgramEnsemble:
    """LightGBM + logistic-regression ensemble over hand-level n-gram counts."""

    def __init__(
        self,
        vocabulary: Dict[str, int],
        lgb_model: Any,
        lr_model: Any,
        lgb_weight: float = 0.6,
        aggregation: str = "mean",
        score_low: float = 0.04,
        score_high: float = 0.49,
        stretch_center: float | None = None,
        stretch_scale: float | None = None,
    ) -> None:
        self.vocabulary = dict(vocabulary)
        self.lgb_model = lgb_model
        self.lr_model = lr_model
        self.lgb_weight = float(lgb_weight)
        self.aggregation = aggregation
        self.score_low = float(score_low)
        self.score_high = float(score_high)
        # Monotonic sigmoid stretch: chunk-mean hand probabilities concentrate
        # tightly around ~0.5, so a global sigmoid recenters/spreads them
        # without changing ordering (AP-safe).
        self.stretch_center = stretch_center
        self.stretch_scale = stretch_scale

    # ---- vectorization ----
    def _matrix(self, hands: Sequence[Dict[str, Any]]) -> np.ndarray:
        rows = np.zeros((len(hands), len(self.vocabulary)), dtype=np.float32)
        for i, hand in enumerate(hands):
            doc = hand_ngram_doc(hand)
            for key, value in doc.items():
                j = self.vocabulary.get(key)
                if j is not None:
                    rows[i, j] = value
        return rows

    def _hand_probs(self, hands: Sequence[Dict[str, Any]]) -> np.ndarray:
        if not hands:
            return np.zeros(0)
        x = self._matrix(hands)
        p_lgb = self.lgb_model.predict_proba(x)[:, 1]
        p_lr = self.lr_model.predict_proba(x)[:, 1]
        return self.lgb_weight * p_lgb + (1.0 - self.lgb_weight) * p_lr

    def _raw_chunk_score(self, chunk: Sequence[Dict[str, Any]]) -> float:
        probs = self._hand_probs([h for h in chunk if isinstance(h, dict)])
        if probs.size == 0:
            return 0.0
        if self.aggregation == "median":
            return float(np.median(probs))
        return float(np.mean(probs))

    @staticmethod
    def _sigmoid_stretch(raw: float, center: float, scale: float) -> float:
        scale = max(float(scale), 1e-6)
        z = (float(raw) - float(center)) / scale
        z = max(-40.0, min(40.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def _map_absolute(self, raw: float) -> float:
        if self.stretch_center is not None and self.stretch_scale:
            stretched = self._sigmoid_stretch(raw, self.stretch_center, self.stretch_scale)
        else:
            stretched = max(0.0, min(1.0, float(raw)))
        return self.score_low + (self.score_high - self.score_low) * stretched

    # ---- Poker44Model integration point ----
    def predict_chunk_scores(
        self,
        chunks: Sequence[Sequence[Dict[str, Any]]],
        feature_rows: Any = None,
    ) -> List[float]:
        return [self._map_absolute(self._raw_chunk_score(chunk)) for chunk in chunks]
