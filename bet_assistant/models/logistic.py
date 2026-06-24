"""Logistic regression for binary markets, in pure Python.

A transparent baseline for any yes/no market (e.g. "team to win", "over X").
Trained by batch gradient descent with L2 regularisation. Features are expected
to be standardised by the caller or kept on similar scales; we also fit an
intercept.

Kept deliberately small and inspectable — no third-party ML dependency — so its
behaviour can be validated in the backtest harness before it flags live bets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .base import InsufficientModelData, ModelOutput
from .mathutils import sigmoid, wilson_half_width


@dataclass
class LogisticRegression:
    learning_rate: float = 0.1
    epochs: int = 500
    l2: float = 1e-3
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    _n_train: int = 0

    def fit(self, X: list[list[float]], y: list[int]) -> "LogisticRegression":
        if not X or not y or len(X) != len(y):
            raise ValueError("X and y must be non-empty and equal length")
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        m = len(X)
        for _ in range(self.epochs):
            grad_w = [0.0] * n_features
            grad_b = 0.0
            for xi, yi in zip(X, y):
                pred = self._raw(xi)
                err = pred - yi
                for j in range(n_features):
                    grad_w[j] += err * xi[j]
                grad_b += err
            for j in range(n_features):
                grad_w[j] = grad_w[j] / m + self.l2 * self.weights[j]
                self.weights[j] -= self.learning_rate * grad_w[j]
            self.bias -= self.learning_rate * grad_b / m
        self._n_train = m
        return self

    def _raw(self, x: list[float]) -> float:
        z = self.bias + sum(w * xi for w, xi in zip(self.weights, x))
        return sigmoid(z)

    def predict_proba(self, x: list[float]) -> float:
        if not self.weights:
            raise InsufficientModelData("model is not fitted")
        if len(x) != len(self.weights):
            raise ValueError("feature length mismatch")
        return self._raw(x)

    def market_binary(
        self, x: list[float], market: str, yes_label: str = "yes", no_label: str = "no"
    ) -> ModelOutput:
        p = self.predict_proba(x)
        return ModelOutput(
            market=market,
            probabilities={yes_label: p, no_label: 1.0 - p},
            confidence_half_width=wilson_half_width(max(p, 1 - p), self._n_train),
            effective_samples=self._n_train,
        )
