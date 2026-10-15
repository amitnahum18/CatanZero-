"""Inference for the exported small ridge regressor; no NumPy dependency."""

import json
from math import isfinite
from pathlib import Path

from catanzero.features import FeatureEncoder, feature_schema
from catanzero.regression_data import normalize


class LinearMoveScorer:
    def __init__(self, model_path):
        path = Path(model_path)
        self.model = json.loads(path.read_text(encoding="utf-8"))
        self.normalizer = json.loads((path.parent / "normalizer.json").read_text(encoding="utf-8"))
        self.feature_version = self.model["feature_version"]
        schema = feature_schema(self.feature_version)
        if self.model["feature_version"] != schema["version"] or self.model["feature_names"] != schema["feature_names"]:
            raise ValueError("model feature schema mismatch")
        if self.normalizer["feature_names"] != schema["feature_names"] or self.normalizer["fit_split"] != "train":
            raise ValueError("normalizer schema or fit split mismatch")
        self.weights, self.intercept = self.model["coefficients"], self.model["intercept"]
        if len(self.weights) != schema["input_columns"] or any(not isfinite(v) for v in [self.intercept, *self.weights]):
            raise ValueError("invalid linear model coefficients")
        if len(self.normalizer["scale"]) != len(self.weights) or len(self.normalizer["offset"]) != len(self.weights):
            raise ValueError("invalid normalizer length")
        if any(not isfinite(s) or s <= 0 for s in self.normalizer["scale"]):
            raise ValueError("invalid normalizer scale")
        if any(not isfinite(offset) for offset in self.normalizer["offset"]):
            raise ValueError("invalid normalizer offset")

    def score(self, encoder, inputs):
        if encoder.version != self.feature_version:
            raise ValueError("encoder/model feature version mismatch")
        x = normalize(encoder.encode(inputs), self.normalizer)
        raw = self.intercept + sum(weight * value for weight, value in zip(self.weights, x))
        return min(max(raw, 0.0), 1.0)

    def rank(self, map_document, inputs):
        encoder = FeatureEncoder(map_document, version=self.feature_version)
        ranked = [{"action": action, "score": self.score(encoder, {**inputs, "action": action})}
                  for action in inputs["legal_actions"]]
        return sorted(ranked, key=lambda row: row["score"], reverse=True)
