"""Dependency-free inference for exported numeric sklearn histogram GB trees."""

import json
from math import isfinite
from pathlib import Path


class PortableTrees:
    def __init__(self, model, normalizer):
        self.model, self.normalizer = model, normalizer
        self.size = model["input_columns"]
        if model["feature_names"] != normalizer["feature_names"] or normalizer["fit_split"] != "train":
            raise ValueError("feature order or normalization split mismatch")
        if len(model["feature_names"]) != self.size or len(normalizer["offset"]) != self.size or len(normalizer["scale"]) != self.size:
            raise ValueError("feature/normalizer size mismatch")
        if any(not isfinite(float(s)) or s <= 0 for s in normalizer["scale"]):
            raise ValueError("invalid scale")
        if any(not isfinite(float(o)) for o in normalizer["offset"]):
            raise ValueError("invalid offset")
        if not isfinite(float(model["baseline"])):
            raise ValueError("invalid baseline")
        for tree in model["trees"]:
            if not tree:
                raise ValueError("empty tree")
            for node in tree:
                if not isfinite(float(node["value"])):
                    raise ValueError("invalid tree value")
                if not node["leaf"] and (not 0 <= node["feature"] < self.size or
                                          not 0 <= node["left"] < len(tree) or
                                          not 0 <= node["right"] < len(tree) or
                                          not isfinite(float(node["threshold"]))):
                    raise ValueError("invalid split node")

    @classmethod
    def load(cls, model_path, normalizer_path):
        return cls(json.loads(Path(model_path).read_text(encoding="utf-8")),
                   json.loads(Path(normalizer_path).read_text(encoding="utf-8")))

    def predict_normalized(self, x):
        if len(x) != self.size or any(not isfinite(float(value)) for value in x):
            raise ValueError("invalid numeric input")
        value = float(self.model["baseline"])
        for tree in self.model["trees"]:
            index = 0
            for _ in range(len(tree) + 1):
                node = tree[index]
                if node["leaf"]:
                    value += node["value"]
                    break
                index = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
            else:
                raise ValueError("cyclic tree")
        return value

    def predict_raw_features(self, x):
        if len(x) != self.size:
            raise ValueError("invalid feature count")
        normalized = [(v - offset) / scale for v, offset, scale in
                      zip(x, self.normalizer["offset"], self.normalizer["scale"])]
        return self.predict_normalized(normalized)
