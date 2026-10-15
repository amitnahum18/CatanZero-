"""Numeric-only inference for the original frozen 402 ensemble; no ML runtime."""
import json
from math import isfinite
from pathlib import Path
import struct


class Champion402:
    def __init__(self, document):
        self.document = document
        self.feature_version = document['feature_version']
        self.feature_names = document['feature_names']
        self.size = len(self.feature_names)
        if document['format'] != 'catanzero-402-numeric-v1' or self.size != 264:
            raise ValueError('Unsupported Model 402 export.')
        self.ranker = document['ranker']
        self.critic = document['critic']
        self.scale, self.mix = document['scale'], document['mix']
        if not self.ranker or not self.critic or self.scale <= 0:
            raise ValueError('Incomplete Model 402 export.')

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    @staticmethod
    def tree_value(tree, row):
        index = 0
        while True:
            feature, threshold, left, right, value = tree[index]
            if feature < 0:
                return value
            index = left if row[feature] <= threshold else right

    def predict(self, row):
        if len(row) != self.size or not all(isfinite(x) for x in row):
            raise ValueError('Invalid candidate features.')
        # sklearn tree predictors cast inputs to float32 before traversing.
        # LightGBM uses the original float64 values. Preserve both semantics.
        critic_row = struct.unpack(f'<{self.size}f', struct.pack(f'<{self.size}f', *row))
        ranking = sum(self.tree_value(tree, row) for tree in self.ranker)
        value = sum(self.tree_value(tree, critic_row) for tree in self.critic) / len(self.critic)
        return ranking / self.scale + self.mix * value

    def choose(self, rows):
        return max(range(len(rows)), key=lambda i: self.predict(rows[i]))
