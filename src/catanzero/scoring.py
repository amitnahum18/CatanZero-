"""Combine a supplied move heuristic with the acting player's terminal outcome.

This module does not generate heuristic scores or claim calibrated probabilities.
Final outcomes and scoring metadata belong in targets, never in model inputs.
"""

import argparse
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class ScoreConfig:
    heuristic_weight: float = 0.5
    outcome_weight: float = 0.5

    def __post_init__(self):
        weights = (self.heuristic_weight, self.outcome_weight)
        if any(not isfinite(w) or w < 0 for w in weights):
            raise ValueError("weights must be finite and nonnegative")
        if sum(weights) <= 0 or not isfinite(sum(weights)):
            raise ValueError("sum of weights must be finite and positive")

    @property
    def normalized_weights(self) -> tuple[float, float]:
        total = self.heuristic_weight + self.outcome_weight
        return self.heuristic_weight / total, self.outcome_weight / total


def score_move(
    heuristic_score: float,
    actor: str,
    winner: str | None,
    termination: str,
    config: ScoreConfig = ScoreConfig(),
) -> dict:
    """Return auditable targets. Truncation never creates a loss label.

    Heuristics must already use a documented, fixed scale from zero to one.
    Weights are fixed for the run, independent of turn count or game length.
    """
    if not isfinite(heuristic_score) or not 0 <= heuristic_score <= 1:
        raise ValueError("heuristic_score must be finite and between 0 and 1")
    if not actor:
        raise ValueError("actor must be nonempty")
    if termination not in ("victory", "turn_limit", "action_limit"):
        raise ValueError("unknown termination reason")
    if (termination == "victory") != (winner is not None):
        raise ValueError("victory requires a winner; truncated games require winner=None")
    heuristic_weight, outcome_weight = config.normalized_weights
    outcome = float(actor == winner) if termination == "victory" else None
    # A purely heuristic run does not require a terminal outcome.
    combined = (
        heuristic_score
        if outcome_weight == 0
        else None
        if outcome is None
        else heuristic_weight * heuristic_score + outcome_weight * outcome
    )
    return {
        "heuristic_score": heuristic_score,
        "terminal_outcome": outcome,
        "weighted_score": combined,
        "scoring": {
            "version": "weighted_heuristic_outcome_v1",
            "heuristic_weight": heuristic_weight,
            "outcome_weight": outcome_weight,
            "weight_schedule": "constant",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Weight supplied move heuristics and game outcomes.")
    parser.add_argument("input", type=Path, help="JSONL rows: actor, winner, termination, heuristic_score")
    parser.add_argument("--output", type=Path, required=True, help="new JSONL file; no overwrite")
    parser.add_argument("--heuristic-weight", type=float, default=0.5)
    parser.add_argument("--outcome-weight", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        config = ScoreConfig(args.heuristic_weight, args.outcome_weight)
        rows = []
        # Validate all rows before creating output, so invalid input leaves no partial file.
        with args.input.open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    targets = score_move(
                        row["heuristic_score"], row["actor"], row["winner"], row["termination"], config
                    )
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"row {number}: {error}") from error
                rows.append({**row, "targets": targets})
        with args.output.open("x", encoding="utf-8") as destination:
            for row in rows:
                destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Scored {len(rows)} rows: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
