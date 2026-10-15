"""One fresh four-player match: two frozen learned policies plus two F bots."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

from catanatron import Game
from catanatron.models.player import Player
from catanatron.players.value import ValueFunctionPlayer
from catanatron.serialization import action_to_json, map_to_json, state_to_json
from catanatron.state_functions import player_key

from catanzero.dataset import observation, ordered_actions
from catanzero.features import FeatureEncoder, feature_schema, VERSION
from catanzero.portable_trees import PortableTrees
from catanzero.search_labels import live_belief
from catanzero.simulation import COLORS
from catanzero.tournament import atomic_json, canonical_hash


class LearnedPlayer(Player):
    LABEL = "Frozen Gradient Boosting"

    def __init__(self, color, policy, model_path, normalizer_path):
        super().__init__(color)
        self.policy = policy
        self.scorer = PortableTrees.load(model_path, normalizer_path)
        self.feature_version = self.scorer.model.get("feature_version", VERSION)
        if self.scorer.model["feature_names"] != feature_schema(self.feature_version)["feature_names"]:
            raise ValueError("portable model feature schema mismatch")
        self.encoder = None
        self.last_scores = None

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            self.last_scores = None
            return playable_actions[0]
        if self.encoder is None:
            self.encoder = FeatureEncoder(map_to_json(game.state.board.map), version=self.feature_version)
        inputs = {"observation": observation(game, self.color), "map_ref": 0,
                  "legal_actions": [action_to_json(a) for a in playable_actions]}
        self.last_scores = [self.scorer.predict_raw_features(self.encoder.encode({**inputs, "action": action_to_json(a)}))
                            for a in playable_actions]
        # Raw scores preserve ordering when several predictions exceed [0,1].
        # Tie breaking is the same canonical action order for both policies.
        return playable_actions[max(range(len(playable_actions)), key=self.last_scores.__getitem__)]


def make_learned_match(manifest, seed, board_index, leg):
    preview = Game([ValueFunctionPlayer(c) for c in COLORS], seed=seed)
    first, second = board_index % 4, (board_index + 1) % 4
    a_seat, b_seat = (first, second) if leg == 0 else (second, first)
    a_color, b_color = preview.state.colors[a_seat], preview.state.colors[b_seat]
    direct = manifest.get("lineup") == "AABB"
    a_seats = [a_seat, (a_seat + 2) % 4]
    b_seats = [b_seat, (b_seat + 2) % 4]
    direct_a_colors = {preview.state.colors[s] for s in a_seats}
    by_color, players = {}, []
    for color in COLORS:
        policy = ("A_mixed" if color in direct_a_colors else "B_terminal") if direct else (
            "A_mixed" if color == a_color else "B_terminal" if color == b_color else "F")
        by_color[color.value] = policy
        players.append(LearnedPlayer(color, policy, manifest[policy]["portable_path"], manifest["normalizer_path"])
                       if policy != "F" else ValueFunctionPlayer(color))
    game = Game(players, seed=seed)
    assert game.state.colors == preview.state.colors
    assert game.state.random.getstate() == preview.state.random.getstate()
    assert map_to_json(game.state.board.map) == map_to_json(preview.state.board.map)
    return game, by_color, {"A_mixed": a_seats if direct else a_seat,
                           "B_terminal": b_seats if direct else b_seat}


def run(manifest, seed, board_index, leg, output, max_seconds=120, max_turns=1000, max_actions=3000):
    started = perf_counter()
    game, policies, seats = make_learned_match(manifest, seed, board_index, leg)
    initial_map_hash = canonical_hash(map_to_json(game.state.board.map))
    if initial_map_hash in manifest["source_board_hashes"]:
        raise ValueError("evaluation board overlaps training/validation/test source")
    colors_by_policy = {policy: [color for color, p in policies.items() if p == policy]
                        for policy in ("A_mixed", "B_terminal")}
    metadata = {"seed": seed, "board_index": board_index, "leg": leg,
                "policy_seats": seats, "policy_colors": {p: c[0] if len(c) == 1 else c for p, c in colors_by_policy.items()},
                "seating_order": [c.value for c in game.state.colors], "policies_by_color": policies,
                "initial_map_hash": initial_map_hash,
                "initial_stream_hash": canonical_hash(game.state.random.getstate())}
    timings = {policy: [] for policy in ("A_mixed", "B_terminal", "F")}
    action_counts = {policy: Counter() for policy in timings}
    deadline, termination, error = started + max_seconds, None, None

    def result(reason):
        winner = game.winning_color()
        return {**metadata, "termination": reason, "error": error,
                "winner": winner.value if winner else None,
                "winner_policy": policies[winner.value] if winner else None,
                "turns": game.state.num_turns, "actions": len(game.state.action_records),
                "actual_vp": {c.value: game.state.player_state[f"{player_key(game.state, c)}_ACTUAL_VICTORY_POINTS"] for c in game.state.colors},
                "elapsed_seconds": perf_counter() - started,
                "decision_seconds": {p: {"count": len(t), "sum": sum(t), "mean": sum(t) / len(t) if t else None,
                                           "max": max(t) if t else None} for p, t in timings.items()},
                "decision_action_counts": {p: dict(counts) for p, counts in action_counts.items()}}

    atomic_json(output / "checkpoint.json", result("running"))
    try:
        with gzip.open(output / "decisions.jsonl.gz", "wt", encoding="utf-8") as trace:
            while game.winning_color() is None:
                if perf_counter() >= deadline:
                    termination = "time_limit"
                    break
                if game.state.num_turns >= max_turns:
                    termination = "turn_limit"
                    break
                if len(game.state.action_records) >= max_actions:
                    termination = "action_limit"
                    break
                actor = game.state.current_color()
                policy = policies[actor.value]
                actions = ordered_actions(game)
                before = observation(game, actor)
                scores = None
                elapsed = 0.0
                if len(actions) == 1:
                    selected = actions[0]
                else:
                    # F needs a synthetic belief; the learned scorer reads only
                    # its whitelist, with identical values under every belief.
                    view = live_belief(game, actor, seed * 1_000_003 + len(game.state.action_records))
                    view.playable_actions = actions
                    state_before = state_to_json(game)
                    rng_before = game.state.random.getstate()
                    wall_started = perf_counter()
                    player = game.state.current_player()
                    selected = player.decide(view, actions)
                    elapsed = perf_counter() - wall_started
                    timings[policy].append(elapsed)
                    action_counts[policy][selected.action_type.value] += 1
                    scores = player.last_scores if isinstance(player, LearnedPlayer) else None
                    if game.state.random.getstate() != rng_before or state_to_json(game) != state_before:
                        raise ValueError("policy mutated actual game state or future stream")
                if selected not in actions:
                    raise ValueError("illegal policy action")
                record = {"action_index": len(game.state.action_records), "policy": policy,
                          "has_choice": len(actions) > 1, "decision_seconds": elapsed,
                          "inputs": {"map_ref": 0, "observation": before,
                                     "legal_actions": [action_to_json(a) for a in actions],
                                     "action": action_to_json(selected)}, "raw_scores": scores}
                game.execute(selected)
                trace.write(json.dumps(record, allow_nan=False) + "\n")
                if len(game.state.action_records) % 25 == 0:
                    atomic_json(output / "checkpoint.json", result("running"))
        if game.winning_color() is not None:
            termination = "victory"
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        termination = "error"
    final = result(termination)
    atomic_json(output / "game.json", state_to_json(game))
    atomic_json(output / "result.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--board-index", required=True, type=int)
    parser.add_argument("--leg", required=True, type=int, choices=(0, 1))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--max-actions", type=int, default=3000)
    parser.add_argument("--max-turns", type=int, default=1000)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    print(json.dumps(run(manifest, args.seed, args.board_index, args.leg, args.output,
                         args.max_seconds, args.max_turns, args.max_actions)), flush=True)


if __name__ == "__main__":
    main()
