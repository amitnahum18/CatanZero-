"""Balanced frozen A/B versus two static heuristic policies, no lookahead."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from time import perf_counter

from catanatron.serialization import action_to_json, map_to_json, state_to_json
from catanatron.state_functions import player_key

from catanzero.dataset import observation, ordered_actions
from catanzero.simulation import SimulationConfig, create_game
from catanzero.static_policies import ObservationPolicy, forbid_forward_simulation
from catanzero.tournament import atomic_json, canonical_hash

POLICIES = ("A_mixed", "B_terminal", "F_static")


def make_match(manifest, seed, board_index, leg):
    game = create_game(SimulationConfig(players=4), seed)
    first, second = board_index % 4, (board_index + 1) % 4
    a, b = (first, second) if leg == 0 else (second, first)
    by_color = {c.value: "A_mixed" if i == a else "B_terminal" if i == b else "F_static"
                for i, c in enumerate(game.state.colors)}
    document = map_to_json(game.state.board.map)
    policies = {p: ObservationPolicy(p, document, manifest) for p in POLICIES}
    return game, by_color, policies, {"A_mixed": a, "B_terminal": b}


def run(manifest, seed, board_index, leg, output, max_seconds=120, max_turns=1000, max_actions=3000):
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    game, by_color, policies, seats = make_match(manifest, seed, board_index, leg)
    board_hash = canonical_hash(map_to_json(game.state.board.map))
    if board_hash in manifest["source_board_hashes"]:
        raise ValueError("evaluation board overlaps source data")
    metadata = {"seed": seed, "board_index": board_index, "leg": leg, "policy_seats": seats,
                "policies_by_color": by_color, "seating_order": [c.value for c in game.state.colors],
                "initial_map_hash": board_hash, "initial_stream_hash": canonical_hash(game.state.random.getstate()),
                "decision_protocol": "whitelisted_observation_only_no_forward_simulation"}
    times = {p: [] for p in POLICIES}
    counts = {p: Counter() for p in POLICIES}
    guarded = {p: 0 for p in POLICIES}
    forbidden = []
    error = None

    def result(reason):
        winner = game.winning_color()
        return {**metadata, "termination": reason, "error": error,
                "winner": winner.value if winner else None, "winner_policy": by_color[winner.value] if winner else None,
                "actions": len(game.state.action_records), "turns": game.state.num_turns,
                "elapsed_seconds": perf_counter() - started,
                "actual_vp": {c.value: game.state.player_state[f"{player_key(game.state, c)}_ACTUAL_VICTORY_POINTS"]
                              for c in game.state.colors},
                "decision_seconds": {p: {"count": len(t), "sum": sum(t), "mean": sum(t)/len(t) if t else None,
                                          "max": max(t) if t else None} for p, t in times.items()},
                "decision_action_counts": {p: dict(c) for p, c in counts.items()},
                "guarded_decisions": guarded, "forbidden_engine_calls": forbidden}

    atomic_json(output / "checkpoint.json", result("running"))
    termination = None
    try:
        with gzip.open(output / "decisions.jsonl.gz", "wt", encoding="utf-8") as trace:
            while game.winning_color() is None:
                if perf_counter() - started >= max_seconds:
                    termination = "time_limit"
                    break
                if game.state.num_turns >= max_turns:
                    termination = "turn_limit"
                    break
                if len(game.state.action_records) >= max_actions:
                    termination = "action_limit"
                    break
                actor = game.state.current_color()
                policy = by_color[actor.value]
                actions = ordered_actions(game)
                inputs = {"map_ref": 0, "observation": observation(game, actor),
                          "legal_actions": [action_to_json(a) for a in actions]}
                state_before, rng_before = state_to_json(game), game.state.random.getstate()
                t0 = perf_counter()
                with forbid_forward_simulation() as attempts:
                    # No real Game, belief Game, seed, deck or future RNG passed.
                    selected_index, scores = policies[policy].choose(inputs)
                elapsed = perf_counter() - t0
                guarded[policy] += 1
                forbidden.extend(attempts)
                if game.state.random.getstate() != rng_before or state_to_json(game) != state_before:
                    raise ValueError("policy mutated real state or future stream")
                selected = actions[selected_index]
                if len(actions) > 1:
                    times[policy].append(elapsed)
                    counts[policy][selected.action_type.value] += 1
                record = {"action_index": len(game.state.action_records), "policy": policy,
                          "has_choice": len(actions) > 1, "decision_seconds": elapsed,
                          "inputs": {**inputs, "action": action_to_json(selected)}, "raw_scores": scores,
                          "forward_simulation_guard": True, "forbidden_engine_calls": attempts}
                # Exactly the selected actual move, after leaving the guard.
                game.execute(selected)
                trace.write(json.dumps(record, allow_nan=False) + "\n")
                if len(game.state.action_records) % 25 == 0:
                    atomic_json(output / "checkpoint.json", result("running"))
        if game.winning_color() is not None:
            termination = "victory"
    except Exception as exception:
        if "attempts" in locals():
            forbidden.extend(a for a in attempts if a not in forbidden)
        error = f"{type(exception).__name__}: {exception}"
        termination = "error"
    final = result(termination)
    atomic_json(output / "game.json", state_to_json(game))
    atomic_json(output / "result.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--board-index", type=int, required=True)
    parser.add_argument("--leg", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--max-actions", type=int, default=3000)
    parser.add_argument("--max-turns", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.manifest.read_text(encoding="utf-8")), args.seed,
                         args.board_index, args.leg, args.output, args.max_seconds,
                         args.max_turns, args.max_actions)), flush=True)


if __name__ == "__main__":
    main()
