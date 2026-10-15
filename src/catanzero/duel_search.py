"""Frozen v4 versus one F opponent, with search independently on/off."""
import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

from catanatron.serialization import action_to_json, map_to_json, state_to_json
from catanatron.state_functions import player_key
from catanzero.board_exclusions import board_hash_variants
from catanzero.dataset import observation, ordered_actions
from catanzero.duel_protocol import CONDITIONS
from catanzero.regression_search import RegressionSearch, public_f_choice
from catanzero.simulation import SimulationConfig, create_game
from catanzero.static_policies import forbid_forward_simulation
from catanzero.tournament import atomic_json, canonical_hash

def run_match(config, condition, index, output):
    output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    seed = config["base_seed"] + index
    game = create_game(SimulationConfig(players=2), seed)
    seat = index % 2
    candidate = game.state.colors[seat]
    document = map_to_json(game.state.board.map)
    if board_hash_variants(document) & set(config["excluded_board_hashes"]):
        raise ValueError("test board overlaps existing data")
    depth, opponent = CONDITIONS[condition]
    policy = RegressionSearch(json.loads(Path(config["model"]).read_text()),
                              json.loads(Path(config["normalizer"]).read_text()), document,
                              depth=depth, beam=config["beam"], beliefs=config["beliefs"],
                              max_branch_actions=config["max_branch_actions"],
                              allow_fewer_players=True, opponent_policy=opponent)
    metadata = {"condition": condition, "game_index": index, "seed": seed,
                "candidate_color": candidate.value, "candidate_seat": seat,
                "initial_map_hash": canonical_hash(document),
                "initial_stream_hash": canonical_hash(game.state.random.getstate()),
                "seating_order": [c.value for c in game.state.colors], "players": 2,
                "v4_search": depth == 2, "opponent": opponent}
    timing = {"v4": {"count": 0, "seconds": 0., "max_seconds": 0.},
              "F": {"count": 0, "seconds": 0., "max_seconds": 0.}}
    reason = "running"
    with gzip.open(output / "decisions.jsonl.gz", "wt", encoding="utf-8", compresslevel=1) as trace:
        while game.winning_color() is None:
            if len(game.state.action_records) >= config["max_actions"]:
                reason = "action_limit"
                break
            if perf_counter() - start >= config["max_seconds"]:
                reason = "time_limit"
                break
            actor = game.state.current_color()
            actions = ordered_actions(game)
            inputs = {"observation": observation(game, actor), "map_ref": 0,
                      "legal_actions": [action_to_json(a) for a in actions]}
            action_index = len(game.state.action_records)
            decision_seed = seed * 1000003 + action_index
            stream = game.state.random.getstate()
            who = "v4" if actor == candidate else "F"
            tick = perf_counter()
            detail = None
            if len(actions) == 1:
                selected_index = 0
            elif who == "v4":
                if depth == 1:
                    with forbid_forward_simulation():
                        selected_index, detail = policy.choose(game, actions, decision_seed)
                else:
                    selected_index, detail = policy.choose(game, actions, decision_seed)
            elif opponent == "F":
                selected_index = actions.index(public_f_choice(game, actions, decision_seed))
            else:
                with forbid_forward_simulation():
                    selected_index, scores = policy.teacher.choose(inputs)
                detail = {"static_scores": scores}
            elapsed = perf_counter() - tick
            if len(actions) > 1:
                timing[who]["count"] += 1
                timing[who]["seconds"] += elapsed
                timing[who]["max_seconds"] = max(timing[who]["max_seconds"], elapsed)
            assert stream == game.state.random.getstate()
            assert inputs["observation"] == observation(game, actor)
            assert len(game.state.action_records) == action_index
            trace.write(json.dumps({"action_index": action_index, "policy": who,
                                    "inputs": inputs, "selected_index": selected_index,
                                    "decision_seed": decision_seed, "detail": detail,
                                    "elapsed_seconds": elapsed}, allow_nan=False) + "\n")
            game.execute(actions[selected_index])
    winner = game.winning_color()
    result = {**metadata, "termination": "victory" if winner else reason,
              "candidate_won": winner == candidate if winner else None,
              "winner": winner.value if winner else None,
              "actions": len(game.state.action_records), "turns": game.state.num_turns,
              "elapsed_seconds": perf_counter() - start, "timing": timing,
              "search_stats": policy.stats,
              "victory_points": {c.value: game.state.player_state[
                  f"{player_key(game.state, c)}_ACTUAL_VICTORY_POINTS"] for c in game.state.colors}}
    atomic_json(output / "result.json", result)
    atomic_json(output / "game.json", state_to_json(game))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run_match(json.loads(args.config.read_text()), args.condition, args.index,
                               args.output)), flush=True)


if __name__ == "__main__":
    main()
