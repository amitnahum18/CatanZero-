"""Real episodes for tree FQI: one learner versus three static opponents."""

import argparse
from array import array
import gzip
import json
import pickle
from pathlib import Path
import random
from time import perf_counter

from catanatron.serialization import action_to_json, map_to_json, state_to_json

from catanzero.dataset import observation, ordered_actions
from catanzero.simulation import SimulationConfig, create_game
from catanzero.static_policies import ObservationPolicy, forbid_forward_simulation
from catanzero.tournament import atomic_json, canonical_hash


def episode(config, index, output):
    started = perf_counter()
    seed = config["base_seed"] + index
    game = create_game(SimulationConfig(), seed)
    learner = game.state.colors[index % 4]
    document = map_to_json(game.state.board.map)
    board_hash = canonical_hash(document)
    if board_hash in config.get("excluded_board_hashes", []):
        raise ValueError("board overlaps earlier data")
    teacher = ObservationPolicy("F_static", document, {})
    candidate = ObservationPolicy("learner", document, config) if config.get("learner") else teacher
    rng = random.Random(seed * 1000003 + 17)
    states, trace, guarded, explored = [], [], 0, 0
    termination = None
    while game.winning_color() is None:
        if perf_counter() - started >= config.get("max_seconds", 60):
            termination = "time_limit"
            break
        if len(game.state.action_records) >= config.get("max_actions", 2000):
            termination = "action_limit"
            break
        actor = game.state.current_color()
        actions = ordered_actions(game)
        inputs = {"map_ref": 0, "observation": observation(game, actor),
                  "legal_actions": [action_to_json(a) for a in actions]}
        actor_is_learner = actor == learner
        before_rng = game.state.random.getstate()
        with forbid_forward_simulation() as attempts:
            if actor_is_learner and config.get("collect", False):
                vectors = [teacher.encoder.encode({**inputs, "action": a}) for a in inputs["legal_actions"]]
                teacher_scores = [teacher.heuristic({**inputs, "action": a}, x)
                                  for a, x in zip(inputs["legal_actions"], vectors)]
                selected_index = max(range(len(actions)), key=teacher_scores.__getitem__)
                if len(actions) > 1 and rng.random() < config.get("epsilon", 0.1):
                    selected_index = rng.randrange(len(actions))
                    explored += 1
                own = next(p for p in inputs["observation"]["players"] if p["color"] == actor.value)
                states.append({"action_index": len(trace), "actor": actor.value,
                               "features": array("f", (v for x in vectors for v in x)),
                               "candidate_count": len(actions), "selected_index": selected_index,
                               "teacher_scores": teacher_scores, "vp": own["actual_vp"],
                               "has_choice": len(actions) > 1})
            else:
                selected_index, _ = (candidate if actor_is_learner else teacher).choose(inputs)
        assert not attempts and game.state.random.getstate() == before_rng
        guarded += 1
        selected = actions[selected_index]
        trace.append({"action_index": len(trace), "actor": actor.value,
                      "selected_intent": action_to_json(selected), "guarded": True})
        game.execute(selected)
    winner = game.winning_color()
    if winner is not None:
        termination = "victory"
    result = {"game_index": index, "seed": seed, "learner_seat": index % 4,
              "learner_color": learner.value, "initial_map_hash": board_hash,
              "termination": termination, "winner": winner.value if winner else None,
              "learner_won": winner == learner if winner else None,
              "actions": len(trace), "turns": game.state.num_turns,
              "guarded_decisions": guarded, "forbidden_engine_calls": 0,
              "learner_states": len(states), "exploration_selections": explored,
              "elapsed_seconds": perf_counter() - started}
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(output / "experience.pkl.gz", "wb") as stream:
        pickle.dump({"result": result, "states": states, "trace": trace}, stream, protocol=5)
    atomic_json(output / "game.json", state_to_json(game))
    atomic_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(episode(json.loads(args.config.read_text()), args.index, args.output)), flush=True)


if __name__ == "__main__":
    main()
