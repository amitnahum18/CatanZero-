"""Convert recorded tournaments to perspective-safe, multi-belief training rows.

Never rerun a bot or sample a replacement outcome. Search-bot choices can be
used for imitation; weighted heuristic targets are still weak supervision.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import json
from math import exp
from multiprocessing import get_context
from pathlib import Path
import shutil
from statistics import fmean, pstdev
from time import perf_counter

from catanatron.models.enums import ActionType
from catanatron.serialization import action_record_from_json, action_to_json, map_to_json, state_to_json

from catanzero.cli import positive_int, write_json
from catanzero.dataset import PilotHeuristic, REGRET_SCALE, VALUE_WEIGHTS, observation, ordered_actions
from catanzero.scoring import ScoreConfig, score_move
from catanzero.search_labels import live_belief
from catanzero.simulation import ENGINE_REVISION
from catanzero.tournament import make_match

VERSION = "public_multibelief_mean_regret_v3"
VERIFY_FIELDS = ("board", "player_state", "num_turns", "action_records",
                 "resource_freqdeck", "development_listdeck")


def seed_splits(seeds):
    """One split per initial board/stream, including across tournament groups."""
    return {seed: "validation" if index % 6 == 4 else "test" if index % 6 == 5 else "train"
            for index, seed in enumerate(sorted(set(seeds)))}


def multibelief_label(game, actions, selected, samples, seed):
    if samples < 1:
        raise ValueError("at least one belief sample is required")
    actor = selected.color
    evaluator = PilotHeuristic(game)
    values = [[] for _ in actions]
    for trial in range(samples):
        view = live_belief(game, actor, seed + trial * 1_000_033)
        for index, action in enumerate(actions):
            values[index].append(evaluator.expected_value(view, action))
    means = [fmean(v) for v in values]
    best_index = max(range(len(actions)), key=means.__getitem__)
    selected_index = actions.index(selected)
    regret = max(means[best_index] - means[selected_index], 0.0)
    return {
        "version": VERSION, "heuristic_score": exp(-regret / REGRET_SCALE),
        "raw_regret": regret, "selected_value": means[selected_index],
        "best_value": means[best_index], "best_action": action_to_json(actions[best_index]),
        "alternatives_evaluated": len(actions), "belief_samples": samples,
        "aggregation": "mean action value, then regret and fixed exponential scale",
        "candidate_values": [{"action": action_to_json(a), "mean": means[i],
                              "std": pstdev(values[i])}
                             for i, a in enumerate(actions)],
    }


def convert_match(document, metadata, index, split, samples):
    """Construct inputs before each recorded action; verify the authoritative end."""
    started = perf_counter()
    game, candidate = make_match(metadata["bot"], metadata["seed"], metadata["candidate_seat"])
    initial = state_to_json(game)
    if initial["colors"] != document["colors"] or initial["map"] != document["map"]:
        raise ValueError("recorded initial conditions do not match this engine")
    rows = []
    bot_seats = {c.value: metadata["bot"] if c == candidate else "F" for c in game.state.colors}
    label_seconds = 0.0
    for action_index, payload in enumerate(document["action_records"]):
        if game.winning_color() is not None:
            raise ValueError("recorded action occurs after victory")
        record = action_record_from_json(payload)
        selected = record.action
        if selected.action_type in (ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD):
            selected = selected._replace(value=None)
        actions = ordered_actions(game)
        if selected not in actions:
            raise ValueError(f"illegal recorded action at {action_index}")
        actor = game.state.current_color()
        before = observation(game, actor)
        choice = len(actions) > 1
        label_started = perf_counter()
        label = multibelief_label(game, actions, selected, samples,
                                  metadata["seed"] * 1_000_003 + action_index) if choice else None
        label_seconds += perf_counter() - label_started
        rows.append({
            "game_index": index, "action_index": action_index, "split": split,
            "actor_bot": bot_seats[actor.value], "has_choice": choice,
            "candidate_action": actor == candidate,
            "inputs": {"map_ref": index, "observation": before,
                       "legal_actions": [action_to_json(a) for a in actions],
                       "action": action_to_json(selected)},
            "heuristic": label,
        })
        game.execute(selected, action_record=record)
    replayed = state_to_json(game)
    for field in VERIFY_FIELDS:
        if replayed[field] != document[field]:
            raise ValueError(f"authoritative replay mismatch: {field}")
    winner = game.winning_color()
    winner_value = winner.value if winner else None
    if winner_value != metadata["winner"] or len(rows) != metadata["actions"]:
        raise ValueError("tournament outcome/count does not match replay")
    termination = "victory" if winner else "turn_limit" if metadata["termination"] == "turn_limit" else "action_limit"
    for row in rows:
        actor = row["inputs"]["observation"]["actor"]
        outcome = float(actor == winner_value) if winner else None
        row["train_eligible"] = row["has_choice"] and winner is not None
        row["imitation_eligible"] = row["train_eligible"] and row["candidate_action"]
        row["targets"] = score_move(row["heuristic"]["heuristic_score"], actor, winner_value,
                                    termination, ScoreConfig()) if row["has_choice"] else {
            "heuristic_score": None, "weighted_score": None, "terminal_outcome": outcome}
    result = {
        "game_index": index, "source_game_index": metadata["game_index"],
        "seed": metadata["seed"], "split": split, "winner": winner_value,
        "termination": termination, "source_termination": metadata["termination"],
        "turns": game.state.num_turns, "actions": len(rows),
        "decisions": sum(r["has_choice"] for r in rows),
        "eligible_rows": sum(r["train_eligible"] for r in rows),
        "candidate_rows": sum(r["imitation_eligible"] for r in rows),
        "bot_seats": bot_seats, "candidate_color": candidate.value,
        "label_seconds": label_seconds, "elapsed_seconds": perf_counter() - started,
        "source_match_directory": metadata["match_directory"],
        "replay_verified": True,
    }
    return rows, result, {"map_ref": index, "map": map_to_json(game.state.board.map),
                          "land_nodes": sorted(game.state.board.map.land_nodes)}


def save_match(job):
    document_path, metadata, index, split, samples, output = job
    document = json.loads(document_path.read_text(encoding="utf-8"))
    rows, result, map_record = convert_match(document, metadata, index, split, samples)
    name = f"game_{index:04d}"
    moves = output / "games" / f"{name}.moves.jsonl.gz"
    with gzip.open(moves, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    shutil.copyfile(document_path, output / "games" / f"{name}.json")
    result["moves_file"] = str(moves.relative_to(output))
    return result, map_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bot", default="AB:2")
    parser.add_argument("--belief-samples", type=positive_int, default=8)
    parser.add_argument("--workers", type=positive_int, default=4)
    args = parser.parse_args(argv)
    source = args.source.resolve()
    if (source / "FAILED.json").exists():
        parser.error("source tournament is marked FAILED")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config["engine_revision"] != ENGINE_REVISION:
        parser.error("engine revision differs from source tournament")
    matches = [json.loads(line) for line in (source / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = seed_splits(r["seed"] for r in matches)
    matches = sorted((r for r in matches if r["bot"] == args.bot), key=lambda r: r["game_index"])
    if not matches or any(r["termination"] != "victory" or r.get("error") for r in matches):
        parser.error("selected bot must have complete, error-free recorded matches")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "games").mkdir()
    workers = min(args.workers, len(matches))
    write_json(output / "config.json", {
        "schema_version": 1, "version": "recorded_tournament_multibelief_v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "engine_revision": ENGINE_REVISION,
        "source_tournament": str(source), "games": len(matches), "candidate_bot": args.bot,
        "opponents": ["F", "F", "F"], "workers": workers,
        "heuristic_version": VERSION, "value_weights": VALUE_WEIGHTS,
        "regret_scale": REGRET_SCALE, "belief_samples_per_decision": args.belief_samples,
        "weights": {"heuristic": 0.5, "outcome": 0.5},
        "split_rule": "sorted unique initial seeds: indices mod 6 0-3 train, 4 validation, 5 test",
        "limitations": ["12-game source ranking is not proof of expert strength",
                        "averaging beliefs reduces sampling noise, not heuristic bias",
                        "weighted score is not calibrated confidence or optimal-action probability",
                        "no public-history belief posterior or player-to-player negotiations",
                        "map_ref is local to this output; qualify it when combining datasets"],
    })
    jobs = [(source / "matches" / args.bot.replace(":", "-") / f"game_{r['game_index']:04d}" / "game.json",
             r, index, splits[r["seed"]], args.belief_samples, output)
            for index, r in enumerate(matches)]
    started = perf_counter()
    results, actions, bots, candidate_actions, candidate_splits = [], Counter(), Counter(), Counter(), Counter()
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool, \
                (output / "results.jsonl").open("x", encoding="utf-8") as result_stream, \
                (output / "maps.jsonl").open("x", encoding="utf-8") as map_stream, \
                gzip.open(output / "candidate.moves.jsonl.gz", "wt", encoding="utf-8") as candidates:
            for result, map_record in pool.map(save_match, jobs):
                results.append(result)
                result_stream.write(json.dumps(result) + "\n")
                result_stream.flush()
                map_stream.write(json.dumps(map_record) + "\n")
                map_stream.flush()
                with gzip.open(output / result["moves_file"], "rt", encoding="utf-8") as stream:
                    for line in stream:
                        row = json.loads(line)
                        if row["has_choice"]:
                            actions[row["inputs"]["action"][1]] += 1
                            bots[row["actor_bot"]] += 1
                        if row["imitation_eligible"]:
                            candidates.write(line)
                            candidate_actions[row["inputs"]["action"][1]] += 1
                            candidate_splits[row["split"]] += 1
                print(f"[{len(results)}/{len(matches)}] actions={result['actions']} candidate={result['candidate_rows']} replay verified", flush=True)
        summary = {"games": len(results), "completed_games": len(results), "truncated_games": 0,
                   "actions": sum(r["actions"] for r in results),
                   "decisions": sum(r["decisions"] for r in results),
                   "eligible_rows": sum(r["eligible_rows"] for r in results),
                   "candidate_rows": sum(r["candidate_rows"] for r in results),
                   "decision_action_counts": dict(actions), "decision_bot_counts": dict(bots),
                   "candidate_action_counts": dict(candidate_actions),
                   "candidate_split_rows": dict(candidate_splits),
                   "split_games": dict(Counter(r["split"] for r in results)),
                   "label_seconds": sum(r["label_seconds"] for r in results),
                   "elapsed_seconds": perf_counter() - started, "workers": workers}
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        write_json(output / "FAILED.json", {"error": str(error), "use_for_training": False})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
