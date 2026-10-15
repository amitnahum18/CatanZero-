from catanzero.observation import PUBLIC_FIELDS, ordered_actions, observation
"""Small, auditable pilot dataset with perspective-safe inputs and heuristic labels."""

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import json
from math import exp
from multiprocessing import get_context
from pathlib import Path
import random
from time import perf_counter

from catanatron import Game
from catanatron.models.decks import starting_devcard_bank
from catanatron.models.enums import RESOURCES, DEVELOPMENT_CARDS, ActionRecord, ActionType
from catanatron.models.map import LandTile, number_probability
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.serialization import action_to_json, map_to_json, state_to_json
from catanatron.state_functions import player_key

from catanzero.cli import positive_int, write_json
from catanzero.scoring import ScoreConfig, score_move
from catanzero.simulation import COLORS, ENGINE_REVISION, SimulationRandomPlayer

HEURISTIC_VERSION = "public_belief_regret_v2"
REGRET_SCALE = 3.0
VALUE_WEIGHTS = {
    "vp": 3.0, "production": 2.5, "variety": 0.35,
    "enemy_vp": -0.8, "enemy_production": -1.0,
    "readiness": 0.4, "stock": 0.06, "discard_risk": -0.15,
    "road": 0.05, "knights": 0.15, "dev_cards": 0.3,
    "expansion": 0.6, "winning": 100.0,
}






def belief_game(game, actor, seed):
    """Replace hidden information with a synthetic allocation based on visible counts.

    One belief sample per decision: intentionally a pilot approximation, not an
    accurate posterior or strong search teacher. Independent RNG prevents copies
    (which share RNG in this engine) from altering actual game outcomes.
    """
    result = game.copy()
    result.random = result.state.random = random.Random(seed)
    state = result.state
    unseen = Counter(starting_devcard_bank())
    actor_key = player_key(state, actor)
    for card in DEVELOPMENT_CARDS:
        unseen[card] -= state.player_state[f"{actor_key}_{card}_IN_HAND"]
        unseen[card] -= sum(state.player_state[f"{player_key(state, c)}_PLAYED_{card}"] for c in state.colors)
    pool = list(unseen.elements())
    result.state.random.shuffle(pool)
    # The resource bank is visible. Conserve its per-resource totals; otherwise
    # a real legal maritime trade can become impossible in the synthetic state.
    resource_pool = [19 - state.player_state[f"{actor_key}_{r}_IN_HAND"] - bank
                     for r, bank in zip(RESOURCES, game.state.resource_freqdeck)]
    for color in state.colors:
        if color == actor:
            continue
        key = player_key(state, color)
        resource_count = sum(state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES)
        card_count = sum(state.player_state[f"{key}_{c}_IN_HAND"] for c in DEVELOPMENT_CARDS)
        for r in RESOURCES:
            state.player_state[f"{key}_{r}_IN_HAND"] = 0
        for _ in range(resource_count):
            index = state.random.choice([i for i, count in enumerate(resource_pool) if count])
            resource_pool[index] -= 1
            state.player_state[f"{key}_{RESOURCES[index]}_IN_HAND"] += 1
        allocated = Counter(pool.pop() for _ in range(card_count))
        for card in DEVELOPMENT_CARDS:
            state.player_state[f"{key}_{card}_IN_HAND"] = allocated[card]
        state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] = (
            state.player_state[f"{key}_VICTORY_POINTS"] + allocated["VICTORY_POINT"]
        )
        for card in DEVELOPMENT_CARDS:
            if card != "VICTORY_POINT":
                state.player_state[f"{key}_{card}_OWNED_AT_START"] = allocated[card] > 0
    assert not any(resource_pool), "synthetic opponent allocation must conserve resources"
    state.resource_freqdeck = game.state.resource_freqdeck.copy()
    state.development_listdeck = pool
    return result


class PilotHeuristic:
    def __init__(self, game):
        self.node_yields = {}
        for node in game.state.board.map.land_nodes:
            self.node_yields[node] = [
                (tuple(coord), tile.resource, number_probability(tile.number))
                for coord, tile in game.state.board.map.tiles.items()
                if isinstance(tile, LandTile) and tile.resource is not None and node in tile.nodes.values()
            ]

    def value(self, game, actor):
        state = game.state
        key = player_key(state, actor)
        hand = {r: state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES}
        production = {c: Counter() for c in state.colors}
        for node, (color, kind) in state.board.buildings.items():
            for coordinate, resource, probability in self.node_yields[node]:
                if coordinate != state.board.robber_coordinate:
                    production[color][resource] += probability * (2 if kind == "CITY" else 1)
        enemies = [c for c in state.colors if c != actor]
        buildable = state.board.buildable_node_ids(actor)
        expansion = max((sum(p for _, _, p in self.node_yields[n]) for n in buildable), default=0)
        city = (min(hand["WHEAT"], 2) + min(hand["ORE"], 3)) / 5
        settlement = sum(min(hand[r], 1) for r in ("WOOD", "BRICK", "SHEEP", "WHEAT")) / 4
        vp = state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
        stock = sum(hand.values())
        features = {
            "vp": vp, "production": sum(production[actor].values()),
            "variety": len(production[actor]),
            "enemy_vp": max(state.player_state[f"{player_key(state, c)}_VICTORY_POINTS"] for c in enemies),
            "enemy_production": max(sum(production[c].values()) for c in enemies),
            "readiness": city + settlement, "stock": min(stock, 8),
            "discard_risk": max(stock - state.discard_limit, 0),
            "road": min(state.player_state[f"{key}_LONGEST_ROAD_LENGTH"], 5),
            "knights": min(state.player_state[f"{key}_PLAYED_KNIGHT"], 3),
            "dev_cards": sum(state.player_state[f"{key}_{c}_IN_HAND"] for c in DEVELOPMENT_CARDS if c != "VICTORY_POINT"),
            "expansion": expansion, "winning": float(vp >= game.vps_to_win),
        }
        return sum(VALUE_WEIGHTS[f] * value for f, value in features.items())

    def expected_value(self, game, action, trade_followup=True):
        actor = action.color
        if action.action_type == ActionType.ROLL:
            outcomes = [((n // 2, n - n // 2), number_probability(n)) for n in range(2, 13)]
        elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            # Pool = unknown deck plus unknown opponent cards, not real hidden deck composition.
            counts = Counter(game.state.development_listdeck)
            for color in game.state.colors:
                if color != actor:
                    key = player_key(game.state, color)
                    counts.update({c: game.state.player_state[f"{key}_{c}_IN_HAND"] for c in DEVELOPMENT_CARDS})
            total = sum(counts.values())
            outcomes = [(card, count / total) for card, count in sorted(counts.items()) if count]
        elif action.action_type == ActionType.MOVE_ROBBER and action.value[1] is not None:
            key = player_key(game.state, action.value[1])
            counts = {r: game.state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES}
            total = sum(counts.values())
            outcomes = [(r, count / total) for r, count in counts.items() if count] if total else [(None, 1.0)]
        else:
            outcomes = [(None, 1.0)]
        expected = 0.0
        for outcome, probability in outcomes:
            child = game.copy()
            child.random = child.state.random = random.Random(0)
            if action.action_type == ActionType.BUY_DEVELOPMENT_CARD and outcome not in child.state.development_listdeck:
                child.state.development_listdeck[-1] = outcome
            child.execute(action, action_record=ActionRecord(action, outcome))
            value = self.value(child, actor)
            # Limited two-ply lookahead catches trade -> city/settlement/road.
            if trade_followup and action.action_type == ActionType.MARITIME_TRADE and child.state.current_color() == actor:
                for next_action in ordered_actions(child):
                    if next_action.action_type in (ActionType.BUILD_CITY, ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD):
                        value = max(value, self.expected_value(child, next_action, False))
            expected += probability * value
        return expected

    def label(self, game, actions, selected):
        values = {action: self.expected_value(game, action) for action in actions}
        best = max(values.values())
        regret = max(best - values[selected], 0.0)
        return {
            "heuristic_score": exp(-regret / REGRET_SCALE), "raw_regret": regret,
            "selected_value": values[selected], "best_value": best,
            "best_action": action_to_json(max(values, key=values.get)),
            "version": HEURISTIC_VERSION, "alternatives_evaluated": len(values),
        }


def make_players(codes, game_index):
    classes = {"R": SimulationRandomPlayer, "W": WeightedRandomPlayer,
               "VP": VictoryPointPlayer, "F": ValueFunctionPlayer}
    rotated = codes[game_index % len(codes):] + codes[:game_index % len(codes)]
    return [classes[code](color) for code, color in zip(rotated, COLORS)], dict(zip(COLORS, rotated))


def collect_game(seed, index, codes, max_turns, max_actions, score_config):
    started = perf_counter()
    players, bot_codes = make_players(codes, index)
    game = Game(players, seed=seed)
    evaluator = PilotHeuristic(game)
    rows = []
    label_seconds = 0.0
    while game.winning_color() is None and game.state.num_turns < max_turns and len(rows) < max_actions:
        actor = game.state.current_color()
        actions = ordered_actions(game)
        before = observation(game, actor)
        proxy = belief_game(game, actor, seed * 1_000_003 + len(rows))
        # All policies act on a synthetic hidden-state allocation, never the real future RNG/deck.
        player = game.state.current_player()
        selected = actions[0] if len(actions) == 1 else player.decide(proxy, actions)
        choice = len(actions) > 1
        label_started = perf_counter()
        label = evaluator.label(proxy, actions, selected) if choice else None
        label_seconds += perf_counter() - label_started
        row = {
            "game_index": index, "action_index": len(rows), "actor_bot": bot_codes[actor],
            "has_choice": choice,
            "inputs": {"map_ref": index, "observation": before,
                       "legal_actions": [action_to_json(a) for a in actions],
                       "action": action_to_json(selected)},
            "heuristic": label,
        }
        game.execute(selected)
        rows.append(row)
    winner = game.winning_color()
    termination = "victory" if winner else "turn_limit" if game.state.num_turns >= max_turns else "action_limit"
    for observer in game.observers:
        observer.after(game)
    for row in rows:
        actor = row["inputs"]["observation"]["actor"]
        row["train_eligible"] = row["has_choice"] and winner is not None
        row["targets"] = (
            score_move(row["heuristic"]["heuristic_score"], actor,
                       winner.value if winner else None, termination, score_config)
            if row["has_choice"] else {"heuristic_score": None, "weighted_score": None,
                                      "terminal_outcome": float(actor == winner.value) if winner else None}
        )
    result = {
        "game_index": index, "seed": seed, "winner": winner.value if winner else None,
        "termination": termination, "turns": game.state.num_turns, "actions": len(rows),
        "decisions": sum(r["has_choice"] for r in rows),
        "eligible_rows": sum(r["train_eligible"] for r in rows),
        "bot_seats": {c.value: bot_codes[c] for c in game.state.colors},
        "elapsed_seconds": perf_counter() - started, "label_seconds": label_seconds,
    }
    return game, rows, result


def collect_and_save(job):
    """Write one game's large artifacts locally; return only aggregates and samples."""
    seed, index, codes, max_turns, max_actions, scoring, output, save_games = job
    game, rows, result = collect_game(seed, index, codes, max_turns, max_actions, scoring)
    result["split"] = "test" if index % 10 == 9 else "validation" if index % 10 == 8 else "train"
    path = output / "games" / f"game_{index:04d}.moves.jsonl.gz"
    action_counts, bot_counts, score_bins = Counter(), Counter(), Counter()
    audit_groups = {}
    # Independent from the game RNG and deterministic across worker schedules.
    audit_rng = random.Random(seed + 91_337)
    with gzip.open(path, "wt", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, allow_nan=False) + "\n")
            if row["has_choice"]:
                action_counts[row["inputs"]["action"][1]] += 1
                bot_counts[row["actor_bot"]] += 1
                group = (row["inputs"]["action"][1], row["actor_bot"],
                         row["inputs"]["observation"]["initial_phase"])
                count, previous = audit_groups.get(group, (0, None))
                audit_groups[group] = (count + 1, row if audit_rng.randrange(count + 1) == 0 else previous)
                score = row["targets"]["weighted_score"]
                if score is not None:
                    score_bins[str(min(int(score * 10), 9))] += 1
    if save_games:
        write_json(output / "games" / f"game_{index:04d}.json", state_to_json(game))
    result["moves_file"] = str(path.relative_to(output))
    return {
        "result": result,
        "map": {"map_ref": index, "map": map_to_json(game.state.board.map),
                "land_nodes": sorted(game.state.board.map.land_nodes)},
        "action_counts": action_counts, "bot_counts": bot_counts,
        "score_bins": score_bins, "audit_groups": audit_groups,
    }


def collected_games(jobs, workers):
    """Bound in-flight work and emit in index order for reproducible metadata."""
    if workers == 1:
        for job in jobs:
            yield collect_and_save(job)
        return
    # Explicit spawn also exercises the same isolation on Linux and Windows.
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        pending = deque()
        jobs = iter(jobs)
        try:
            for _ in range(2 * workers):
                job = next(jobs, None)
                if job is None:
                    break
                pending.append(executor.submit(collect_and_save, job))
            while pending:
                yield pending.popleft().result()
                job = next(jobs, None)
                if job is not None:
                    pending.append(executor.submit(collect_and_save, job))
        finally:
            for future in pending:
                future.cancel()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate a weighted-label Catan pilot dataset.")
    parser.add_argument("--games", type=positive_int, default=10)
    parser.add_argument("--workers", type=positive_int, default=1,
                        help="Parallel game processes; default 1. Start with 4 or 8 on a multicore PC.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bots", default="R,W,VP,F", help="2-4 comma-separated codes: R,W,VP,F")
    parser.add_argument("--max-turns", type=positive_int, default=1000)
    parser.add_argument("--max-actions", type=positive_int, default=100_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heuristic-weight", type=float, default=0.5)
    parser.add_argument("--outcome-weight", type=float, default=0.5)
    parser.add_argument("--save-games", action="store_true")
    args = parser.parse_args(argv)
    codes = args.bots.split(",")
    if not 2 <= len(codes) <= 4 or any(c not in ("R", "W", "VP", "F") for c in codes):
        parser.error("--bots requires 2-4 codes from R,W,VP,F")
    try:
        scoring = ScoreConfig(args.heuristic_weight, args.outcome_weight)
        args.output.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    (args.output / "games").mkdir()
    workers = min(args.workers, args.games)
    write_json(args.output / "config.json", {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "engine_revision": ENGINE_REVISION, "games": args.games, "base_seed": args.seed,
        "bots": codes, "max_turns": args.max_turns, "max_actions": args.max_actions,
        "workers": workers,
        "weights": dict(zip(("heuristic", "outcome"), scoring.normalized_weights)),
        "heuristic_version": HEURISTIC_VERSION, "value_weights": VALUE_WEIGHTS,
        "regret_scale": REGRET_SCALE, "belief_samples_per_decision": 1,
        "policy_information": "synthetic public-count belief; no actual hidden hands or future stream",
        "limitations": ["pilot heuristic, not calibrated confidence", "F considers only one opponent",
                        "single synthetic belief; no public-history card tracking",
                        "trade-only two-ply lookahead; no full-game search", "no player-to-player trade offers"],
    })
    results = []
    action_counts, bot_counts, score_bins = Counter(), Counter(), Counter()
    audit_groups = {}
    audit_rng = random.Random(args.seed + 91_337)
    started = perf_counter()
    jobs = ((args.seed + index, index, codes, args.max_turns, args.max_actions,
             scoring, args.output, args.save_games) for index in range(args.games))
    print(f"Collecting {args.games} games with {workers} worker(s)", flush=True)
    with (args.output / "results.jsonl").open("x", encoding="utf-8") as results_file, (args.output / "maps.jsonl").open("x", encoding="utf-8") as maps_file:
        for completed in collected_games(jobs, workers):
            result = completed["result"]
            index = result["game_index"]
            action_counts.update(completed["action_counts"])
            bot_counts.update(completed["bot_counts"])
            score_bins.update(completed["score_bins"])
            for group, (count, candidate) in completed["audit_groups"].items():
                old_count, previous = audit_groups.get(group, (0, None))
                total = old_count + count
                # Merge per-game reservoirs with count-proportional selection.
                audit_groups[group] = (total, candidate if audit_rng.randrange(total) < count else previous)
            maps_file.write(json.dumps(completed["map"]) + "\n")
            maps_file.flush()
            results.append(result)
            results_file.write(json.dumps(result) + "\n")
            results_file.flush()
            print(f"[{index + 1}/{args.games}] {result['termination']} actions={result['actions']} decisions={result['decisions']} elapsed={result['elapsed_seconds']:.2f}s", flush=True)
    summary = {
        "games": len(results), "completed_games": sum(r["winner"] is not None for r in results),
        "truncated_games": sum(r["winner"] is None for r in results),
        "actions": sum(r["actions"] for r in results), "decisions": sum(r["decisions"] for r in results),
        "eligible_rows": sum(r["eligible_rows"] for r in results),
        "decision_action_counts": dict(action_counts), "decision_bot_counts": dict(bot_counts),
        "weighted_score_decile_counts": dict(sorted(score_bins.items())),
        "elapsed_seconds": perf_counter() - started,
        "workers": workers,
        "label_seconds": sum(r["label_seconds"] for r in results),
        "split_games": dict(Counter(r["split"] for r in results)),
    }
    write_json(args.output / "summary.json", summary)
    with (args.output / "audit_sample.jsonl").open("x", encoding="utf-8") as stream:
        # Prefer rare action/bot/phase strata over a sample consisting only of setup.
        audit_sample = sorted(audit_groups.values(), key=lambda item: item[0])[:30]
        for _, row in audit_sample:
            stream.write(json.dumps(row) + "\n")
    print(f"Dataset: {args.output.resolve()} | eligible={summary['eligible_rows']} | seconds={summary['elapsed_seconds']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
