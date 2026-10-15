"""Paired-seed, balanced-seat comparison of installed bots against three F bots."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter, process_time, sleep

from catanatron import Game
from catanatron.registry import REGISTRY
import catanatron.players
from catanatron.serialization import action_to_json, map_to_json, state_to_json
from catanatron.state_functions import player_key

from catanzero.cli import positive_int, write_json
from catanzero.dataset import ordered_actions
from catanzero.search_labels import live_belief
from catanzero.simulation import COLORS, ENGINE_REVISION


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_match(bot, seed, seat):
    preview = Game([REGISTRY.build("F", c) for c in COLORS], seed=seed)
    candidate_color = preview.state.colors[seat]
    if bot == "F":
        return preview, candidate_color
    game = Game([REGISTRY.build(bot if c == candidate_color else "F", c) for c in COLORS], seed=seed)
    assert game.state.colors == preview.state.colors
    assert map_to_json(game.state.board.map) == map_to_json(preview.state.board.map)
    assert game.state.random.getstate() == preview.state.random.getstate()
    return game, candidate_color


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    # Windows readers/scanners can briefly deny rename of a checkpoint.
    # Keep the previous valid document until replacement succeeds; never
    # fall back to a non-atomic write, and propagate persistent failures.
    for attempt in range(7):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 6:
                raise
            sleep(min(0.025 * 2 ** attempt, 0.4))


def run_match(bot, seed, seat, index, output, max_seconds, max_turns, max_actions):
    started = perf_counter()
    game, candidate = make_match(bot, seed, seat)
    metadata = {
        "bot": bot, "game_index": index, "seed": seed, "candidate_seat": seat,
        "candidate_color": candidate.value, "seating_order": [c.value for c in game.state.colors],
        "initial_map_hash": canonical_hash(map_to_json(game.state.board.map)),
        "initial_stream_hash": canonical_hash(game.state.random.getstate()),
    }
    times = {"candidate": [], "opponents": []}
    deadline = started + max_seconds
    termination = None
    error = None
    last_actor = None

    def result(reason):
        winner = game.winning_color()
        return {
            **metadata, "termination": reason,
            "winner": winner.value if winner else None,
            "candidate_won": None if winner is None else winner == candidate,
            "turns": game.state.num_turns, "actions": len(game.state.action_records),
            "elapsed_seconds": perf_counter() - started,
            "decision_times": times,
            "actual_vp": {c.value: game.state.player_state[f"{player_key(game.state, c)}_ACTUAL_VICTORY_POINTS"] for c in game.state.colors},
            "last_actor": last_actor, "error": error,
        }

    atomic_json(output / "checkpoint.json", result("running"))
    try:
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
            last_actor = actor.value
            actions = ordered_actions(game)
            if len(actions) == 1:
                selected = actions[0]
            else:
                view = live_belief(game, actor, seed * 1_000_003 + len(game.state.action_records))
                view.playable_actions = actions
                before_rng = game.state.random.getstate()
                wall_started, cpu_started = perf_counter(), process_time()
                selected = game.state.current_player().decide(view, actions)
                timing = {"wall_seconds": perf_counter() - wall_started,
                          "cpu_seconds": process_time() - cpu_started,
                          "action_index": len(game.state.action_records),
                          "type": selected.action_type.value, "legal_actions": len(actions)}
                times["candidate" if actor == candidate else "opponents"].append(timing)
                if game.state.random.getstate() != before_rng:
                    raise ValueError("bot search advanced the actual environment RNG")
                if selected not in actions:
                    raise ValueError("bot returned an illegal move")
                if perf_counter() >= deadline:
                    termination = "time_limit"
                    break
            game.execute(selected)
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


def isolated_match(job):
    bot, seed, seat, index, output, seconds, turns, actions = job
    command = [sys.executable, "-m", "catanzero.tournament", "--single", bot,
               "--seed", str(seed), "--seat", str(seat), "--index", str(index),
               "--output", str(output), "--max-seconds", str(seconds),
               "--max-turns", str(turns), "--max-actions", str(actions)]
    started = perf_counter()
    timed_out = False
    with (output / "worker.log").open("x", encoding="utf-8") as log:
        try:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=seconds + 25)
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out, returncode = True, None
    path = output / "result.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
    else:
        checkpoint = output / "checkpoint.json"
        result = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {
            "bot": bot, "seed": seed, "candidate_seat": seat, "game_index": index,
            "decision_times": {"candidate": [], "opponents": []}, "actions": 0,
        }
        result.update(termination="hard_time_limit" if timed_out else "worker_error",
                      winner=None, candidate_won=None, unfinished_decision_time_not_measured=True)
    result.update(process_wall_seconds=perf_counter() - started, worker_returncode=returncode,
                  worker_hard_timeout=timed_out, match_directory=str(output))
    atomic_json(output / "supervised_result.json", result)
    return result


def summarize_matches(results):
    groups = {}
    for bot in dict.fromkeys(r["bot"] for r in results):
        matches = [r for r in results if r["bot"] == bot]
        completed = sum(r["termination"] == "victory" for r in matches)
        wins = sum(r["candidate_won"] is True for r in matches)
        unfinished = len(matches) - completed
        timings = [t for r in matches for t in r["decision_times"]["candidate"]]
        wall = [t["wall_seconds"] for t in timings]
        cpu = [t["cpu_seconds"] for t in timings]
        groups[bot] = {
            "games": len(matches), "completed_games": completed, "wins": wins,
            "losses": completed - wins, "unfinished_games": unfinished,
            "win_rate_completed": wins / completed if completed else None,
            "win_fraction_all_scheduled": wins / len(matches),
            "win_fraction_censoring_bounds": [wins / len(matches), (wins + unfinished) / len(matches)],
            "termination_counts": dict(Counter(r["termination"] for r in matches)),
            "candidate_decisions": len(wall),
            "median_decision_wall_seconds": statistics.median(wall) if wall else None,
            "mean_decision_wall_seconds": statistics.mean(wall) if wall else None,
            "median_decision_cpu_seconds": statistics.median(cpu) if cpu else None,
            "mean_decision_cpu_seconds": statistics.mean(cpu) if cpu else None,
            "max_decision_wall_seconds": max(wall) if wall else None,
            "mean_match_wall_seconds": statistics.mean(r["process_wall_seconds"] for r in matches),
            "seat_counts": dict(Counter(r["candidate_seat"] for r in matches)),
        }
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bots", default="F,AB:2,M:10")
    parser.add_argument("--games", type=positive_int, default=12, help="Games per candidate, multiple of four.")
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--max-turns", type=positive_int, default=1000)
    parser.add_argument("--max-actions", type=positive_int, default=3000)
    parser.add_argument("--single", help=argparse.SUPPRESS)
    parser.add_argument("--seat", type=int, choices=range(4), default=0, help=argparse.SUPPRESS)
    parser.add_argument("--index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("positive finite max-seconds required")
    if args.single:
        run_match(args.single, args.seed, args.seat, args.index, args.output,
                  args.max_seconds, args.max_turns, args.max_actions)
        return 0
    if args.games % 4:
        parser.error("games must be a multiple of four for balanced seats")
    bots = args.bots.split(",")
    if len(set(bots)) != len(bots) or any(b not in ("F", "AB:2", "M:10") for b in bots):
        parser.error("distinct candidate specs from F,AB:2,M:10 are required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    write_json(args.output / "config.json", {
        "version": "paired_seed_bot_tournament_v1", "engine_revision": ENGINE_REVISION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bots": bots, "games_per_bot": args.games, "base_seed": args.seed,
        "opponents": ["F", "F", "F"], "workers": args.workers,
        "max_seconds_per_match": args.max_seconds, "hard_timeout_seconds": args.max_seconds + 25,
        "max_turns": args.max_turns, "max_actions": args.max_actions,
        "policy_information": "fresh synthetic live public-count belief at each choice; own hand preserved; real future stream detached",
        "decision_timing_scope": "non-forced bot.decide only; excludes belief creation; wall and process CPU",
        "limitations": ["12-game pilot is not a reliable strength ranking", "compare installed algorithms and budgets only",
                        "single sampled hidden state; no history posterior", "wall timing depends on concurrent load",
                        "timeouts are censored; not losses", "no player-to-player trade negotiation"],
    })
    jobs = []
    # Interleave groups to avoid giving one algorithm only the startup workload.
    for index in range(args.games):
        for bot in bots:
            output = args.output / "matches" / bot.replace(":", "-") / f"game_{index:04d}"
            output.mkdir(parents=True)
            jobs.append((bot, args.seed + index, index % 4, index, output,
                         args.max_seconds, args.max_turns, args.max_actions))
    results = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool, (args.output / "results.jsonl").open("x", encoding="utf-8") as stream:
            futures = [pool.submit(isolated_match, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                results.append(result)
                print(f"[{len(results)}/{len(jobs)}] {result['bot']} seed={result['seed']} seat={result['candidate_seat']} {result['termination']} candidate_won={result['candidate_won']} seconds={result['process_wall_seconds']:.2f}", flush=True)
    except Exception as error:
        write_json(args.output / "FAILED.json", {"error": str(error), "partial": True})
        raise
    groups = summarize_matches(sorted(results, key=lambda r: (bots.index(r["bot"]), r["game_index"])))
    write_json(args.output / "summary.json", {"groups": groups, "elapsed_seconds": perf_counter() - started})
    print(json.dumps(groups, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
