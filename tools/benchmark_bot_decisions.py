"""Bounded, single-process timing of built-in bots on recorded positions."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from catanatron.models.enums import ActionType
from catanatron import Color
from catanatron.serialization import action_record_from_json, state_from_json, state_to_json
from catanatron.registry import REGISTRY
import catanatron.players

from catanzero.simulation import SimulationConfig, create_game


def worker(path, spec, trials):
    # Avoid creating an unbounded multiprocessing pool for this timing probe.
    import catanatron.players.playouts as playouts
    playouts.USE_MULTIPROCESSING = False
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    timings = []
    for _ in range(trials):
        players = [REGISTRY.build("R", Color[color]) for color in document["colors"]]
        game = state_from_json(document, players)
        color = game.state.current_color()
        bot = REGISTRY.build(spec, color)
        started = time.perf_counter()
        action = bot.decide(game, game.playable_actions)
        timings.append(time.perf_counter() - started)
        assert action in game.playable_actions
    print("TIMING=" + json.dumps({"median_seconds": statistics.median(timings), "trials": trials}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--worker")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    if args.worker:
        worker(args.source, args.worker, args.trials)
        return
    document = json.loads(args.source.read_text(encoding="utf-8"))
    game = create_game(SimulationConfig(players=len(document["colors"])), document["game"]["seed"])
    output = Path("runs") / "bot-timing"
    output.mkdir(exist_ok=True)
    positions = []
    # Initial placement and states with real choices in early/middle/late play.
    thresholds = [0, 80, 300, 800]
    for index in range(len(document["action_records"]) + 1):
        if thresholds and index >= thresholds[0] and len(game.playable_actions) > 1:
            path = output / f"position_{index}.json"
            path.write_text(json.dumps(state_to_json(game)), encoding="utf-8")
            positions.append((path, index, len(game.playable_actions)))
            thresholds.pop(0)
        if index < len(document["action_records"]):
            record = action_record_from_json(document["action_records"][index])
            action = record.action
            if action.action_type in (ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD):
                action = action._replace(value=None)
            game.execute(action, action_record=record)
    results = []
    for spec in ("R", "W", "VP", "F", "AB:2", "SAB:2", "M:10", "G:25"):
        for path, index, legal_actions in positions:
            row = {"bot": spec, "action_index": index, "legal_actions": legal_actions}
            started = time.perf_counter()
            try:
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), str(path), "--worker", spec, "--trials", str(args.trials)],
                    capture_output=True, text=True, timeout=args.timeout,
                )
                if result.returncode:
                    row["error"] = result.stderr[-2000:]
                else:
                    line = next(line for line in result.stdout.splitlines() if line.startswith("TIMING="))
                    row.update(json.loads(line[7:]))
            except subprocess.TimeoutExpired:
                row["timeout_seconds_including_startup"] = args.timeout
            row["probe_wall_seconds"] = round(time.perf_counter() - started, 3)
            results.append(row)
            print(json.dumps(row), flush=True)
            (output / "decision_timings.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
