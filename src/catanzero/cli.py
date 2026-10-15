"""Command-line runner for individual games and simulation batches."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import secrets
from uuid import uuid4

from catanatron.serialization import action_record_to_json, state_to_json

from catanzero.simulation import ENGINE_REVISION, SimulationConfig, run_game, summarize


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def write_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Catan simulations with random players.")
    parser.add_argument("--games", type=positive_int, default=1)
    parser.add_argument("--players", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--seed", type=int, help="base seed; game i uses seed + i (zero-based)")
    parser.add_argument("--max-turns", type=positive_int, default=1000)
    parser.add_argument("--max-actions", type=positive_int, default=100_000)
    parser.add_argument("--output", type=Path, help="new output directory; existing paths are refused")
    parser.add_argument("--save-games", action="store_true", help="save full engine states and action histories")
    parser.add_argument("--replay", action="store_true", help="save games and offline HTML board replays")
    parser.add_argument("--trace", action="store_true", help="print each executed action and its outcome")
    args = parser.parse_args(argv)
    config = SimulationConfig(args.players, args.max_turns, args.max_actions)
    base_seed = args.seed if args.seed is not None else secrets.randbits(63)
    output = args.output or Path("runs") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error(f"cannot create output directory {output}: {error}")

    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_seed": base_seed,
        "requested_games": args.games,
        "config": asdict(config),
        "player_type": "SimulationRandomPlayer",
        "engine": {"name": "catanatron", "version": version("catanatron"), "revision": ENGINE_REVISION},
        "python_version": platform.python_version(),
    }
    write_json(output / "config.json", metadata)
    print(f"CatanZero | {args.games} game(s) | {args.players} random players | seed={base_seed}")
    results = []

    def trace(game, record):
        print(f"  turn={game.state.num_turns} {action_record_to_json(record)}")

    # Write each completed result immediately, retaining useful partial batches.
    with (output / "results.jsonl").open("x", encoding="utf-8") as stream:
        for index in range(args.games):
            game, result = run_game(config, base_seed + index, trace if args.trace else None)
            results.append(result)
            stream.write(json.dumps({"game_index": index, **result.to_dict()}) + "\n")
            stream.flush()
            if args.save_games or args.replay:
                source = output / f"game_{index:04d}.json"
                write_json(source, state_to_json(game))
                if args.replay:
                    from catanzero.viewer import export_replay

                    replay = export_replay(source)
                    print(f"Replay: {replay.resolve()}")
            scores = ", ".join(f"{color}={points}" for color, points in result.victory_points.items())
            print(
                f"[{index + 1}/{args.games}] seed={result.seed} "
                f"winner={result.winner or 'none'} ({result.termination}) "
                f"turns={result.turns} actions={result.actions} | VP: {scores}"
            )
    summary = summarize(results)
    write_json(output / "summary.json", summary)
    print(f"Completed: {summary['completed_games']} | Truncated: {summary['truncated_games']}")
    print(f"Results: {output.resolve()}")
    return 0
