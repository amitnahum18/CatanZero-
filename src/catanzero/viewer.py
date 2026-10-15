"""Export a saved game as a standalone, offline HTML replay."""

import argparse
import json
from math import cos, pi, sin, sqrt
from pathlib import Path

from catanatron.models.enums import RESOURCES, ActionType
from catanatron.models.map import LandTile, Port
from catanatron.serialization import action_record_from_json, state_to_json
from catanatron.state_functions import player_key

from catanzero.simulation import SimulationConfig, create_game


def replay_data(document: dict) -> dict:
    """Replay recorded outcomes, rather than sample new rolls or decisions."""
    game = create_game(SimulationConfig(players=len(document["colors"])), document["game"]["seed"])
    initial = state_to_json(game)
    if initial["map"] != document["map"] or initial["colors"] != document["colors"]:
        raise ValueError("Saved game does not match this engine's seeded base-board setup")
    nodes = {}
    tiles = []
    for coordinate, tile in game.state.board.map.tiles.items():
        x = sqrt(3) * 64 * (coordinate[0] + coordinate[2] / 2)
        y = 96 * coordinate[2]
        if isinstance(tile, LandTile):
            tiles.append({"x": x, "y": y, "coordinate": coordinate, "kind": "land", "resource": tile.resource, "number": tile.number})
            for direction, node_id in tile.nodes.items():
                angle = (-90 + ["NORTH", "NORTHEAST", "SOUTHEAST", "SOUTH", "SOUTHWEST", "NORTHWEST"].index(direction.value) * 60) * pi / 180
                nodes[str(node_id)] = [round(x + 64 * cos(angle), 4), round(y + 64 * sin(angle), 4)]
        elif isinstance(tile, Port):
            tiles.append({"x": x, "y": y, "coordinate": coordinate, "kind": "port", "resource": tile.resource})

    dice = None

    def snapshot(action=None):
        players = []
        for color in game.state.colors:
            key = player_key(game.state, color)
            state = game.state.player_state
            players.append({
                "color": color.value,
                "vp": state[f"{key}_ACTUAL_VICTORY_POINTS"],
                "public_vp": state[f"{key}_VICTORY_POINTS"],
                "resources": [state[f"{key}_{resource}_IN_HAND"] for resource in RESOURCES],
                "road": state[f"{key}_HAS_ROAD"],
                "army": state[f"{key}_HAS_ARMY"],
                "knights": state[f"{key}_PLAYED_KNIGHT"],
            })
        winner = game.winning_color()
        return {
            "turn": game.state.num_turns,
            "current": game.state.current_color().value,
            "winner": winner.value if winner else None,
            "buildings": [[node, color.value, kind] for node, (color, kind) in game.state.board.buildings.items()],
            "roads": [[list(edge), color.value] for edge, color in game.state.board.roads.items()],
            "robber": list(game.state.board.robber_coordinate),
            "players": players, "dice": dice, "action": action,
        }

    frames = [snapshot()]
    for payload in document["action_records"]:
        record = action_record_from_json(payload)
        # The engine records sampled rolls/cards in the action value; the
        # legal decision itself has value=None for these two action types.
        action = record.action
        if action.action_type in (ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD):
            action = action._replace(value=None)
        game.execute(action, action_record=record)
        if record.action.action_type.value == "ROLL":
            dice = payload[1]
        frames.append(snapshot(payload))
    # Confirm that the displayed replay ends on the actual persisted game.
    replayed = state_to_json(game)
    for field in ("board", "player_state", "num_turns", "action_records", "resource_freqdeck", "development_listdeck"):
        if replayed[field] != document[field]:
            raise ValueError(f"Replay mismatch in {field}")
    return {"seed": game.seed, "nodes": nodes, "tiles": tiles, "frames": frames}


def export_replay(source: Path, output: Path | None = None) -> Path:
    document = json.loads(source.read_text(encoding="utf-8"))
    data = replay_data(document)
    template = Path(__file__).with_name("replay.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    destination = output or source.with_suffix(".html")
    if destination.resolve() == source.resolve():
        raise ValueError("The replay output must not overwrite the source game")
    destination.write_text(template.replace("__REPLAY_DATA__", payload), encoding="utf-8")
    return destination


def main():
    parser = argparse.ArgumentParser(description="Export a saved Catan game to an offline HTML replay")
    parser.add_argument("game", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(export_replay(args.game, args.output).resolve())


if __name__ == "__main__":
    main()
