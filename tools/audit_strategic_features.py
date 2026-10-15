"""Compare predecision deterministic features with real pinned-engine outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from catanatron.serialization import action_record_from_json, map_to_json, action_to_json
from catanzero.simulation import create_game, SimulationConfig
from catanzero.dataset import observation, ordered_actions
from catanzero.features import FeatureEncoder, feature_schema, STRATEGIC_VERSION
from catanzero.static_policies import forbid_forward_simulation
from catanzero.strategic_features import RECIPES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=4)
    args = parser.parse_args()
    checked = Counter()
    graph_engine_differences = []
    for directory in sorted(args.matches.glob("game_*"))[:args.games]:
        result = json.loads((directory / "result.json").read_text())
        document = json.loads((directory / "game.json").read_text())
        game = create_game(SimulationConfig(), result["seed"])
        encoder = FeatureEncoder(map_to_json(game.state.board.map), version=STRATEGIC_VERSION)
        for payload in document["action_records"]:
            record = action_record_from_json(payload)
            action = record.action
            if action.action_type.value in ("BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY", "PLAY_KNIGHT_CARD"):
                inputs = {"observation": observation(game, action.color), "map_ref": 0,
                          "legal_actions": [action_to_json(a) for a in ordered_actions(game)], "action": action_to_json(action)}
                with forbid_forward_simulation():
                    features = dict(zip(feature_schema(STRATEGIC_VERSION)["feature_names"], encoder.encode(inputs)))
                own = next(p for p in inputs["observation"]["players"] if p["color"] == action.color.value)
                # Only the audit executes a deterministic candidate, after X has
                # been constructed and the no-simulation guard has closed.
                child = game.copy()
                child.random = child.state.random = random.Random(1)
                child.execute(action)
                after = next(p for p in observation(child, action.color)["players"] if p["color"] == action.color.value)
                assert after["actual_vp"]-own["actual_vp"] == features["action_estimated_vp_gain"]
                if child.state.board.road_lengths[action.color] != features["action_road_length_after"]:
                    graph_engine_differences.append({"game": directory.name, "action_index": len(game.state.action_records),
                        "engine_length": child.state.board.road_lengths[action.color],
                        "graph_length": features["action_road_length_after"]})
                cards = sum(after["resources"].values())
                discards = cards//2 if cards > inputs["observation"]["discard_limit"] else 0
                assert discards == features["after_action_discard_if_seven"]
                for recipe, needs in RECIPES.items():
                    deficit = sum(max(need-after["resources"][r], 0) for r, need in needs.items())
                    assert deficit == features[f"after_action_{recipe}_resource_deficit"]
                checked[action.action_type.value] += 1
            game.execute(action, validate_action=False, action_record=record)
    report = {"passed": True, "source": str(args.matches.resolve()), "checked_by_action_type": dict(checked),
              "checked_decisions": sum(checked.values()), "comparisons": ["estimated immediate VP change in sampled actions",
              "resource discard risk after action", "all four building resource deficits after action"],
              "graph_length_vs_engine_cache_differences": graph_engine_differences,
              "road_projection_limit": "graph trail length can differ from pinned engine cached lengths near enemy buildings; award/VP fields are estimates, not an oracle",
              "feature_extraction_guarded_against_simulation": True, "scope": "sampled deterministic decisions; no game-strength claim"}
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
