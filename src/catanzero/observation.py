"""Shared observation whitelist and stable action ordering (no training imports)."""
import json
from catanatron.models.enums import RESOURCES, DEVELOPMENT_CARDS
from catanatron.serialization import action_to_json
from catanatron.state_functions import player_key

PUBLIC_FIELDS = (
    "VICTORY_POINTS", "ROADS_AVAILABLE", "SETTLEMENTS_AVAILABLE", "CITIES_AVAILABLE",
    "HAS_ROAD", "HAS_ARMY", "HAS_ROLLED", "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN",
    "LONGEST_ROAD_LENGTH",
) + tuple(f"PLAYED_{card}" for card in DEVELOPMENT_CARDS)


def ordered_actions(game):
    return sorted(game.playable_actions, key=lambda a: json.dumps(action_to_json(a)))


def observation(game, actor):
    """Whitelist own/private and opponents/public fields; omit deck order, RNG, seed."""
    state = game.state
    players = []
    for color in state.colors:
        key = player_key(state, color)
        player = {"color": color.value, **{f: state.player_state[f"{key}_{f}"] for f in PUBLIC_FIELDS}}
        resources = {r: state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES}
        cards = {c: state.player_state[f"{key}_{c}_IN_HAND"] for c in DEVELOPMENT_CARDS}
        player["resource_count"] = sum(resources.values())
        player["development_count"] = sum(cards.values())
        if color == actor:
            player.update(resources=resources, development_cards=cards,
                          actual_vp=state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"])
            player["playable_at_turn_start"] = {
                c: state.player_state[f"{key}_{c}_OWNED_AT_START"]
                for c in DEVELOPMENT_CARDS if c != "VICTORY_POINT"
            }
        players.append(player)
    return {
        "actor": actor.value, "turn_owner": state.colors[state.current_turn_index].value,
        "turn": state.num_turns, "prompt": state.current_prompt.value,
        "initial_phase": state.is_initial_build_phase,
        "discarding": state.is_discarding, "moving_robber": state.is_moving_knight,
        "free_roads_available": state.free_roads_available,
        "discard_counts": list(state.discard_counts), "discard_limit": state.discard_limit,
        "development_deck_size": len(state.development_listdeck), "target_vp": game.vps_to_win,
        "resource_bank": dict(zip(RESOURCES, state.resource_freqdeck)),
        "players": players,
        "buildings": [[n, c.value, kind] for n, (c, kind) in sorted(state.board.buildings.items())],
        "roads": [[list(edge), c.value] for edge, c in sorted(state.board.roads.items())],
        "robber": list(state.board.robber_coordinate),
    }
