"""Known resource/piece capacity and current legal construction opportunities."""
from catanzero.strategic_features import RECIPES

KINDS = {"road": "BUILD_ROAD", "settlement": "BUILD_SETTLEMENT", "city": "BUILD_CITY", "development": "BUY_DEVELOPMENT_CARD"}
BUILD_GROUPS = {"building_capacity":
    [f"{prefix}_{recipe}_{suffix}" for prefix in ("own", "after_action") for recipe in RECIPES
     for suffix in ("resource_capacity", "piece_limited_capacity")]
    + [f"own_{recipe}_{suffix}" for recipe in RECIPES for suffix in ("legal_action_count", "can_build_now")]
    + [f"{prefix}_{suffix}" for prefix in ("own", "after_action") for suffix in
       ("free_road_capacity", "funded_road_capacity", "can_fund_two_roads", "can_fund_road_and_settlement", "can_fund_two_roads_and_settlement")]
    + ["own_two_road_sequence_feasible", "own_playable_road_card_capacity"]}
BUILD_BINARY = {f"own_{recipe}_can_build_now" for recipe in RECIPES}
BUILD_BINARY.update(f"{prefix}_{suffix}" for prefix in ("own", "after_action") for suffix in
                    ("can_fund_two_roads", "can_fund_road_and_settlement", "can_fund_two_roads_and_settlement"))
BUILD_BINARY.add("own_two_road_sequence_feasible")


def geometric_road_options(graph, actor, roads, buildings):
    sources = {n for n, (color, _) in buildings.items() if color == actor}
    sources.update(n for edge, color in roads.items() if color == actor for n in edge)
    sources.difference_update(n for n, (color, _) in buildings.items() if color != actor)
    return {tuple(sorted((n, other))) for n in sources for other in graph.neighbors[n]
            if tuple(sorted((n, other))) not in roads}


def build_features(inputs, base, graph):
    obs = inputs["observation"]
    actor, kind, value = inputs["action"]
    own = next(p for p in obs["players"] if p["color"] == actor)
    context = graph.context(obs)
    buildings, roads = context["buildings"], context["roads"]
    settlements = sum(c == actor and k == "SETTLEMENT" for c, k in buildings.values())
    hand = own["resources"]
    adjusted = {r: hand[r]-base[f"action_cost_{r}"]+base[f"action_gain_{r}"] for r in hand}
    if kind == "BUILD_SETTLEMENT" and obs["initial_phase"] and settlements == 1:
        for tile in graph.map.adjacent_tiles[value]:
            if tile.resource:
                adjusted[tile.resource] += 1
    pieces = {"road": own["ROADS_AVAILABLE"], "settlement": own["SETTLEMENTS_AVAILABLE"],
              "city": min(own["CITIES_AVAILABLE"], settlements), "development": obs["development_deck_size"]}
    after_pieces = pieces.copy()
    if kind == "BUILD_ROAD":
        after_pieces["road"] -= 1
    elif kind == "BUILD_SETTLEMENT":
        after_pieces["settlement"] -= 1
        after_pieces["city"] = min(own["CITIES_AVAILABLE"], settlements+1)
    elif kind == "BUILD_CITY":
        after_pieces["city"] = min(max(own["CITIES_AVAILABLE"]-1, 0), max(settlements-1, 0))
        after_pieces["settlement"] += 1  # upgraded settlement piece is returned
    elif kind == "BUY_DEVELOPMENT_CARD":
        after_pieces["development"] -= 1
    legal = inputs["legal_actions"]
    initial_free = int(obs["initial_phase"] and any(a[1] == "BUILD_ROAD" for a in legal))
    free = initial_free if obs["initial_phase"] else obs["free_roads_available"]
    after_free = free
    if kind == "PLAY_ROAD_BUILDING":
        after_free = 2
    elif kind == "BUILD_ROAD":
        after_free = max(free-1, 0)
    elif kind == "BUILD_SETTLEMENT" and obs["initial_phase"]:
        after_free = 1
    out = {}
    for prefix, stock, available, free_count in (("own", hand, pieces, free), ("after_action", adjusted, after_pieces, after_free)):
        for recipe, needs in RECIPES.items():
            capacity = max(min(int(stock[r]//need) for r, need in needs.items()), 0)
            out[f"{prefix}_{recipe}_resource_capacity"] = capacity
            out[f"{prefix}_{recipe}_piece_limited_capacity"] = min(capacity, max(available[recipe], 0))
        credits = min(max(available["road"], 0), free_count)
        funded = min(max(available["road"], 0), out[f"{prefix}_road_resource_capacity"]+credits)
        out[f"{prefix}_free_road_capacity"] = credits
        out[f"{prefix}_funded_road_capacity"] = funded
        out[f"{prefix}_can_fund_two_roads"] = funded >= 2
        for count, name in ((1, "road_and_settlement"), (2, "two_roads_and_settlement")):
            paid_roads = max(count-credits, 0)
            costs = {"WOOD": paid_roads+1, "BRICK": paid_roads+1, "SHEEP": 1, "WHEAT": 1}
            out[f"{prefix}_can_fund_{name}"] = (available["road"] >= count and available["settlement"] > 0
                                                and all(stock[r] >= n for r, n in costs.items()))
    for recipe, action_kind in KINDS.items():
        count = sum(a[1] == action_kind for a in legal)
        out[f"own_{recipe}_legal_action_count"] = count
        out[f"own_{recipe}_can_build_now"] = count > 0
    out["own_playable_road_card_capacity"] = min(pieces["road"], 2) if any(a[1] == "PLAY_ROAD_BUILDING" for a in legal) else 0
    # Two consecutive road placements require current legality and enough
    # resources/active free credits. No unknown rolls or hypothetical trades.
    can_pay_in_phase = free >= 2 or bool(own["HAS_ROLLED"])
    possible = False
    if not obs["initial_phase"] and can_pay_in_phase and out["own_can_fund_two_roads"]:
        for _, action_kind, edge in legal:
            if action_kind == "BUILD_ROAD":
                after_roads = {**roads, tuple(sorted(edge)): actor}
                if geometric_road_options(graph, actor, after_roads, buildings):
                    possible = True
                    break
    out["own_two_road_sequence_feasible"] = possible
    return out
