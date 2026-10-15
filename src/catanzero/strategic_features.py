"""Deterministic graph and resource features, using predecision information only.

No Game construction, copy, execute, sampled hidden cards, or learned heuristic.
Immediate road/building effects are graph arithmetic, not future rollouts.
"""
from collections import defaultdict
from functools import lru_cache
from heapq import heappop, heappush

from catanatron.models.enums import RESOURCES
from catanatron.models.map import number_probability

RECIPES = {"road": {"WOOD": 1, "BRICK": 1},
           "settlement": {"WOOD": 1, "BRICK": 1, "SHEEP": 1, "WHEAT": 1},
           "city": {"ORE": 3, "WHEAT": 2},
           "development": {"ORE": 1, "SHEEP": 1, "WHEAT": 1}}
EXTRA_GROUPS = {
    "strategic_points": ["own_settlement_count", "own_city_count", "own_building_vp", "own_road_bonus_vp",
                         "own_army_bonus_vp", "own_hidden_vp", "own_points_to_win", "own_public_vp_gap",
                         "action_bonus_vp_gain", "action_estimated_vp_gain", "action_estimated_win"],
    "strategic_hand": [f"{prefix}_{suffix}" for prefix in ("own", "after_action") for suffix in
                       ("above_discard_limit", "excess_over_discard_limit", "discard_if_seven", "expected_discard_next_roll")]
                      + [f"{prefix}_{recipe}_{suffix}" for prefix in ("own", "after_action")
                         for recipe in RECIPES for suffix in ("resource_deficit", "can_afford")]
                      + ["own_missing_production_resources", "own_production_concentration"],
    "strategic_awards": ["own_road_award_gap", "own_road_lead", "own_army_award_gap",
                         "action_road_length_after", "action_road_length_gain", "action_road_award_gain",
                         "action_road_award_loss", "action_enemy_road_length_reduction",
                         "action_enemy_road_award_removed", "action_army_award_gain"],
    "strategic_location": ["location_yield_total", "location_yield_rank", "location_resource_diversity",
                           "location_new_resource_types", "location_robber_exposure", "location_enemy_building_distance"],
    "strategic_expansion": [f"{prefix}_{suffix}" for prefix in ("own", "after_action") for suffix in
                            ("free_sites_at_network", "free_sites_one_road", "free_sites_two_roads", "best_reachable_site_yield")]
                           + ["action_new_sites_at_network", "action_best_new_site_yield", "action_reachable_site_yield_gain"],
}
EXTRA_BINARY = {"action_estimated_win", "own_above_discard_limit", "after_action_above_discard_limit",
                "action_road_award_gain", "action_road_award_loss", "action_enemy_road_award_removed", "action_army_award_gain"}
EXTRA_BINARY.update(f"{prefix}_{recipe}_can_afford" for prefix in ("own", "after_action") for recipe in RECIPES)


@lru_cache(maxsize=4096)
def longest_road(edges, blocked):
    """Longest edge-simple trail. Enemy buildings may end, but not join, roads."""
    adjacent = defaultdict(list)
    for index, (a, b) in enumerate(edges):
        adjacent[a].append((b, 1 << index))
        adjacent[b].append((a, 1 << index))
    blocked = set(blocked)

    @lru_cache(maxsize=None)
    def walk(node, used):
        if used and node in blocked:
            return 0
        return max((1+walk(other, used | bit) for other, bit in adjacent[node] if not used & bit), default=0)

    return max((walk(node, 0) for node in tuple(adjacent)), default=0)


def award_holder(lengths, current, minimum):
    best = max(lengths.values(), default=0)
    if best < minimum:
        return None
    if current is not None and lengths[current] == best:
        return current
    leaders = [color for color, length in lengths.items() if length == best]
    return leaders[0] if len(leaders) == 1 else None


class StrategicFeatures:
    def __init__(self, catan_map):
        self.map = catan_map
        self.neighbors = {n: set() for n in catan_map.land_nodes}
        for tile in catan_map.tiles.values():
            for a, b in tile.edges.values():
                if a in self.neighbors and b in self.neighbors:
                    self.neighbors[a].add(b)
                    self.neighbors[b].add(a)
        self.yield_by_node = {}
        self.total_yield = {}
        for node in self.neighbors:
            yields = {r: sum(number_probability(t.number) for t in catan_map.adjacent_tiles[node] if t.resource == r)
                      for r in RESOURCES}
            self.yield_by_node[node] = yields
            self.total_yield[node] = sum(yields.values())
        self._key = self._context = None

    def reachable_sites(self, actor, roads, buildings):
        """Minimum additional roads, at most two, from any existing own network.

        Accounts for distance rule, occupied edges, and enemy buildings. It is
        infrastructure reachability, not affordability or initial-phase legality.
        """
        blocked = {n for n, (color, _) in buildings.items() if color != actor}
        sources = {n for n, (color, _) in buildings.items() if color == actor}
        sources.update(n for edge, color in roads.items() if color == actor for n in edge)
        sources.difference_update(blocked)
        distance = {n: 0 for n in sources}
        agenda = [(0, n) for n in sources]
        # All initial keys are zero, so a sorted list is already a valid heap.
        agenda.sort()
        while agenda:
            cost, node = heappop(agenda)
            if cost != distance[node]:
                continue
            for other in self.neighbors[node]:
                owner = roads.get(tuple(sorted((node, other))))
                if other in blocked or owner not in (None, actor):
                    continue
                following = cost + (owner is None)
                if following <= 2 and following < distance.get(other, 3):
                    distance[other] = following
                    heappush(agenda, (following, other))
        unavailable = set(buildings)
        unavailable.update(other for n in buildings for other in self.neighbors[n])
        return {node: cost for node, cost in distance.items() if node not in unavailable}

    def context(self, obs):
        # Explicit whitelist also keeps the cache independent of hidden fields.
        key = (obs["actor"], tuple(sorted(tuple(b) for b in obs["buildings"])),
               tuple(sorted(set((tuple(sorted(edge)), color) for edge, color in obs["roads"]))),
               tuple((p["color"], p["LONGEST_ROAD_LENGTH"], p["HAS_ROAD"], p["PLAYED_KNIGHT"], p["HAS_ARMY"])
                     for p in obs["players"]))
        if key == self._key:
            return self._context
        buildings = {n: (c, kind) for n, c, kind in obs["buildings"]}
        roads = {tuple(sorted(edge)): color for edge, color in obs["roads"]}
        lengths = {p["color"]: p["LONGEST_ROAD_LENGTH"] for p in obs["players"]}
        holder = next((p["color"] for p in obs["players"] if p["HAS_ROAD"]), None)
        sites = self.reachable_sites(obs["actor"], roads, buildings)
        own_edges = tuple(sorted(edge for edge, color in roads.items() if color == obs["actor"]))
        own_blocked = tuple(sorted(n for n, (color, _) in buildings.items() if color != obs["actor"]))
        self._context = {"buildings": buildings, "roads": roads, "lengths": lengths, "holder": holder, "sites": sites,
                         "own_graph_length": longest_road(own_edges, own_blocked)}
        self._key = key
        return self._context

    def site_features(self, sites, prefix):
        return {f"{prefix}_free_sites_at_network": sum(d == 0 for d in sites.values()),
                f"{prefix}_free_sites_one_road": sum(d == 1 for d in sites.values()),
                f"{prefix}_free_sites_two_roads": sum(d == 2 for d in sites.values()),
                f"{prefix}_best_reachable_site_yield": max((self.total_yield[n] for n in sites), default=0)}

    def encode(self, inputs, base):
        obs = inputs["observation"]
        actor, kind, value = inputs["action"]
        own = next(p for p in obs["players"] if p["color"] == actor)
        enemies = [p for p in obs["players"] if p["color"] != actor]
        context = self.context(obs)
        buildings, roads, lengths = context["buildings"], context["roads"], context["lengths"]
        settlement_count = sum(c == actor and k == "SETTLEMENT" for c, k in buildings.values())
        city_count = sum(c == actor and k == "CITY" for c, k in buildings.values())
        features = {"own_settlement_count": settlement_count, "own_city_count": city_count,
                    "own_building_vp": settlement_count+2*city_count,
                    "own_road_bonus_vp": 2*own["HAS_ROAD"], "own_army_bonus_vp": 2*own["HAS_ARMY"],
                    "own_hidden_vp": own["development_cards"]["VICTORY_POINT"],
                    "own_points_to_win": max(obs["target_vp"]-own["actual_vp"], 0),
                    "own_public_vp_gap": max(p["VICTORY_POINTS"] for p in enemies)-own["VICTORY_POINTS"]}
        hand = own["resources"]
        adjusted = {r: hand[r]-base[f"action_cost_{r}"]+base[f"action_gain_{r}"] for r in RESOURCES}
        if kind == "BUILD_SETTLEMENT" and obs["initial_phase"] and settlement_count == 1:
            # The second initial settlement grants one card per adjacent resource
            # tile. Tile types are known; no dice/deck outcome is consulted.
            for tile in self.map.adjacent_tiles[value]:
                if tile.resource:
                    adjusted[tile.resource] += 1
        for prefix, resources in (("own", hand), ("after_action", adjusted)):
            count = sum(resources.values())
            discards = count//2 if count > obs["discard_limit"] else 0
            features.update({f"{prefix}_above_discard_limit": count > obs["discard_limit"],
                             f"{prefix}_excess_over_discard_limit": max(count-obs["discard_limit"], 0),
                             f"{prefix}_discard_if_seven": discards,
                             f"{prefix}_expected_discard_next_roll": discards/6})
            for recipe, needs in RECIPES.items():
                deficit = sum(max(need-resources[r], 0) for r, need in needs.items())
                features[f"{prefix}_{recipe}_resource_deficit"] = deficit
                features[f"{prefix}_{recipe}_can_afford"] = deficit == 0
        production = [base[f"own_production_{r}"] for r in RESOURCES]
        total = sum(production)
        features["own_missing_production_resources"] = sum(p == 0 for p in production)
        features["own_production_concentration"] = sum((p/total)**2 for p in production) if total else 0

        after_buildings, after_roads, after_lengths = buildings, roads, lengths.copy()
        # The engine's player LONGEST_ROAD_LENGTH field stays zero in setup,
        # while the public roads already form real trails. Use graph length for
        # own geometry; retained v1 fields keep their original engine values.
        after_lengths[actor] = context["own_graph_length"]
        if kind == "BUILD_ROAD":
            after_roads = {**roads, tuple(sorted(value)): actor}
            edges = tuple(sorted(edge for edge, color in after_roads.items() if color == actor))
            blocked = tuple(sorted(n for n, (color, _) in buildings.items() if color != actor))
            after_lengths[actor] = longest_road(edges, blocked)
        elif kind == "BUILD_SETTLEMENT":
            after_buildings = {**buildings, value: (actor, "SETTLEMENT")}
            for enemy in enemies:
                color = enemy["color"]
                edges = tuple(sorted(edge for edge, c in roads.items() if c == color))
                # Only a node joining two or more enemy edges can shorten a trail.
                if sum(value in edge for edge in edges) >= 2:
                    blocked = tuple(sorted(n for n, (c, _) in after_buildings.items() if c != color))
                    after_lengths[color] = longest_road(edges, blocked)
        holder_after = award_holder(after_lengths, context["holder"], 5)
        road_gain = holder_after == actor and not own["HAS_ROAD"]
        road_loss = bool(own["HAS_ROAD"]) and holder_after != actor
        knights = {p["color"]: p["PLAYED_KNIGHT"] for p in obs["players"]}
        army_holder = next((p["color"] for p in obs["players"] if p["HAS_ARMY"]), None)
        if kind == "PLAY_KNIGHT_CARD":
            knights[actor] += 1
        army_after = award_holder(knights, army_holder, 3)
        army_gain = army_after == actor and not own["HAS_ARMY"]
        bonus_delta = 2*(int(road_gain)-int(road_loss)+int(army_gain))
        known_vp_gain = base["vp_gain"]+bonus_delta
        features.update({"own_road_award_gap": 0 if own["HAS_ROAD"] else max(max(5, max(lengths[p["color"]] for p in enemies)+1)-context["own_graph_length"], 0),
                         "own_road_lead": context["own_graph_length"]-max(lengths[p["color"]] for p in enemies),
                         "own_army_award_gap": 0 if own["HAS_ARMY"] else max(max(3, max(p["PLAYED_KNIGHT"] for p in enemies)+1)-own["PLAYED_KNIGHT"], 0),
                         "action_road_length_after": after_lengths[actor], "action_road_length_gain": after_lengths[actor]-context["own_graph_length"],
                         "action_road_award_gain": road_gain, "action_road_award_loss": road_loss,
                         "action_enemy_road_length_reduction": max(lengths[p["color"]]-after_lengths[p["color"]] for p in enemies),
                         "action_enemy_road_award_removed": context["holder"] not in (None, actor) and holder_after != context["holder"],
                         "action_army_award_gain": army_gain,
                         "action_bonus_vp_gain": bonus_delta, "action_estimated_vp_gain": known_vp_gain,
                         "action_estimated_win": own["actual_vp"]+known_vp_gain >= obs["target_vp"]})
        location = [value] if kind in ("BUILD_SETTLEMENT", "BUILD_CITY") else list(value) if kind == "BUILD_ROAD" else []
        quality = sum(self.total_yield[n] for n in location)/len(location) if location else 0
        alternatives = []
        if location:
            for _, other_kind, other_value in inputs["legal_actions"]:
                if other_kind == kind:
                    nodes = list(other_value) if kind == "BUILD_ROAD" else [other_value]
                    alternatives.append(sum(self.total_yield[n] for n in nodes)/len(nodes))
        potential = [base[f"location_potential_{r}"] for r in RESOURCES]
        features.update({"location_yield_total": quality,
                         "location_yield_rank": sum(q <= quality+1e-12 for q in alternatives)/len(alternatives) if alternatives else 0,
                         "location_resource_diversity": sum(q > 0 for q in potential),
                         "location_new_resource_types": sum(q > 0 and p == 0 for q, p in zip(potential, production)),
                         "location_robber_exposure": sum(base[f"location_potential_{r}"]-base[f"location_active_{r}"] for r in RESOURCES),
                         "location_enemy_building_distance": self.enemy_distance(location, buildings, actor)})
        sites = context["sites"]
        after_sites = self.reachable_sites(actor, after_roads, after_buildings) if kind in ("BUILD_ROAD", "BUILD_SETTLEMENT") else sites
        features.update(self.site_features(sites, "own"))
        features.update(self.site_features(after_sites, "after_action"))
        new_sites = {n for n, d in after_sites.items() if d == 0 and sites.get(n) != 0}
        features["action_new_sites_at_network"] = len(new_sites)
        features["action_best_new_site_yield"] = max((self.total_yield[n] for n in new_sites), default=0)
        features["action_reachable_site_yield_gain"] = features["after_action_best_reachable_site_yield"]-features["own_best_reachable_site_yield"]
        return features

    def enemy_distance(self, sources, buildings, actor):
        if not sources:
            return 0
        enemies = {n for n, (color, _) in buildings.items() if color != actor}
        if not enemies:
            return len(self.neighbors)
        seen, agenda = set(sources), [(n, 0) for n in sources]
        for node, distance in agenda:
            if node in enemies:
                return distance
            for other in sorted(self.neighbors[node]):
                if other not in seen:
                    seen.add(other)
                    agenda.append((other, distance+1))
        return len(self.neighbors)
