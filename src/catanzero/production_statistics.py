"""Experimental potential production moments from public tiles and own buildings.

No sampled roll, simulation, hidden cards, or future labels. Resource-bank
shortages are not modeled: these are potential production statistics.
"""
from catanatron.models.enums import RESOURCES
from catanatron.models.map import number_probability
from catanatron.serialization import map_from_json


def income_moments(payouts):
    mean = {r: sum(number_probability(n)*payouts[n].get(r, 0) for n in range(2,13)) for r in RESOURCES}
    totals = {n: sum(payouts[n].values()) for n in range(2,13)}
    total_mean = sum(mean.values())
    out = {"mean_income": total_mean,
           "income_variance": max(sum(number_probability(n)*totals[n]**2 for n in range(2,13))-total_mean**2, 0),
           "probability_no_income": sum(number_probability(n) for n in range(2,13) if totals[n] == 0)}
    for first, second in (("WHEAT", "ORE"), ("WOOD", "BRICK")):
        product_mean = sum(number_probability(n)*payouts[n].get(first,0)*payouts[n].get(second,0) for n in range(2,13))
        out[f"covariance_{first}_{second}"] = product_mean-mean[first]*mean[second]
    return out


class ProductionStatistics:
    def __init__(self, map_document):
        self.map = map_from_json(map_document)

    def moments(self, buildings, robber, actor):
        payouts = {n: {r: 0 for r in RESOURCES} for n in range(2,13)}
        blocked_tile = self.map.tiles[tuple(robber)]
        for node, color, kind in buildings:
            if color != actor:
                continue
            for tile in self.map.adjacent_tiles[node]:
                if tile.resource and tile is not blocked_tile:
                    payouts[tile.number][tile.resource] += 2 if kind == "CITY" else 1
        return income_moments(payouts)

    def features(self, inputs):
        obs = inputs['observation']
        actor, kind, value = inputs['action']
        before = self.moments(obs['buildings'], obs['robber'], actor)
        buildings, robber = obs['buildings'], obs['robber']
        if kind == 'BUILD_SETTLEMENT':
            buildings = [*buildings, [value, actor, 'SETTLEMENT']]
        elif kind == 'BUILD_CITY':
            buildings = [[node,color,'CITY' if node==value and color==actor else building_kind]
                         for node,color,building_kind in buildings]
        elif kind == 'MOVE_ROBBER':
            robber = value[0]
        after = self.moments(buildings, robber, actor)
        # Mean production already exists in the normal feature schema. Focus
        # this prototype on variance, zero-income risk and resource covariance.
        keys = ('income_variance','probability_no_income','covariance_WHEAT_ORE','covariance_WOOD_BRICK')
        return {**{f'stat_current_{key}':before[key] for key in keys},
                **{f'stat_delta_{key}':after[key]-before[key] for key in keys}}
