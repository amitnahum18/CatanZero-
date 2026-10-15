"""Shared local/browser game session; inference imports are loaded on demand."""
from functools import lru_cache
import json
from math import cos, sin, pi, sqrt
from pathlib import Path
import random
import secrets
from time import sleep

from catanatron.models.enums import RESOURCES, ActionType
from catanatron.models.map import LandTile, Port
from catanatron.models.player import Color
from catanatron.serialization import action_to_json, map_to_json, state_to_json, state_from_json, action_record_to_json
from catanzero.observation import observation, ordered_actions
from catanzero.features import FeatureEncoder
from catanzero.simulation import SimulationConfig, SimulationRandomPlayer, create_game

ROOT = Path(__file__).resolve().parents[2]
CATALOG = [{'id': '402', 'name': 'Model 402', 'description': 'Frozen trained model 402'}]
MODEL_IDS = {'402', 'F', 'F_static'}

@lru_cache(maxsize=1)
def load_models():
    import joblib
    import lightgbm as lgb
    directory = ROOT / 'models/champion402'
    if not directory.is_dir():
        directory = ROOT / 'runs/tree-policy-trade-aware-v3'
    doc = json.loads((directory / 'best_model/policy.json').read_text(encoding='utf-8'))
    norm = json.loads((directory / 'normalizer.json').read_text(encoding='utf-8'))
    ranker = lgb.Booster(model_file=str(directory / 'best_model/ranker.model'))
    critic = joblib.load(directory / 'best_model/monte_carlo.joblib')
    critic.n_jobs = 1
    return ranker, critic, doc, norm


def board_geometry(game):
    tiles, nodes = [], {}
    for coord, tile in game.state.board.map.tiles.items():
        x, y = sqrt(3) * 64 * (coord[0] + coord[2] / 2), 96 * coord[2]
        if isinstance(tile, LandTile):
            tiles.append(dict(x=x, y=y, coordinate=list(coord), kind='land',
                              resource=tile.resource, number=tile.number))
            for direction, node in tile.nodes.items():
                angle = (-90 + ['NORTH','NORTHEAST','SOUTHEAST','SOUTH',
                               'SOUTHWEST','NORTHWEST'].index(direction.value)*60)*pi/180
                nodes[str(node)] = [round(x+64*cos(angle), 4), round(y+64*sin(angle), 4)]
        elif isinstance(tile, Port):
            # Mark the exact two coastal intersections that own this port.
            edge = next((edge for direction, edge in tile.edges.items()
                         if direction.value == tile.direction.value), None)
            tiles.append(dict(x=x, y=y, coordinate=list(coord), kind='port',
                              resource=tile.resource, edge=list(edge) if edge else []))
    return {'tiles': tiles, 'nodes': nodes}


class Table:
    def __init__(self, bots, name='You', *, seed=None):
        if not isinstance(bots, list) or not 1 <= len(bots) <= 3 or any(b not in MODEL_IDS for b in bots):
            raise ValueError('Choose 1 to 3 bots.')
        if not isinstance(name, str) or len(name) > 24:
            raise ValueError('Your name must be 24 characters or fewer.')
        self.id = secrets.token_hex(16)
        self.game = create_game(SimulationConfig(players=len(bots)+1),
                                seed if seed is not None else secrets.randbits(48))
        self.human = self.game.state.colors[0]
        self.bots = dict(zip(self.game.state.colors[1:], bots))
        self.name = name.strip() or 'You'
        self._prepare()

    def _prepare(self):
        self.geometry = board_geometry(self.game)
        self.map_doc = map_to_json(self.game.state.board.map)
        self.encoders = {}

    @property
    def revision(self):
        return len(self.game.state.action_records)

    def choose_bot(self):
        import numpy as np
        from catanzero.static_policies import ObservationPolicy, forbid_forward_simulation
        actor = self.game.state.current_color()
        actions = ordered_actions(self.game)
        model = self.bots[actor]
        seed = self.game.seed*1_000_003 + self.revision
        rng_before = self.game.state.random.getstate()
        if len(actions) == 1:
            chosen = actions[0]
        elif model == 'F':
            from catanzero.regression_search import public_f_choice
            chosen = public_f_choice(self.game, actions, seed)
        else:
            inputs = {'map_ref': 0, 'observation': observation(self.game, actor),
                      'legal_actions': [action_to_json(a) for a in actions]}
            with forbid_forward_simulation():
                if model == 'random':
                    chosen = random.Random(seed).choice(actions)
                elif model == 'F_static':
                    if not hasattr(self, 'static'):
                        self.static = ObservationPolicy('F_static', self.map_doc, {}, allow_fewer_players=True)
                    index, _ = self.static.choose(inputs)
                    chosen = actions[index]
                else:
                    ranker, critic, doc, norm = load_models()
                    if norm['feature_version'] not in self.encoders:
                        self.encoders[norm['feature_version']] = FeatureEncoder(
                            self.map_doc, version=norm['feature_version'], allow_fewer_players=True)
                    encoder = self.encoders[norm['feature_version']]
                    raw = np.asarray([encoder.encode({**inputs, 'action': a})
                                      for a in inputs['legal_actions']], dtype=np.float64)
                    scores = (ranker.predict(raw, num_threads=1)/doc['scale']
                              if model != 'critic' else np.zeros(len(actions)))
                    if model != 'ranker':
                        scores += (doc['mix'] if model == '402' else 1.)*critic.predict(raw)
                    chosen = actions[int(np.argmax(scores))]
        assert self.game.state.random.getstate() == rng_before
        return chosen

    def act(self, revision, index=None, *, advance=False):
        if type(revision) is not int or revision != self.revision:
            raise Conflict('The board has changed. Refreshing the current position.')
        if self.game.winning_color():
            raise ValueError('This game has already finished.')
        if advance:
            if self.game.state.current_color() == self.human:
                raise Conflict('It is your turn to choose a move.')
            chosen = self.choose_bot()
        else:
            if self.game.state.current_color() != self.human:
                raise Conflict('Waiting for the opponent to move.')
            actions = ordered_actions(self.game)
            if type(index) is not int or not 0 <= index < len(actions):
                raise ValueError('This move is not legal in the current position.')
            chosen = actions[index]
        self.game.execute(chosen)

    def public_view(self):
        obs = observation(self.game, self.human)
        current = self.game.state.current_color()
        winner = self.game.winning_color()
        players = []
        for seat, player in enumerate(obs['players']):
            color = Color(player['color'])
            human = color == self.human
            model = self.bots.get(color)
            players.append({**player, 'human': human, 'model': model,
                            'name': ('You' if self.name == '\u05d0\u05e0\u05d9' else self.name) if human else f'Bot {seat} · ' +
                            next((m['name'] for m in CATALOG if m['id'] == model), model)})
        logs, dice = [], None
        for record in self.game.state.action_records:
            payload, result = action_record_to_json(record)
            actor, kind, value = payload
            if kind == 'ROLL':
                dice = result
            if kind == 'BUY_DEVELOPMENT_CARD' and actor != self.human.value:
                value = result = None
            elif kind == 'MOVE_ROBBER' and actor != self.human.value and value[1] != self.human.value:
                result = None
            elif kind == 'DISCARD_RESOURCE' and actor != self.human.value:
                value = result = None
            logs.append({'actor': actor, 'type': kind, 'value': value, 'result': result})
        return {'id': self.id, 'revision': self.revision, 'human': self.human.value,
                'current': current.value, 'turn_owner': obs['turn_owner'],
                'human_turn': current == self.human and not winner,
                'winner': winner.value if winner else None, 'turn': obs['turn'],
                'prompt': obs['prompt'], 'initial_phase': obs['initial_phase'],
                'free_roads_available': obs['free_roads_available'],
                'target_vp': self.game.vps_to_win, 'players': players,
                'buildings': obs['buildings'], 'roads': obs['roads'], 'robber': obs['robber'],
                'resource_bank': obs['resource_bank'], 'development_deck_size': obs['development_deck_size'],
                'geometry': self.geometry, 'dice': dice, 'log': logs[-60:],
                'legal_actions': [dict(index=i, type=a.action_type.value, value=action_to_json(a)[2])
                                  for i, a in enumerate(ordered_actions(self.game))]
                                 if current == self.human and not winner else []}

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'{self.id}.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps({'id': self.id, 'human': self.human.value, 'name': self.name,
            'bots': {c.value: b for c, b in self.bots.items()}, 'game': state_to_json(self.game)},
            ensure_ascii=False), encoding='utf-8')
        # Windows scanners can briefly hold a just-written save file open.
        # Keep the atomic replacement and retry transient sharing violations.
        for attempt in range(8):
            try:
                temp.replace(path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                sleep(.01 * 2**attempt)

    @classmethod
    def restore(cls, path):
        data = json.loads(path.read_text(encoding='utf-8'))
        table = cls.__new__(cls)
        table.id, table.name, table.human = data['id'], data['name'], Color(data['human'])
        table.bots = {Color(c): b for c, b in data['bots'].items()}
        table.game = state_from_json(data['game'], [SimulationRandomPlayer(Color(c))
                                                   for c in data['game']['colors']])
        table._prepare()
        return table


class Conflict(ValueError):
    pass
