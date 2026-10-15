"""Worker-facing game API. Rules and observations are shared with local play."""
import json
from pathlib import Path

from catanatron.models.player import Color
from catanatron.serialization import action_to_json, state_to_json, state_from_json
from catanzero.play_core import Table, Conflict, board_geometry
from catanzero.observation import ordered_actions, observation
from catanzero.features import FeatureEncoder, feature_schema
from catanzero.champion_portable import Champion402
from catanzero.simulation import create_game, SimulationConfig, SimulationRandomPlayer


class BrowserTable(Table):
    def choose_bot(self):
        actor = self.game.state.current_color()
        actions = ordered_actions(self.game)
        if len(actions) == 1:
            return actions[0]
        inputs = {'map_ref': 0, 'observation': observation(self.game, actor),
                  'legal_actions': [action_to_json(a) for a in actions]}
        if self.policy.feature_version not in self.encoders:
            encoder = FeatureEncoder(self.map_doc, version=self.policy.feature_version,
                                     allow_fewer_players=True)
            assert feature_schema(self.policy.feature_version)['feature_names'] == self.policy.feature_names
            self.encoders[self.policy.feature_version] = encoder
        encoder = self.encoders[self.policy.feature_version]
        rows = [encoder.encode({**inputs, 'action': a}) for a in inputs['legal_actions']]
        return actions[self.policy.choose(rows)]

    def dump(self):
        # Local browser persistence only. Never uploaded to GitHub or a server.
        return {'id': self.id, 'name': self.name, 'human': self.human.value,
                'bots': {c.value: b for c, b in self.bots.items()}, 'game': state_to_json(self.game)}

    @classmethod
    def from_dump(cls, data, policy):
        table = cls.__new__(cls)
        table.id, table.name, table.human = data['id'], data['name'], Color(data['human'])
        if not data['bots'] or any(b != '402' for b in data['bots'].values()):
            raise ValueError('This saved match requires a different local bot.')
        table.bots = {Color(c): b for c, b in data['bots'].items()}
        table.game = state_from_json(data['game'], [SimulationRandomPlayer(Color(c))
                                                   for c in data['game']['colors']])
        table.policy = policy
        table._prepare()
        return table


class BrowserGame:
    def __init__(self, model_path):
        self.policy = Champion402.load(model_path)
        self.table = None

    def request(self, path, payload=None, saved=None):
        payload = payload or {}
        if path == '/api/bootstrap':
            return {'models': [{'id': '402', 'name': 'Model 402'}],
                    'preview': board_geometry(create_game(SimulationConfig(), 31415))}, None
        if path == '/api/new':
            bots = payload.get('bots')
            if not isinstance(bots, list) or not 1 <= len(bots) <= 3 or any(b != '402' for b in bots):
                raise ValueError('Choose 1 to 3 Model 402 bots.')
            self.table = BrowserTable(bots, payload.get('name', 'You'))
            self.table.policy = self.policy
        elif path.startswith('/api/game/'):
            ident = path.removeprefix('/api/game/')
            if self.table is None or self.table.id != ident:
                if not saved or saved.get('id') != ident:
                    raise KeyError('Saved match not found on this device.')
                self.table = BrowserTable.from_dump(saved, self.policy)
        elif path in ('/api/action', '/api/advance'):
            if self.table is None or payload.get('id') != self.table.id:
                raise KeyError('Match not found on this device.')
            self.table.act(payload.get('revision'), payload.get('index'), advance=path == '/api/advance')
        else:
            raise KeyError('Unknown game request.')
        return self.table.public_view(), self.table.dump()

    def request_json(self, path, payload_json='{}', saved_json='null'):
        try:
            result, saved = self.request(path, json.loads(payload_json), json.loads(saved_json))
            return json.dumps({'result': result, 'saved': saved}, separators=(',', ':'))
        except (Conflict, ValueError, KeyError) as error:
            status = 409 if isinstance(error, Conflict) else 404 if isinstance(error, KeyError) else 400
            return json.dumps({'error': str(error), 'status': status})
