"""Real engine integration: legality, private observations, bot play and recovery."""
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import Action, ActionRecord, ActionType
from catanzero.dataset import ordered_actions, observation
from catanzero.play_server import Table, PlayServer, Conflict, load_models
from catanatron.serialization import action_to_json, state_to_json


@pytest.mark.parametrize('bots', [['402'], ['F', 'F_static'], ['402', 'F', 'F_static']])
def test_legal_setup_distance_rule_and_bot_decisions(bots, tmp_path):
    table = Table(bots, seed=1950 + len(bots))
    while table.game.state.is_initial_build_phase:
        if table.game.state.current_color() == table.human:
            table.act(table.revision, 0)
        else:
            table.act(table.revision, advance=True)
        occupied = set(table.game.state.board.buildings)
        assert all(not set(STATIC_GRAPH.neighbors(n)).intersection(occupied) for n in occupied)
    assert len(table.game.state.board.buildings) == 2*(len(bots)+1)
    # The engine indexes every physical road in both directions.
    assert len({tuple(sorted(e)) for e in table.game.state.board.roads}) == 2*(len(bots)+1)
    assert all(len(t['edge']) == 2 for t in table.geometry['tiles'] if t['kind'] == 'port')
    # Test actual play beyond opening, including random rolls and turn changes.
    for _ in range(25):
        if table.game.winning_color(): break
        if table.game.state.current_color() == table.human:
            actions = ordered_actions(table.game)
            preferred = next((i for i, a in enumerate(actions)
                              if a.action_type in (ActionType.ROLL, ActionType.END_TURN)), 0)
            table.act(table.revision, preferred)
        else:
            table.act(table.revision, advance=True)
    table.save(tmp_path)
    restored = Table.restore(tmp_path / f'{table.id}.json')
    assert restored.public_view() == table.public_view()
    assert state_to_json(restored.game) == state_to_json(table.game)
    if table.game.state.current_color() != table.human:
        assert restored.choose_bot() == table.choose_bot()


def test_private_hand_deck_stream_and_events_never_reach_browser():
    table = Table(['F_static'], seed=1900)
    other = next(iter(table.bots))
    records = table.game.state.action_records
    records.extend([
        ActionRecord(Action(other, ActionType.BUY_DEVELOPMENT_CARD, 'KNIGHT'), 'KNIGHT'),
        ActionRecord(Action(other, ActionType.DISCARD_RESOURCE, 'ORE'), 'ORE'),
        ActionRecord(Action(other, ActionType.MOVE_ROBBER, ((0, 0, 0), None)), 'WHEAT'),
    ])
    view = table.public_view()
    opponent = next(p for p in view['players'] if not p['human'])
    for key in ('resources', 'development_cards', 'actual_vp', 'playable_at_turn_start'):
        assert key not in opponent
    for key in ('seed', 'random_state', 'development_listdeck', 'action_records'):
        assert key not in view
    for event in view['log']:
        assert event['result'] is None
        if event['type'] != 'MOVE_ROBBER': assert event['value'] is None
    assert view['players'][0]['resources'] is not None


def test_save_retries_transient_windows_file_lock(tmp_path, monkeypatch):
    table = Table(['F_static'], seed=1902)
    original = Path.replace
    calls = []
    def temporarily_locked(path, target):
        calls.append(path)
        if len(calls) < 3:
            raise PermissionError('temporary file sharing violation')
        return original(path, target)
    monkeypatch.setattr(Path, 'replace', temporarily_locked)
    table.save(tmp_path)
    assert len(calls) == 3
    assert Table.restore(tmp_path / f'{table.id}.json').public_view() == table.public_view()


def test_402_uses_original_frozen_model_and_candidate_features():
    table = Table(['402'], seed=2001)
    table.act(0, 0)
    table.act(1, 0)
    assert table.game.state.current_color() != table.human
    actions = ordered_actions(table.game)
    choice = table.choose_bot()
    ranker, critic, doc, norm = load_models()
    encoder = table.encoders[norm['feature_version']]
    inputs = {'map_ref': 0, 'observation': observation(table.game, table.game.state.current_color()),
              'legal_actions': [action_to_json(a) for a in actions]}
    raw = np.asarray([encoder.encode({**inputs, 'action': a}) for a in inputs['legal_actions']])
    scores = ranker.predict(raw, num_threads=1)/doc['scale'] + doc['mix']*critic.predict(raw)
    assert choice == actions[int(np.argmax(scores))]


def test_reject_stale_and_illegal_move_without_changing_game():
    table = Table(['F_static'], seed=1901)
    original = state_to_json(table.game)
    with pytest.raises(Conflict): table.act(42, 0)
    with pytest.raises(ValueError): table.act(0, -1)
    with pytest.raises(Conflict): table.act(0, advance=True)
    assert state_to_json(table.game) == original
    table.act(0, 0)
    with pytest.raises(Conflict): table.act(0, 0)


def test_complete_match_402_and_real_rules_to_victory():
    from catanzero.regression_search import public_f_choice
    table = Table(['402'], seed=20101015)
    kinds = set()
    for _ in range(3000):
        if table.game.winning_color(): break
        if table.game.state.current_color() == table.human:
            actions = ordered_actions(table.game)
            chosen = public_f_choice(table.game, actions, 10_000 + table.revision)
            kinds.add(chosen.action_type.value)
            table.act(table.revision, actions.index(chosen))
        else:
            chosen = table.choose_bot()
            kinds.add(chosen.action_type.value)
            table.act(table.revision, advance=True)
    assert table.game.winning_color() is not None
    final = table.public_view()
    assert final['winner'] and not final['human_turn'] and final['legal_actions'] == []
    assert {'ROLL', 'MOVE_ROBBER', 'BUILD_SETTLEMENT', 'BUILD_ROAD', 'END_TURN'} <= kinds
    with pytest.raises(ValueError): table.act(table.revision, 0)


def bank_trade_position(rate):
    """Engine-owned test match; never alters a saved human match."""
    from catanatron.models.actions import generate_playable_actions
    from catanatron.state_functions import player_key
    from catanatron.models.enums import RESOURCES
    table = Table(['F_static'], seed=20261019 + rate)
    ports = [t for t in table.geometry['tiles'] if t['kind'] == 'port']
    coastal = {n for t in ports for n in t['edge']}
    target = next((t for t in ports if t['resource'] == ('WHEAT' if rate == 2 else None)), None)
    human_settlements = 0
    while table.game.state.is_initial_build_phase:
        if table.game.state.current_color() != table.human:
            table.act(table.revision, advance=True)
            continue
        actions = ordered_actions(table.game)
        index = 0
        if actions[0].action_type == ActionType.BUILD_SETTLEMENT:
            candidates = [i for i, a in enumerate(actions) if a.value not in coastal]
            if human_settlements == 0 and rate != 4:
                candidates = [i for i, a in enumerate(actions) if a.value in target['edge']]
            assert candidates
            index = candidates[0]
            human_settlements += 1
        table.act(table.revision, index)
    assert table.game.state.current_color() == table.human
    key = player_key(table.game.state, table.human)
    def set_hand():
        # Return old cards to the bank before allocating a controlled hand.
        for i, resource in enumerate(RESOURCES):
            old = table.game.state.player_state[f'{key}_{resource}_IN_HAND']
            count = rate if resource == 'WHEAT' else 0
            table.game.state.resource_freqdeck[i] += old - count
            table.game.state.player_state[f'{key}_{resource}_IN_HAND'] = count
            assert table.game.state.resource_freqdeck[i] >= 0
        table.game.playable_actions = generate_playable_actions(table.game.state)
    set_hand()
    assert not any(a.action_type == ActionType.MARITIME_TRADE for a in ordered_actions(table.game))
    roll = next(a for a in ordered_actions(table.game) if a.action_type == ActionType.ROLL)
    table.game.execute(roll, action_record=ActionRecord(roll, (1, 1)))
    set_hand()
    return table


@pytest.mark.parametrize('rate', [4, 3, 2])
def test_bank_trade_http_debits_cards_and_persists_result(rate, tmp_path):
    table = bank_trade_position(rate)
    table.save(tmp_path)
    server = PlayServer(('127.0.0.1', 0), tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base+'/api/game/'+table.id) as response:
            before = json.load(response)
        chosen = next(a for a in before['legal_actions'] if a['type'] == 'MARITIME_TRADE'
                      and a['value'][-1] == 'BRICK'
                      and [r for r in a['value'][:-1] if r] == ['WHEAT']*rate)
        payload = {'id': table.id, 'revision': before['revision'], 'index': chosen['index']}
        request = Request(base+'/api/action', json.dumps(payload).encode(),
                          {'Content-Type': 'application/json'})
        with urlopen(request) as response:
            after = json.load(response)
        hand = next(p for p in after['players'] if p['human'])
        assert hand['resources']['WHEAT'] == 0
        assert hand['resources']['BRICK'] == 1 and hand['resource_count'] == 1
        assert after['resource_bank']['WHEAT'] == before['resource_bank']['WHEAT'] + rate
        assert after['resource_bank']['BRICK'] == before['resource_bank']['BRICK'] - 1
        assert after['revision'] == before['revision'] + 1 and after['human_turn']
        assert after['log'][-1]['type'] == 'MARITIME_TRADE'
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 409  # A double click cannot trade twice.
        assert Table.restore(tmp_path / f'{table.id}.json').public_view() == after
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_game_roundtrip_and_origin_validation(tmp_path):
    server = PlayServer(('127.0.0.1', 0), tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    def post(path, payload, origin=None):
        headers = {'Content-Type': 'application/json'}
        if origin: headers['Origin'] = origin
        with urlopen(Request(base+path, json.dumps(payload).encode(), headers), timeout=30) as r:
            return json.load(r)
    try:
        with urlopen(base) as r: assert b'CATANZERO' in r.read()
        with pytest.raises(HTTPError) as error:
            post('/api/new', {'bots': ['F_static']}, 'https://example.org')
        assert error.value.code == 403
        with pytest.raises(HTTPError): post('/api/new', {'bots': []})
        game = post('/api/new', {'bots': ['F_static']})
        assert game['human_turn']
        updated = post('/api/action', {'id': game['id'], 'revision': 0, 'index': 0})
        assert updated['revision'] == 1
        with pytest.raises(HTTPError) as error:
            post('/api/action', {'id': game['id'], 'revision': 0, 'index': 0})
        assert error.value.code == 409
        server.tables.clear()  # Simulate reopening after the process loses its cache.
        with urlopen(base+'/api/game/'+game['id']) as r:
            assert json.load(r) == updated
        with pytest.raises(HTTPError): urlopen(base+'/runs/human-games/'+game['id']+'.json')
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
