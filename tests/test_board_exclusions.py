import json
import pytest
from catanatron.serialization import map_to_json
from catanzero.board_exclusions import board_hash_variants, normalization_source_board_hashes
from catanzero.online_env import OnlineEnvironment
from catanzero.regression_data import canonical_hash as compact_hash
from catanzero.simulation import create_game, SimulationConfig
from catanzero.tournament import canonical_hash as spaced_hash


@pytest.mark.parametrize('hasher', [compact_hash, spaced_hash])
def test_worker_rejects_previously_seen_board_with_either_hash_convention(tmp_path, hasher):
    document = map_to_json(create_game(SimulationConfig(), 123).state.board.map)
    env = OnlineEnvironment({'base_seed': 123, 'excluded_board_hashes': [hasher(document)]})
    with pytest.raises(ValueError, match='overlaps'):
        env.reset(0, tmp_path/'game')


def test_normalization_boards_follow_source_provenance_and_include_holdouts(tmp_path):
    source = tmp_path/'raw'
    source.mkdir()
    maps = [{'map_ref':0,'map':{'tiles':[1,2]}}, {'map_ref':1,'map':{'tiles':[3,4]}}]
    (source/'maps.jsonl').write_text('\n'.join(json.dumps(m) for m in maps))
    export = tmp_path/'export'
    export.mkdir()
    (export/'data_report.json').write_text(json.dumps({'sources':[{'source':str(source)}]}))
    assert set(normalization_source_board_hashes(export)) == set.union(*(board_hash_variants(m['map']) for m in maps))


def test_missing_normalization_provenance_is_rejected(tmp_path):
    (tmp_path/'data_report.json').write_text(json.dumps({'sources':[]}))
    with pytest.raises(ValueError, match='provenance'):
        normalization_source_board_hashes(tmp_path)
