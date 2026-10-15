"""Engine replay, causal reward reconstruction and raw-feature audit export."""
import argparse
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"src"))
from catanatron.serialization import action_to_json,map_to_json,state_to_json
from catanzero.board_exclusions import board_hash_variants
from catanzero.dataset import observation,ordered_actions
from catanzero.features import FeatureEncoder
from catanzero.regression_search import public_f_choice
from catanzero.search_labels import live_belief
from catanzero.simulation import create_game,SimulationConfig
from catanzero.static_policies import forbid_forward_simulation
from catanzero.tree_duel_reward import transition_reward,potential
from catanzero.tournament import atomic_json,canonical_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path);args=parser.parse_args()
    root=args.root.resolve();config=json.loads((root/"config.json").read_text())
    normalizer=json.loads((root/"normalizer.json").read_text())
    is_policy='RL_rounds' in config
    is_dagger='candidate_protocol' in config
    is_trade='trade-aware' in config['algorithm']
    explicit=bool(config.get('phase_plan'))
    if explicit:
        phases=config['phase_plan']
    elif is_trade:
        phases=[f'validation_candidate_{i}' for i in (400,401,402)]+['final_test']
    elif is_dagger:
        phases=[f'validation_candidate_{i}' for i in range(300,305)]+['final_test']
    elif is_policy:
        phases=['validation_candidate_100','validation_candidate_101','validation_candidate_102']
        for i in range(1,config['RL_rounds']+1):
            phases.extend([f'RL_round_{i}',f'validation_candidate_{200+10*i}',f'validation_candidate_{201+10*i}'])
        phases.append('final_test')
    else:
        phases=["baseline_validation"]+[p for i in range(1,config['rounds']+1) for p in (f"training_round_{i}",f"validation_round_{i}")]+["baseline_test","tree_test"]
    all_results={p:[json.loads(s) for s in (root/p/"results.jsonl").read_text().splitlines()] for p in phases}
    train_hashes={r['initial_map_hash'] for p,rows in all_results.items() if p.startswith(('training','RL_round')) for r in rows}
    val_phase=config['validation_phase'] if explicit else 'validation_candidate_400' if is_trade else 'validation_candidate_300' if is_dagger else 'validation_candidate_100' if is_policy else 'baseline_validation'
    val_hashes={r['initial_map_hash'] for r in all_results[val_phase]}
    test_phase=config['test_phase'] if explicit else 'final_test' if is_policy or is_dagger else 'baseline_test'
    test_hashes={r['initial_map_hash'] for r in all_results[test_phase]}
    assert not(train_hashes&val_hashes or train_hashes&test_hashes or val_hashes&test_hashes)
    assert len(train_hashes)==(config['fresh_training_games'] if explicit else 0 if is_dagger else config['RL_rounds']*config['games_per_RL_round'] if is_policy else config['rounds']*config['games_per_round'])
    if is_dagger or (explicit and (root/'teacher_report.json').exists()):
        source_hashes=set(json.loads((root/'teacher_report.json').read_text())['source_board_hashes'])
        assert not(source_hashes&val_hashes or source_hashes&test_hashes)
    counts={"games":0,"actions":0,"reward_transitions":0,"prediction_candidates":0,
            "privacy_checks":0,"native_F_choices_recomputed":0,"max_reward_error":0.}
    start=perf_counter()
    with gzip.open(root/"audit_predictions.jsonl.gz","wt",encoding="utf-8",compresslevel=1) as predictions, \
         gzip.open(root/"audit_transitions.jsonl.gz","wt",encoding="utf-8",compresslevel=1) as transitions:
        for phase in phases:
            directory=root/phase;pc=json.loads((directory/"config.json").read_text());training=pc['training']
            metadata=[json.loads(s) for s in (directory/"transitions.jsonl").read_text().splitlines()] if training else []
            positions={(m['game_index'],m['action_index']):i for i,m in enumerate(metadata)}
            used=set()
            rows=all_results[phase]
            for result in sorted(rows,key=lambda r:r['game_index']):
                index=result['game_index'];sample=index in (0,len(rows)-1)
                game=create_game(SimulationConfig(players=2),result['seed'])
                document=map_to_json(game.state.board.map)
                assert not(board_hash_variants(document)&set(config['excluded_board_hashes']))
                assert canonical_hash(document)==result['initial_map_hash']
                assert canonical_hash(game.state.random.getstate())==result['initial_stream_hash']
                learner=game.state.colors[index%2];assert learner.value==result['learner_color']
                encoder=FeatureEncoder(document,normalizer['feature_version'],allow_fewer_players=True)
                last=None;privacy_kinds=set();total_reward=0.;initial_phi=None

                def audit_transition(previous,following,raw_next,terminal):
                    nonlocal total_reward,initial_phi
                    pos=positions[index,previous['response']['action_index']]
                    m=metadata[pos];used.add(pos)
                    r,d=transition_reward(previous['response'],following,terminal=terminal,won=result['learner_won'] if terminal else None)
                    assert m['actor']==learner.value and m['phase']==phase and m['terminal']==terminal
                    assert m['next_action_index']==following['action_index'] and m['selected_index']==previous['selected']
                    assert m['collection_model_version']==pc['model_version']
                    error=max(abs(m['reward']-r),abs(m['discount']-d));assert error<1e-12
                    counts['max_reward_error']=max(counts['max_reward_error'],error)
                    expected_phi=potential(previous['response']['vp'],previous['response']['leader_public_vp'])
                    assert abs(expected_phi-m['phi'])<1e-12
                    assert abs(m['next_phi']-(0 if terminal else potential(following['vp'],following['leader_public_vp'])))<1e-12
                    if initial_phi is None:initial_phi=expected_phi
                    total_reward+=r;counts['reward_transitions']+=1
                    transitions.write(json.dumps({'phase':phase,'position':pos,'x':previous['raw'][previous['selected']] if previous['raw'] is not None else None,
                                                   'next_x':raw_next,'reward':r,'discount':d})+'\n')

                with gzip.open(directory/f"game_{index:04d}/decisions.jsonl.gz","rt",encoding='utf-8') as stream:
                    for line in stream:
                        row=json.loads(line);actor=game.state.current_color();actions=ordered_actions(game)
                        expected={'map_ref':0,'observation':observation(game,actor),'legal_actions':[action_to_json(a) for a in actions]}
                        selected=actions.index(next(a for a in actions if action_to_json(a)==row['inputs']['action']))
                        assert row['action_index']==len(game.state.action_records)
                        assert row['inputs']=={**expected,'action':action_to_json(actions[selected])}
                        assert row['policy']==(pc['learner_policy'] if actor==learner else 'F')
                        if actor==learner:
                            assert row['forward_simulation_guard'] is True and row['model_version']==pc['model_version']
                            if not training:assert not row['explored']
                            if training or sample:
                                raw=None
                                if sample:
                                    with forbid_forward_simulation():raw=[encoder.encode({**expected,'action':a}) for a in expected['legal_actions']]
                                own=next(p for p in expected['observation']['players'] if p['color']==learner.value)
                                enemy=next(p for p in expected['observation']['players'] if p['color']!=learner.value)
                                response={'action_index':row['action_index'],'learner_color':learner.value,
                                          'vp':own['actual_vp'],'leader_public_vp':enemy['VICTORY_POINTS']}
                                if training and last is not None:audit_transition(last,response,raw,False)
                                last={'response':response,'raw':raw,'selected':selected}
                            if sample:
                                predictions.write(json.dumps({'phase':phase,'game_index':index,'action_index':row['action_index'],
                                                              'version':row['model_version'],'raw':raw,'scores':row['raw_scores'],
                                                              'selected':selected,'explored':row['explored']})+'\n')
                                counts['prediction_candidates']+=len(raw)
                                kind=actions[selected].action_type.value
                                if index==0 and kind not in privacy_kinds:
                                    changed=json.loads(json.dumps(expected))
                                    changed['observation'].update(seed=999,winner='fake',future_deck=['VICTORY_POINT'])
                                    for p in changed['observation']['players']:
                                        if p['color']!=learner.value:p.update(resources={'ORE':999},actual_vp=99,development_cards={'VICTORY_POINT':9})
                                    with forbid_forward_simulation():assert raw==[encoder.encode({**changed,'action':a}) for a in changed['legal_actions']]
                                    privacy_kinds.add(kind);counts['privacy_checks']+=1
                        else:
                            assert row['forward_simulation_guard'] is False
                            if sample and len(actions)>1:
                                before_rng=game.state.random.getstate();before=state_to_json(game)
                                seed=result['seed']*1000003+row['action_index']
                                assert public_f_choice(game,actions,seed)==actions[selected]
                                assert before_rng==game.state.random.getstate() and before==state_to_json(game)
                                counts['native_F_choices_recomputed']+=1
                        game.execute(actions[selected]);counts['actions']+=1
                if training and last is not None and result['termination']=='victory':
                    audit_transition(last,{'action_index':result['actions'],'learner_color':learner.value,'vp':result['vp']},[],True)
                    assert abs(total_reward-((1. if result['learner_won'] else -1.)-initial_phi))<1e-10
                final=state_to_json(game);saved=json.loads((directory/f"game_{index:04d}/game.json").read_text())
                final['game'].pop('id');saved['game'].pop('id');assert final==saved
                assert (game.winning_color().value if game.winning_color() else None)==result['winner']
                assert len(game.state.action_records)==result['actions']
                counts['games']+=1
            assert len(used)==len(metadata)
            print(f"AUDIT {phase}: {len(rows)} games, elapsed={perf_counter()-start:.1f}s",flush=True)
    report={'status':'passed',**counts,'unique_training_boards':len(train_hashes),'validation_boards':len(val_hashes),
            'test_boards':len(test_hashes),'train_validation_test_disjoint':True,
            'learner_forward_simulation':False,'reward_full_episode_telescoping_verified':True,
            'scope':'all games engine replay/whitelist/legals/final state and all training rewards; first and last game in each phase feature/F decision checks',
            'elapsed_seconds':perf_counter()-start}
    atomic_json(root/'engine_audit.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
