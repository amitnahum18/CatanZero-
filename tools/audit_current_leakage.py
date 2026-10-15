"""Independent leakage audit of feature SARSA v4 and its fresh paired evaluation.

Engine mode uses the pinned simulator; updates mode uses the ML environment.
Outputs are separate audit artifacts and never modify models or original logs.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / 'runs/online-sarsa-features-100-v4'
COMPARE = REPO / 'runs/frozen-linear-v3-vs-v4-200-v1'
OUT = REPO / 'runs/leakage-audit-v4-v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def write(name, report):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


def digest(value, compact=False):
    kwargs = {'separators': (',', ':')} if compact else {}
    return hashlib.sha256(json.dumps(value, sort_keys=True, **kwargs).encode()).hexdigest()


def engine():
    from catanatron.serialization import action_record_from_json, action_to_json, map_to_json, state_to_json
    from catanatron.models.enums import ActionType
    from catanzero.dataset import observation, ordered_actions
    from catanzero.features import FeatureEncoder, VERSION
    from catanzero.online_linear import LinearActionValues
    from catanzero.production_statistics import ProductionStatistics
    from catanzero.regression_data import privacy_invariance_check, fit_normalizer
    from catanzero.simulation import create_game, SimulationConfig
    from catanzero.static_policies import forbid_forward_simulation
    from catanzero.temporal_features import TemporalFeatures, METRICS, PAST_ACTIONS, snapshot

    OUT.mkdir(exist_ok=True)
    norm = read(TRAIN/'normalizer.json')
    source = Path(read(TRAIN/'config.json')['resume_from'])
    phases = [(TRAIN, 'initial_evaluation', source, False),
              (TRAIN, 'live_training', TRAIN, True),
              (TRAIN, 'final_evaluation', TRAIN, False),
              (COMPARE, 'baseline_evaluation', COMPARE/'baseline', False),
              (COMPARE, 'expanded_evaluation', COMPARE/'expanded', False)]
    excluded = set(read(COMPARE/'config.json')['excluded_board_hashes'])
    ancestor = TRAIN
    earlier = set()
    while ancestor:
        for phase in ('initial_evaluation', 'live_training', 'final_evaluation'):
            earlier.update(r['initial_map_hash'] for r in lines(ancestor/phase/'results.jsonl'))
        parent = read(ancestor/'config.json').get('resume_from')
        ancestor = Path(parent) if parent else None
    earlier.update(read(REPO/'runs/fqi-pilot-500-v1/experience_audit.json')['collection']['board_hashes'])
    # Both serialization conventions are checked, rather than trusting mixed stored hashes.
    dataset = REPO/'runs/dataset-pilot-100-v2'
    maps = {m['map_ref']: m['map'] for m in lines(dataset/'maps.jsonl')}
    normalization_boards = {digest(m, compact=c) for m in maps.values() for c in (False, True)}
    result_counts, actions, predictions, privacy, stat_checks = {}, 0, 0, 0, 0
    board_sets = {}
    temporal_count = 0
    preview = iter(lines(REPO/'runs/feature-history-diagnostics-v1/history_preview.jsonl.gz'))
    memory = TemporalFeatures()
    private_memory = TemporalFeatures()
    independent_history = []
    with gzip.open(OUT/'training_cases.jsonl.gz', 'wt', encoding='utf-8') as cases:
        for root, phase, model_root, training in phases:
            results = sorted(lines(root/phase/'results.jsonl'), key=lambda r:r['game_index'])
            phase_norm = read(model_root/'normalizer.json')
            frozen = LinearActionValues.load(model_root/'linear_model.json', model_root/'normalizer.json')
            snapshots = {}
            boards = set()
            sample_ids = {0,1,2,3,len(results)-4,len(results)-3,len(results)-2,len(results)-1}
            for number, result in enumerate(results, 1):
                directory = root/phase/f"game_{result['game_index']:04d}"
                document = read(directory/'game.json')
                game = create_game(SimulationConfig(), result['seed'])
                map_document = map_to_json(game.state.board.map)
                spaced, compact = digest(map_document), digest(map_document, True)
                assert spaced == result['initial_map_hash'] and map_document == document['map']
                boards.add(spaced)
                assert spaced not in normalization_boards and compact not in normalization_boards
                if root == COMPARE:
                    assert spaced not in earlier and spaced not in excluded and compact not in excluded
                encoder = FeatureEncoder(map_document, version=phase_norm.get('feature_version', VERSION))
                stats = ProductionStatistics(map_document)
                if training:
                    independent_history = []
                checked = 0
                strata = set()
                for index, (row, payload) in enumerate(zip(lines(directory/'decisions.jsonl.gz'), document['action_records'], strict=True)):
                    record = action_record_from_json(payload)
                    selected = record.action
                    if selected.action_type in (ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD):
                        selected = selected._replace(value=None)
                    inputs = row['inputs']
                    assert inputs['observation'] == observation(game, game.state.current_color())
                    assert inputs['legal_actions'] == [action_to_json(a) for a in ordered_actions(game)]
                    assert inputs['action'] == action_to_json(selected) and row['action_index'] == index
                    assert row['forward_simulation_guard'] and result['forbidden_engine_calls'] == 0
                    obs = inputs['observation']
                    assert not set(obs).intersection(('winner','seed','random_state','development_listdeck','targets'))
                    for p in obs['players']:
                        if p['color'] != obs['actor']:
                            assert not set(p).intersection(('resources','development_cards','actual_vp','playable_at_turn_start'))
                    own = selected.color.value == result['learner_color']
                    assert (row['policy'] == result['learner_policy']) == own
                    if own:
                        sampled = result['game_index'] in sample_ids
                        state_before = game.state.random.getstate()
                        with forbid_forward_simulation():
                            vector = encoder.encode(inputs) if training or sampled else None
                            if sampled:
                                if training:
                                    version = row['model_version']
                                    if version not in snapshots:
                                        snapshots[version] = LinearActionValues(read(TRAIN/'snapshots'/f'version_{version:06d}.json'), phase_norm)
                                    scorer = snapshots[version]
                                else:
                                    scorer = frozen
                                    assert row['model_version'] == (0 if phase in ('initial_evaluation','baseline_evaluation','expanded_evaluation') else scorer.model['model_version'])
                                scores = [scorer.predict(encoder.encode({**inputs,'action':a})) for a in inputs['legal_actions']]
                                assert len(scores) == len(row['raw_scores'])
                                assert max(abs(a-b) for a,b in zip(scores,row['raw_scores'])) < 1e-10
                                idx = inputs['legal_actions'].index(inputs['action'])
                                if not row['explored']:
                                    assert abs(scores[idx]-max(scores)) < 1e-10
                                predictions += len(scores)
                                kind = selected.action_type.value
                                if kind not in strata:
                                    assert privacy_invariance_check(encoder, row)
                                    changed = deepcopy(inputs)
                                    changed.update(seed=999,winner='FUTURE',future_roll=[6,6],targets={'win':1})
                                    changed['observation'].update(seed=999,winner='FUTURE',random_state='future',development_listdeck=['VICTORY_POINT']*100)
                                    for p in changed['observation']['players']:
                                        if p['color'] != obs['actor']:
                                            p.update(resources={'ORE':999},actual_vp=999,development_cards={'KNIGHT':999})
                                    assert stats.features(inputs) == stats.features(changed)
                                    privacy += 1
                                    stat_checks += 1
                                    strata.add(kind)
                        assert game.state.random.getstate() == state_before
                        if not training:
                            assert not row['explored']
                        else:
                            own_state = next(p for p in obs['players'] if p['color']==obs['actor'])
                            normalized = [max(-4,min(4,(v-o)/s)) for v,o,s in zip(vector,norm['offset'],norm['scale'],strict=True)]
                            cases.write(json.dumps({'game_index':result['game_index'],'actor':obs['actor'],'action_index':index,
                                'x':normalized,'selected_index':inputs['legal_actions'].index(inputs['action']),
                                'vp':own_state['actual_vp'],'model_version':row['model_version']})+'\n')
                            history = memory.features(obs, str(directory.resolve()))
                            saved = next(preview)
                            assert saved['metadata']=={'game_index':result['game_index'],'action_index':index}
                            assert saved['features']==history
                            current = snapshot(obs)
                            # Independent indexed prefix reference: current action never included.
                            for lag in (1,3,5):
                                available = len(independent_history)>=lag
                                assert history[f'history_{lag}_available']==float(available)
                                for metric in METRICS:
                                    value = current[metric]-independent_history[-lag][0][metric] if available else 0.
                                    assert history[f'history_{lag}_delta_{metric}']==value
                            for kind in PAST_ACTIONS:
                                expected = sum(k==kind for _,k in independent_history)/len(independent_history) if independent_history else 0.
                                assert history[f'history_5_share_{kind}']==expected
                            for metric in ('resources','city_deficit'):
                                values=[m[metric] for m,_ in independent_history]
                                mean=sum(values)/len(values) if values else 0.
                                std=(sum((v-mean)**2 for v in values)/len(values))**.5 if values else 0.
                                assert history[f'history_5_mean_{metric}']==mean
                                assert history[f'history_5_std_{metric}']==std
                            changed_obs = deepcopy(obs)
                            changed_obs.update(winner='FUTURE',seed=999,future_roll=[6,6])
                            for p in changed_obs['players']:
                                if p['color']!=obs['actor']:
                                    p.update(resources={'ORE':999},actual_vp=999,development_cards={'KNIGHT':999})
                            assert private_memory.features(changed_obs,str(directory.resolve()))==history
                            memory.commit(obs,str(directory.resolve()),inputs['action'][1])
                            private_memory.commit(changed_obs,str(directory.resolve()),inputs['action'][1])
                            independent_history.append((current,inputs['action'][1]))
                            independent_history=independent_history[-5:]
                            temporal_count += 1
                    game.execute(selected, action_record=record)
                    actions += 1
                    checked += 1
                assert checked == result['actions'] == len(document['action_records'])
                final = state_to_json(game)
                for key in ('board','player_state','num_turns','action_records','resource_freqdeck','development_listdeck'):
                    assert final[key] == document[key]
                winner = game.winning_color()
                assert result['winner']==(winner.value if winner else None)
                assert result['learner_won']==(winner.value==result['learner_color'] if winner else None)
                if number % 50 == 0:
                    print(f'REPLAY {phase}: {number}/{len(results)}',flush=True)
            result_counts[phase]=len(results)
            board_sets[phase]=boards
    assert next(preview,None) is None
    assert not board_sets['live_training'] & board_sets['initial_evaluation']
    assert board_sets['initial_evaluation']==board_sets['final_evaluation']
    assert board_sets['baseline_evaluation']==board_sets['expanded_evaluation']
    assert not board_sets['baseline_evaluation'] & earlier
    # Refit only the raw training rows; validation/test do not enter statistics.
    encoders={k:FeatureEncoder(m,version=norm['feature_version']) for k,m in maps.items()}
    schema=read(TRAIN/'feature_schema.json')
    source_results=list(lines(dataset/'results.jsonl'))
    split_boards={s:set() for s in ('train','validation','test')}
    def raw_training():
        for result in source_results:
            for row in lines(dataset/result['moves_file']):
                if row['train_eligible']:
                    inputs=row['inputs']
                    split_boards[result['split']].add(digest(maps[inputs['map_ref']]))
                    if result['split']=='train':
                        yield encoders[inputs['map_ref']].encode(inputs)
    fitted=fit_normalizer(raw_training(),schema)
    exported=read(REPO/'runs/regression-build-capacity-v3/normalizer.json')
    assert fitted==exported
    for a,b in (('train','validation'),('train','test'),('validation','test')):
        assert not split_boards[a]&split_boards[b]
    for key in ('offset','scale','train_mean','train_std'):
        assert norm[key][164:]==fitted[key][164:]
    write('engine.json',{'passed':True,'games_replayed':result_counts,'actions_predecision_and_final_state_verified':actions,
        'candidate_scores_recomputed_in_8_games_per_phase':predictions,'privacy_checks':privacy,
        'experimental_statistic_privacy_checks':stat_checks,'causal_history_rows_checked':temporal_count,
        'training_only_normalizer_rows_recomputed':fitted['fit_rows'],'all_test_boards_disjoint_from_model_ancestry_and_normalization':True,
        'normalization_boards_missing_from_old_exclusion_list':len({digest(m) for m in maps.values()}-excluded),
        'scope':'All 564 games replayed; candidate scores independently recomputed in 40 games; history is an undeployed prototype.'})


def updates():
    import joblib
    import numpy as np
    config=read(TRAIN/'config.json')
    source=Path(config['resume_from'])
    estimator=joblib.load(TRAIN/'initial_model.joblib')
    old=joblib.load(source/'model.joblib')
    assert hashlib.sha256((source/'model.joblib').read_bytes()).hexdigest()==config['source_model_sha256']
    assert np.array_equal(estimator.coef_[:164],old.coef_) and not estimator.coef_[164:].any()
    assert np.array_equal(estimator.intercept_,old.intercept_) and estimator.t_==old.t_
    assert estimator.get_params()==old.get_params()
    norm=read(TRAIN/'normalizer.json')
    old_norm=read(source/'normalizer.json')
    for key in ('offset','scale','feature_names','train_mean','train_std'):
        if key in old_norm:
            assert norm[key][:164]==old_norm[key]
    cases={(r['game_index'],r['action_index']):r for r in lines(OUT/'training_cases.jsonl.gz')}
    results={r['game_index']:r for r in lines(TRAIN/'live_training/results.jsonl')}
    ordered={g:sorted([i for gg,i in cases if gg==g]) for g in results}
    next_indices={(g,i):(ids[n+1] if n+1<len(ids) else results[g]['actions']) for g,ids in ordered.items() for n,i in enumerate(ids)}
    seen=set()
    transitions=changes=0
    for version,update in enumerate(lines(TRAIN/'updates.jsonl.gz'),1):
        assert update['version']==version
        samples=update['samples']
        x=np.asarray([s['x'] for s in samples]);following=np.asarray([s['next_x'] for s in samples])
        assert x.shape[1]==264 and following.shape==x.shape and np.max(np.abs(x))<=4
        for s in samples:
            key=(s['game_index'],s['action_index'])
            assert s['phase']=='live_training' and key not in seen
            seen.add(key)
            actual=cases[key]
            assert s['actor']==actual['actor']==results[key[0]]['learner_color']
            np.testing.assert_allclose(s['x'],actual['x'],atol=1e-12,rtol=0)
            assert s['selected_index']==actual['selected_index'] and s['action_model_version']==actual['model_version']
            ni=s['next_action_index'];assert ni==next_indices[key]
            terminal=ni==results[key[0]]['actions']
            assert terminal==s['terminal']
            gap=ni-key[1]
            discount=0 if terminal else .999**gap
            if terminal:
                assert results[key[0]]['termination']=='victory'
                np.testing.assert_array_equal(s['next_x'],np.zeros(264))
                reward=.999**(gap-1)*float(results[key[0]]['learner_won'])-.05*actual['vp']
            else:
                nxt=cases[(key[0],ni)]
                assert nxt['actor']==actual['actor']
                np.testing.assert_allclose(s['next_x'],nxt['x'],atol=1e-12,rtol=0)
                assert s['next_selected_index']==nxt['selected_index'] and s['next_action_model_version']==nxt['model_version']
                reward=discount*.05*nxt['vp']-.05*actual['vp']
            assert abs(s['discount']-discount)<1e-12 and abs(s['reward']-reward)<1e-12
        targets=np.asarray([s['reward'] for s in samples])+np.asarray([s['discount'] for s in samples])*estimator.predict(following)
        np.testing.assert_allclose(targets,[s['target'] for s in samples],atol=1e-12,rtol=0)
        previous=estimator.coef_.copy()
        estimator.partial_fit(x,targets,sample_weight=1/(1+np.sum(x*x,axis=1)))
        saved=read(TRAIN/'snapshots'/f'version_{version:06d}.json')
        np.testing.assert_allclose(estimator.coef_,saved['weights'],atol=1e-12,rtol=0)
        assert abs(estimator.intercept_[0]-saved['intercept'])<1e-12
        changes+=bool(np.any(estimator.coef_!=previous))
        transitions+=len(samples)
    final=joblib.load(TRAIN/'model.joblib')
    np.testing.assert_array_equal(estimator.coef_,final.coef_)
    np.testing.assert_array_equal(estimator.intercept_,final.intercept_)
    assert estimator.t_==final.t_ and seen==set(cases)
    report=read(TRAIN/'report.json')
    assert version==report['central_updates'] and transitions==report['training_transitions']
    for label in ('baseline','expanded'):
        assert hashlib.sha256((COMPARE/label/'model.joblib').read_bytes()).hexdigest()==read(COMPARE/'config.json')['source_model_sha256'][label]
        for file in ('model.joblib','linear_model.json','normalizer.json'):
            assert (COMPARE/label/file).read_bytes()==(Path(read(COMPARE/'config.json')['sources'][label])/file).read_bytes()
    frozen_decisions=0
    for root,phase,frozen_version in ((TRAIN,'initial_evaluation',0),(TRAIN,'final_evaluation',report['central_updates']),
                                (COMPARE,'baseline_evaluation',0),(COMPARE,'expanded_evaluation',0)):
        for result in lines(root/phase/'results.jsonl'):
            for row in lines(root/phase/f"game_{result['game_index']:04d}"/'decisions.jsonl.gz'):
                if row['policy']==result['learner_policy']:
                    assert row['model_version']==frozen_version and not row['explored']
                    frozen_decisions+=1
    write('updates.json',{'passed':True,'updates_recomputed':version,'actual_predecision_training_inputs_checked':transitions,
        'all_rewards_and_actual_next_actions_recomputed':True,'evaluation_samples_in_updates':0,
        'final_weights_and_optimizer_exact_match':True,'new_weights_initially_zero':True,'nonzero_updates':changes,
        'old_train_only_normalization_preserved':True,'frozen_comparison_packages_unchanged':True,
        'all_frozen_evaluation_decision_versions_checked':frozen_decisions})


def ancestry_normalization():
    import pickle
    import numpy as np
    root=REPO/'runs/fqi-pilot-500-v1'
    selected=[]
    ids=[]
    for index in range(400):
        with gzip.open(root/'collection'/f'game_{index:04d}'/'experience.pkl.gz','rb') as stream:
            episode=pickle.load(stream)
        assert episode['result']['game_index']==index
        for state in episode['states']:
            assert state['actor']==episode['result']['learner_color']
            start=164*state['selected_index']
            selected.append(np.asarray(state['features'][start:start+164],dtype=np.float32))
            ids.append((index,state['action_index']))
    array=np.asarray(selected,dtype=np.float32)
    normalizer=read(root/'normalizer.json')
    mean=array.mean(axis=0,dtype=np.float64)
    std=array.std(axis=0,dtype=np.float64)
    std[std<1e-12]=1.
    np.testing.assert_array_equal(mean,normalizer['offset'])
    np.testing.assert_array_equal(std,normalizer['scale'])
    assert len(array)==normalizer['fit_rows']==41924
    transitions=np.load(root/'transitions.npz')
    assert np.array_equal(transitions['train_mask'],transitions['metadata'][:,0]<400)
    assert ids==[(int(g),int(a)) for g,a,_ in transitions['metadata'][transitions['train_mask']]]
    origin=read(TRAIN/'normalizer.json')
    np.testing.assert_array_equal(origin['offset'][:164],mean)
    np.testing.assert_array_equal(origin['scale'][:164],std)
    write('ancestry_normalization.json',{'passed':True,'original_normalizer_rows_recomputed':len(array),
        'original_training_games':400,'validation_games_excluded':100,'statistics_exact_match':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('engine','updates','ancestry-normalization'))
    args=parser.parse_args()
    {'engine':engine,'updates':updates,'ancestry-normalization':ancestry_normalization}[args.mode]()
