"""Preview undeployed candidate features on real training decisions only."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
from catanatron.serialization import map_to_json
from catanzero.decision_interactions import CandidateDecisionFeatures
from catanzero.simulation import create_game, SimulationConfig
from catanzero.static_policies import forbid_forward_simulation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root
    results=sorted((json.loads(s) for s in (root/'live_training/results.jsonl').read_text().splitlines()),key=lambda r:r['game_index'])
    selected=results[:8]+results[-8:]
    count=0;names=None;kinds=set()
    with gzip.open(root/'candidate_interactions_preview.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as out:
        for result in selected:
            game=create_game(SimulationConfig(),result['seed'])
            prototype=CandidateDecisionFeatures(map_to_json(game.state.board.map))
            seen=set()
            directory=root/'live_training'/f"game_{result['game_index']:04d}"
            with gzip.open(directory/'decisions.jsonl.gz','rt',encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line)
                    if row['policy']!=result['learner_policy']:
                        continue
                    inputs=row['inputs'];kind=inputs['action'][1]
                    if kind in seen:
                        continue
                    before=game.state.random.getstate()
                    changed=deepcopy(inputs)
                    changed.update(target=1.,winner='future',seed=999)
                    changed['observation'].update(winner='future',random_state='future',development_listdeck=['VICTORY_POINT']*100)
                    for p in changed['observation']['players']:
                        if p['color']!=changed['observation']['actor']:
                            p.update(actual_vp=999,resources={'ORE':999},development_cards={'KNIGHT':999})
                    with forbid_forward_simulation():
                        features=prototype.features(inputs)
                        assert features==prototype.features(changed)
                    assert game.state.random.getstate()==before
                    if names is None:names=list(features)
                    assert list(features)==names
                    out.write(json.dumps({'metadata':{'game_index':result['game_index'],'action_index':row['action_index']},'features':features})+'\n')
                    count+=1;seen.add(kind);kinds.add(kind)
    report={'feature_count':len(names),'feature_names':names,'sampled_training_games':len(selected),
        'preview_rows':count,'action_types':sorted(kinds),'privacy_and_no_forward_simulation_checks':count,
        'trained':False,'deployed':False,'normalization_fitted':False,'source_split':'training only',
        'design':'16 contextual products plus 8 existing public production-moment candidates',
        'limitations':['No measured win-rate benefit','Must fit any future normalization on training data only',
            'Potential production ignores resource-bank shortages','Estimated award points retain existing encoder limitations']}
    (root/'candidate_interactions_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
