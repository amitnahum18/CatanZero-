"""Training-only, public-predecision candidate matrices for policy explanations."""
import argparse
import gzip
import json
from pathlib import Path
from catanatron.serialization import map_to_json
from catanzero.features import FeatureEncoder, VERSION
from catanzero.simulation import create_game, SimulationConfig
from catanzero.regression_data import privacy_invariance_check
from catanzero.static_policies import forbid_forward_simulation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    root=args.root
    norm=json.loads((root/'normalizer.json').read_text())
    names=norm['feature_names']
    results=sorted((json.loads(s) for s in (root/'live_training/results.jsonl').read_text().splitlines()),key=lambda r:r['game_index'])
    # Four consecutive seats per evenly spaced block; training data only.
    block=max(4,(len(results)//25)//4*4)
    selected_ids={i+j for i in range(0,len(results),block) for j in range(4) if i+j<len(results)}
    cases=candidates=games=0
    with gzip.open(root/'decision_cases.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as out:
        for result in results:
            if result['game_index'] not in selected_ids:
                continue
            directory=root/'live_training'/f"game_{result['game_index']:04d}"
            game=create_game(SimulationConfig(),result['seed'])
            encoder=FeatureEncoder(map_to_json(game.state.board.map),version=norm.get('feature_version',VERSION))
            rows=[]
            with gzip.open(directory/'decisions.jsonl.gz','rt',encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line)
                    if row['policy']==result['learner_policy'] and len(row['inputs']['legal_actions'])>1:
                        rows.append(row)
            if not rows:
                continue
            indices=sorted({round(i*(len(rows)-1)/max(min(24,len(rows))-1,1)) for i in range(min(24,len(rows)))})
            for index in indices:
                row=rows[index]
                inputs=row['inputs']
                assert inputs['observation']['actor']==result['learner_color']
                before=game.state.random.getstate()
                with forbid_forward_simulation():
                    vectors=[encoder.encode({**inputs,'action':a}) for a in inputs['legal_actions']]
                    assert privacy_invariance_check(encoder,row)
                assert game.state.random.getstate()==before
                assert all(len(v)==len(names) for v in vectors)
                out.write(json.dumps({'game_index':result['game_index'],'action_index':row['action_index'],
                    'selected_action_kind':inputs['action'][1],'initial_phase':inputs['observation']['initial_phase'],
                    'candidate_action_kinds':[a[1] for a in inputs['legal_actions']],
                    'raw_features':vectors})+'\n')
                cases+=1
                candidates+=len(vectors)
            games+=1
            if games%20==0:
                print(f'EXPLANATIONS: {games} training games, {cases} choice states, {candidates} candidates',flush=True)
    report={'source':'training only; evenly spaced own choice decisions','sampled_games':games,'choice_states':cases,
        'candidate_vectors':candidates,'features':len(names),'balanced_seat_sampling':True,
        'no_labels_or_test_games_used':True,'privacy_and_no_forward_simulation_checks':cases}
    (root/'decision_cases_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
