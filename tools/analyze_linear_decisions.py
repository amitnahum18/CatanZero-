"""Explain linear action rankings using candidate differences, not weight size alone."""
import argparse
import gzip
import json
from pathlib import Path
import joblib
import numpy as np


def analyze(root):
    norm=json.loads((root/'normalizer.json').read_text())
    schema=json.loads((root/'feature_schema.json').read_text())
    config=json.loads((root/'config.json').read_text())
    source=Path(config['resume_from'])
    assert json.loads((source/'normalizer.json').read_text())==norm
    weights=np.asarray(json.loads((root/'linear_model.json').read_text())['weights'])
    old=np.asarray(json.loads((source/'linear_model.json').read_text())['weights'])
    estimator=joblib.load(root/'model.joblib')
    names=norm['feature_names']
    centered=np.zeros(len(names));margin=np.zeros(len(names));varying=np.zeros(len(names));switch=np.zeros(len(names))
    group_counts={k:0 for k in schema['groups']}
    group_effect={k:0. for k in schema['groups']}
    phase_counts={'initial':0,'normal':0}
    phase_contributions={k:np.zeros(len(names)) for k in phase_counts}
    phase_switch={k:np.zeros(len(names)) for k in phase_counts}
    examples=[]
    cases=candidate_count=old_new_changed=0
    with gzip.open(root/'decision_cases.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            z=np.clip((np.asarray(row['raw_features'])-norm['offset'])/norm['scale'],-4,4)
            assert z.shape[1]==len(names) and len(z)>1 and np.isfinite(z).all()
            scores=estimator.predict(z)
            np.testing.assert_allclose(scores,z@weights+estimator.intercept_[0],atol=1e-12,rtol=0)
            chosen=int(np.argmax(scores))
            second=int(np.argsort(scores,kind='stable')[-2])
            contributions=(z-z.mean(axis=0))*weights
            observed=np.abs(contributions).mean(axis=0)
            centered+=observed
            varying+=np.ptp(z,axis=0)>1e-12
            delta=(z[chosen]-z[second])*weights
            assert abs(delta.sum()-(scores[chosen]-scores[second]))<1e-12
            margin+=np.abs(delta)
            removed=scores[:,None]-z*weights
            changed=removed.max(axis=0)-removed[chosen]>1e-10
            switch+=changed
            phase='initial' if row['initial_phase'] else 'normal'
            phase_counts[phase]+=1
            phase_contributions[phase]+=observed
            phase_switch[phase]+=changed
            old_scores=z@old
            old_new_changed+=bool(old_scores.max()-old_scores[chosen]>1e-10)
            for group,document in schema['groups'].items():
                indices=[names.index(name) for name in document['names']]
                without=scores-z[:,indices]@weights[indices]
                loss=float(without.max()-without[chosen])
                group_counts[group]+=loss>1e-10
                group_effect[group]+=loss
            if phase=='normal' and len(examples)<12:
                indices=np.argsort(-np.abs(delta))[:5]
                examples.append({'game_index':row['game_index'],'action_index':row['action_index'],
                    'greedy_action':row['candidate_action_kinds'][chosen],
                    'runner_up_action':row['candidate_action_kinds'][second],
                    'margin':float(scores[chosen]-scores[second]),
                    'largest_margin_terms':[{'feature':names[i],'contribution':float(delta[i])} for i in indices]})
            cases+=1
            candidate_count+=len(z)
    def ranking(contribution,counts,denominator):
        return [{'feature':names[i],'coefficient':float(weights[i]),'old_coefficient':float(old[i]),
                 'mean_absolute_candidate_difference_contribution':float(contribution[i]/max(denominator,1)),
                 'fixed_model_feature_removal_switch_fraction':float(counts[i]/max(denominator,1)),
                 'candidate_variation_fraction':float(varying[i]/max(cases,1))}
                for i in np.argsort(-contribution)]
    report={'input_columns':len(names),'sampled_choice_states':cases,'candidate_vectors':candidate_count,
        'phase_counts':phase_counts,'coefficient_change_l2':float(np.linalg.norm(weights-old)),
        'new_vs_old_greedy_decision_change_fraction':old_new_changed/max(cases,1),
        'ranking':ranking(centered,switch,cases),
        'phase_rankings':{p:ranking(phase_contributions[p],phase_switch[p],phase_counts[p]) for p in phase_counts},
        'group_ablation':[{'group':g,'fixed_model_removal_switch_fraction':group_counts[g]/max(cases,1),
            'mean_lost_score':group_effect[g]/max(cases,1)} for g in sorted(group_counts,key=lambda g:-group_counts[g])],
        'state_only_in_sample':[names[i] for i in range(len(names)) if varying[i]==0],
        'examples':examples,'source_split':'live_training only',
        'method':'Mean absolute w_j*(z_aj - candidate_mean_j), with frozen model; equal weight per decision',
        'limitations':['Coefficient magnitude alone is not decision importance','Shared state features cancel between actions, except indirect future learning effects',
            'Feature removal describes reliance of this fixed model; not game win impact or performance after retraining',
            'Correlated features share attribution; grouped effects and per-feature effects differ','Explanations are training-distribution samples, not a held-out predictive metric']}
    (root/'decision_influence.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'choice_states':cases,'top_normal_decision_features':report['phase_rankings']['normal'][:10],
        'old_new_choice_change_fraction':report['new_vs_old_greedy_decision_change_fraction']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
