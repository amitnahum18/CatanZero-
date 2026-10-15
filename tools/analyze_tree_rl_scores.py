"""Measure scoring influence and decision-varying signals on training states."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from continue_champion_tree_rl import load_policy
from train_residual_tree_fqi import write


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    root=parser.parse_args().root.resolve();normalizer=json.loads((root/'normalizer.json').read_text())
    data=dict(np.load(root/'RL_round_1/experience.npz'))
    ranges=[(a,b) for a,b in data['ranges'] if b-a>1][:2000]
    xs=[data['next_x'][a:b] for a,b in ranges];raw=np.concatenate(xs)
    models=[load_policy(root/f'candidate_{v}',normalizer) for v in (500,501,502)]
    teacher=models[0].teacher.predict(raw,num_threads=1)/models[0].scale
    varying=np.any(np.stack([np.ptp(x,axis=0)>1e-12 for x in xs]),axis=0)
    names=normalizer['feature_names'];reports={}
    for model in models:
        total=model.predict(raw);value=model.mc.predict(raw)
        cursor=0;changed=0;spreads=[];margins=[]
        for x in xs:
            n=len(x);base=teacher[cursor:cursor+n];combined=total[cursor:cursor+n]
            changed+=int(np.argmax(base)!=np.argmax(combined));spreads.append(float(np.ptp(model.mix*value[cursor:cursor+n])))
            top=np.sort(base);margins.append(float(top[-1]-top[-2]));cursor+=n
        importance=model.mc.feature_importances_;top=np.argsort(importance)[-12:][::-1]
        reports[str(model.version)]={'mix':model.mix,'decisions':len(xs),'changed_actions':changed,
              'fraction_changed_from_teacher_only':changed/len(xs),'median_weighted_MC_spread':float(np.median(spreads)),
              'median_teacher_top_two_margin':float(np.median(margins)),
              'MC_importance_on_decision_varying_features':float(importance[varying].sum()),
              'top_MC_features':[{'feature':names[i],'importance':float(importance[i]),
                                  'varies_between_legal_actions_in_sample':bool(varying[i])} for i in top]}
    report={'source':'first 2000 multi-action REAL next decision sets from training only',
            'policies':reports,'probability_calibration_claimed':False,
            'reward_assessment':{'terminal_objective':'maximize actual probability of winning against native F',
             'gamma':1.,'win':1.,'loss':-1.,'return':'outcome - Phi(current)',
             'Phi_is_constant_between_actions_in_same_state':True,
             'intermediate_shaping_changes_full_MC_optimal_action_order':False,
             'teacher_scores':'ordinal relative preference, not true expected return',
             'MC_limitation':'one observed action/outcome per state; unchosen counterfactuals remain unknown',
             'replay_limitation':'old exploratory-policy data mixed with current-policy data; not unbiased current-policy evaluation',
             'future_option':'more current-policy episodes, controlled exploration, validation-gated policy improvement'},
            'importance_limitation':'impurity importance is descriptive and affected by correlated features; not causal'}
    write(root/'ranking_assessment.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
