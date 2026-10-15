"""Extract a causal history-feature prototype from previously recorded training games."""
import argparse
import gzip
import json
from pathlib import Path
from catanzero.temporal_features import TemporalFeatures


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    results=sorted((json.loads(s) for s in (args.source/'live_training/results.jsonl').read_text().splitlines()),key=lambda r:r['game_index'])
    counts={1:0,3:0,5:0}
    decisions=0
    names=None
    memory=TemporalFeatures()
    with gzip.open(args.output/'history_preview.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as target:
        for result in results:
            directory=args.source/'live_training'/f"game_{result['game_index']:04d}"
            game_key=str(directory.resolve())
            with gzip.open(directory/'decisions.jsonl.gz','rt',encoding='utf-8') as source:
                for line in source:
                    row=json.loads(line)
                    if row['policy']!='Linear_SARSA':
                        continue
                    inputs=row['inputs']
                    obs=inputs['observation']
                    assert obs['actor']==result['learner_color']
                    features=memory.features(obs,game_key)
                    if names is None:
                        names=list(features)
                    assert names==list(features)
                    for lag in counts:
                        counts[lag]+=features[f'history_{lag}_available']==1
                    target.write(json.dumps({'metadata':{'game_index':result['game_index'],'action_index':row['action_index']},
                                             'features':features})+'\n')
                    memory.commit(obs,game_key,inputs['action'][1])
                    decisions+=1
    report={'source_split':'training only','games':len(results),'decisions':decisions,'feature_count':len(names),
            'feature_names':names,'available_history_counts':counts,
            'lag_unit':'prior decisions of the same actor; not table actions or complete turns',
            'causal_order':'compute current history features before committing current observation/action',
            'limitations':['experimental extraction only; no model trained or deployed','state-history summaries need action interactions for direct linear policy effects',
                           'resource changes mix own spending, production, trades and robber effects','not a calibrated hidden-card posterior']}
    (args.output/'history_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('games','decisions','feature_count','available_history_counts')}))


if __name__=='__main__':
    main()
