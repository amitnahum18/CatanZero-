"""Aggregate native-F labels on the stronger learner's own visited states."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
from concurrent.futures import ThreadPoolExecutor,as_completed
import gzip
import json
from pathlib import Path
import pickle
import subprocess
from time import perf_counter
import numpy as np

REPO=Path(__file__).resolve().parents[1]
SOURCE=REPO/'runs/tree-policy-native-f-max-v1'
ROOT=REPO/'runs/tree-policy-dagger-max-v2'


def job(round_index,index):
    source=SOURCE/f'RL_round_{round_index}'/f'game_{index:04d}'
    output=ROOT/'labels'/f'round_{round_index}_game_{index:04d}'
    if (output/'result.json').exists():return json.loads((output/'result.json').read_text())
    with (ROOT/'labels'/f'worker_{round_index}_{index:04d}.log').open('w',encoding='utf-8') as log:
        subprocess.run([str(REPO/'.venv/Scripts/python.exe'),'-u','-m','catanzero.label_f_policy',
                        '--source',str(source),'--output',str(output)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,
                        timeout=180,check=True,env={**os.environ,'PYTHONHASHSEED':'0'})
    return json.loads((output/'result.json').read_text())


def main():
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'labels').mkdir(exist_ok=True);start=perf_counter();infos=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(job,r,i) for r in (1,2) for i in range(100)]
        for future in as_completed(futures):
            infos.append(future.result())
            if len(infos)%50==0:print(f'DAGGER LABELS {len(infos)}/200, {perf_counter()-start:.1f}s',flush=True)
    old=dict(np.load(SOURCE/'teacher_data.npz'));metadata=[json.loads(s) for s in (SOURCE/'teacher_groups.jsonl').read_text().splitlines()]
    x=[old['x']];y=[old['y']];groups=old['groups'].tolist();masks=old['validation_groups'].tolist()
    for r in (1,2):
        for index in range(100):
            with gzip.open(ROOT/'labels'/f'round_{r}_game_{index:04d}'/'labels.pkl.gz','rb') as stream:records=pickle.load(stream)
            for record in records:
                scores=record['score'];unique=sorted(set(scores));rank={s:i for i,s in enumerate(unique)}
                x.append(np.asarray(record['features'],dtype=np.float32));y.append(np.asarray([max(0,3-(len(unique)-1-rank[s])) for s in scores],dtype=np.int32))
                groups.append(len(scores));masks.append(index>=80)
                metadata.append({'round':r+3,'game_index':index,'seed':record['seed'],'action_index':record['action_index'],
                                 'teacher_index':record['teacher_index'],'candidate_count':len(scores)})
    np.savez_compressed(ROOT/'teacher_data.npz',x=np.concatenate(x),y=np.concatenate(y),groups=np.asarray(groups),validation_groups=np.asarray(masks,dtype=bool))
    (ROOT/'teacher_groups.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in metadata),encoding='utf-8')
    prior=json.loads((SOURCE/'teacher_report.json').read_text())
    report={'games':500,'initial_source_games':300,'learner_visited_games':200,'decision_groups':len(groups),'candidate_rows':sum(groups),
            'label_validation_games':100,'test_games_used':0,'source_board_hashes':sorted(set(prior['source_board_hashes'])|{i['initial_map_hash'] for i in infos}),
            'sources':prior['training_sources']+[str(SOURCE/f'RL_round_{r}') for r in (1,2)],'seconds':perf_counter()-start}
    (ROOT/'teacher_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_board_hashes'}),flush=True)


if __name__=='__main__':main()
