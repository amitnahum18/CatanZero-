"""Parallel free native-F labels from the 300 existing training games only."""
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
ROOT=REPO/'runs/tree-policy-native-f-max-v1'
SOURCE=REPO/'runs/residual-extratrees-depth10-f-duel-v1'


def job(round_index,index):
    source=SOURCE/f'training_round_{round_index}'/f'game_{index:04d}'
    output=ROOT/'labels'/f'round_{round_index}_game_{index:04d}'
    if (output/'result.json').exists():return json.loads((output/'result.json').read_text())
    command=[str(REPO/'.venv/Scripts/python.exe'),'-u','-m','catanzero.label_f_policy',
             '--source',str(source),'--output',str(output)]
    with (ROOT/'labels'/f'worker_{round_index}_{index:04d}.log').open('w',encoding='utf-8') as log:
        subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180,
                       env={**os.environ,'PYTHONHASHSEED':'0'})
    return json.loads((output/'result.json').read_text())


def main():
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'labels').mkdir(exist_ok=True)
    start=perf_counter();infos=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(job,r,i) for r in range(1,4) for i in range(100)]
        for future in as_completed(futures):
            infos.append(future.result())
            if len(infos)%50==0:print(f'F LABELS {len(infos)}/300, elapsed={perf_counter()-start:.1f}s',flush=True)
    x=[];y=[];weights=[];groups=[];metadata=[];validation=[]
    for round_index in range(1,4):
        for index in range(100):
            with gzip.open(ROOT/'labels'/f'round_{round_index}_game_{index:04d}'/'labels.pkl.gz','rb') as stream:
                records=pickle.load(stream)
            for record in records:
                features=np.asarray(record['features'],dtype=np.float32);scores=record['score']
                unique=sorted(set(scores));rank={s:i for i,s in enumerate(unique)}
                relevance=np.asarray([max(0,3-(len(unique)-1-rank[s])) for s in scores],dtype=np.int32)
                x.append(features);y.append(relevance);weights.append(np.full(len(features),1/len(features)))
                groups.append(len(features));validation.append(index>=80)
                metadata.append({'round':round_index,'game_index':index,'seed':record['seed'],'action_index':record['action_index'],
                                 'teacher_index':record['teacher_index'],'candidate_count':len(features)})
    np.savez_compressed(ROOT/'teacher_data.npz',x=np.concatenate(x),y=np.concatenate(y),weights=np.concatenate(weights),
                        groups=np.asarray(groups),validation_groups=np.asarray(validation,dtype=bool))
    (ROOT/'teacher_groups.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in metadata),encoding='utf-8')
    report={'games':300,'training_sources':[str(SOURCE/f'training_round_{r}') for r in range(1,4)],
            'decision_groups':len(groups),'candidate_rows':sum(groups),'validation_label_games':60,
            'feature_count':264,'uses_game_test_states':False,'seconds':perf_counter()-start,
            'labeler':'native F scores on detached public belief; best/second/third relevance 3/2/1, others 0',
            'source_board_hashes':sorted({i['initial_map_hash'] for i in infos})}
    (ROOT/'teacher_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_board_hashes'}),flush=True)


if __name__=='__main__':main()
