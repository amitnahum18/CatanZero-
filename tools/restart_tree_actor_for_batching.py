"""Stop only this owned experiment; preserve completed phases and all data."""
from pathlib import Path
import json
import shutil
import psutil

repo=Path(__file__).resolve().parents[1]
root=repo/'runs/tree-policy-actor-rl-v5'
matches=[p for p in psutil.process_iter() if p.name().lower().startswith('python')
         and 'optimize_tree_actor_rl.py' in ' '.join(p.cmdline()) and Path(p.cwd()).resolve()==repo]
if not matches:raise ValueError('expected running owned trainer')
main=max(matches,key=lambda p:p.memory_info().rss)
children=main.children(recursive=True)
for p in children:
    assert str(root).lower() in ' '.join(p.cmdline()).lower()
    assert Path(p.cwd()).resolve()==repo
snapshot=root/'pre_batching_restart';snapshot.mkdir(exist_ok=False)
for name in ('config.json','tracking.json','fit.json','actor_data.npz','selection_history.json'):
    if (root/name).exists():shutil.copyfile(root/name,snapshot/name)
main.terminate()
for p in children:
    if p.is_running():p.terminate()
psutil.wait_procs([main]+children,timeout=10)
assert not main.is_running()
archived=[]
for phase in json.loads((root/'config.json').read_text())['phase_plan']:
    source=root/phase
    if source.exists() and not (source/'summary.json').exists():
        target=snapshot/phase
        assert source.resolve().is_relative_to(root.resolve()) and target.resolve().is_relative_to(root.resolve())
        # Python literal Path objects, no shell-composed filesystem operations.
        shutil.move(source,target);archived.append(phase)
print(json.dumps({'stopped_owned_main':main.pid,'stopped_children':len(children),
                  'archived_incomplete_phases':archived,'completed_phases_and_models_retained':True}))
