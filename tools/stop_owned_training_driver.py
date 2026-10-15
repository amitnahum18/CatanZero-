"""Stop only a specifically named local task driver and its game workers."""
import argparse
import json
from pathlib import Path
import psutil

parser = argparse.ArgumentParser()
parser.add_argument("script", choices=["evaluate_tree_checkpoint.py", "train_gbrl_curriculum.py"])
parser.add_argument("output", type=Path)
args = parser.parse_args()
root = args.output.resolve()
matches = []
for p in psutil.process_iter(["cmdline", "cwd"]):
    try:
        cmd = p.info["cmdline"] or []
        if not any(Path(s).name == args.script for s in cmd) or "--output" not in cmd:
            continue
        value = Path(cmd[cmd.index("--output")+1])
        if not value.is_absolute():
            value = Path(p.info["cwd"]) / value
        if value.resolve() == root:
            matches.append(p)
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        continue
ids = {p.pid for p in matches}
matches = [p for p in matches if not any(child.pid in ids for child in p.children(recursive=True))]
assert len(matches) == 1, f"Expected one owned driver, found {len(matches)}"
p = matches[0]
owned = []
for child in p.children(recursive=True):
    try:
        cmd = child.cmdline()
        if "catanzero.online_env" not in cmd:
            raise RuntimeError("Refusing to stop an unrecognized child process")
        config = Path(cmd[cmd.index("catanzero.online_env")+1]).resolve()
        assert root in config.parents
        owned.append(child)
    except psutil.NoSuchProcess:
        pass
p.terminate()
for child in owned:
    try:
        child.terminate()
    except psutil.NoSuchProcess:
        pass
_, alive = psutil.wait_procs([p, *owned], timeout=3)
for child in alive:
    child.kill()
print(json.dumps({"stopped_driver": p.pid, "owned_workers_stopped": len(owned), "output": str(root)}))
