"""Stop this task's driver only after a complete batch checkpoint is committed."""
import argparse
import json
from pathlib import Path
import time
import psutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
args = parser.parse_args()
root = args.root.resolve()
matches = []
for process in psutil.process_iter(["pid", "cmdline", "cwd", "create_time"]):
    try:
        cmd = process.info["cmdline"] or []
        if not any(Path(part).name == "train_gbrl_curriculum.py" for part in cmd) or "--output" not in cmd:
            continue
        output = Path(cmd[cmd.index("--output")+1])
        if not output.is_absolute():
            output = Path(process.info["cwd"]) / output
        if output.resolve() == root:
            matches.append((process, process.info["create_time"]))
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        continue
if len(matches) > 1:
    # Windows venv launchers may mirror the interpreter's command line.
    # Select the actual interpreter, not its forwarding launcher.
    matching_ids = {p.pid for p, _ in matches}
    matches = [(p, created) for p, created in matches
               if not any(child.pid in matching_ids for child in p.children(recursive=True))]
assert len(matches) == 1, f"Expected exactly one task driver, found {len(matches)}"
target, created = matches[0]
print(f"Watching task driver {target.pid}; will stop after a full batch and frozen evaluation", flush=True)
while target.is_running():
    try:
        progress = json.loads((root / "progress.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        time.sleep(.1)
        continue
    if progress.get("batches"):
        # The driver writes progress only after finishing the MLflow run and
        # closing every worker belonging to the completed batch.
        if target.create_time() != created:
            raise RuntimeError("Task driver PID was reused")
        children = target.children(recursive=True)
        owned = []
        for child in children:
            cmd = child.cmdline()
            if "catanzero.online_env" in cmd:
                config = Path(cmd[cmd.index("catanzero.online_env")+1]).resolve()
                if root not in config.parents:
                    raise RuntimeError("Refusing to stop a worker outside this task")
                owned.append(child)
        target.terminate()
        for child in owned:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs([target, *owned], timeout=3)
        outcome = {"stopped_after_complete_checkpoint": True, "driver_pid": target.pid,
                   "completed_batches": len(progress["batches"]), "complete_checkpoint": progress["batches"][-1]["directory"],
                   "reason": "review complete frozen evaluation before starting another batch"}
        (root / "checkpoint_review.json").write_text(json.dumps(outcome, indent=2))
        print(json.dumps(outcome, indent=2), flush=True)
        break
    time.sleep(.1)
