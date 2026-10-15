"""Attach completed engine audit, honest recovery timing and experiment notes."""
import os
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import argparse
import json
from pathlib import Path
import shutil
import mlflow

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--original-run", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    tracking = json.loads((root / "tracking.json").read_text())
    report = json.loads((root / "report.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    config = json.loads((root / "config.json").read_text())
    assert audit["status"] == "passed" and audit["games_replayed"] == 400
    assert report["weights_unchanged"] and all(g["completed"] == 100 for g in report["conditions"].values())
    mlflow.set_tracking_uri(tracking["tracking_uri"])
    client = mlflow.MlflowClient()
    prior = client.get_run(args.original_run)
    current = client.get_run(tracking["run_id"])
    assert prior.info.status == "FAILED" and current.info.status == "FINISHED"
    report["recovery_execution_seconds"] = report.get("recovery_execution_seconds", report["elapsed_seconds"])
    report["original_execution_seconds"] = (prior.info.end_time - prior.info.start_time) / 1000.
    report["elapsed_seconds"] = report["original_execution_seconds"] + report["recovery_execution_seconds"]
    report["original_failed_run_id"] = args.original_run
    report["technical_recovery"] = config["technical_recovery"]
    report["audit"] = audit
    report["test_verification"] = {"distinct_tests_passed": 43,
                                   "scope": "duel/search, spatial/strategic/build feature encoders, belief sampling regression"}
    (root / "report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    groups = report["conditions"]
    table = "| v4 search | Opponent | v4 wins | F wins | Mean v4 decision |\n|---|---|---:|---:|---:|\n"
    for key, label in (("neither", "No"), ("f_only", "No"), ("v4_only", "Yes"), ("both", "Yes")):
        group = groups[key]
        opponent = "F_static" if key in ("neither", "v4_only") else "native F (one-action search)"
        table += f"| {label} | {opponent} | {group['v4_wins']}/100 | {group['F_wins']}/100 | {1000*group['timing']['v4']['mean_seconds']:.2f} ms |\n"
    text = f"""# Frozen v4 versus one F opponent: two-player search comparison

v4 with bounded search won **24/100** against native F; the same frozen v4 without search won **30/100**. The paired difference is not statistically significant (12 gained wins, 18 lost wins, exploratory two-sided p={report['paired']['f_only_to_both']['paired_p_unadjusted']:.4f}). This pilot provides no evidence to replace direct v4 scoring with this search wrapper.

{table}

## Fixed protocol

Four conditions use the same 100 fresh boards, initial environment streams and learner seats (50 in each relative seat), with exactly two real players and a target of 10 VP. All 400 games reached victory. No training, exploration, hyperparameter tuning or registry alias promotion occurred. Weights and train-fitted normalization from `runs/online-sarsa-features-100-v4` remain unchanged. Canonical tie-breaking is used consistently in the direct and searched v4 variants; this differs from randomized ties in the previous four-player evaluator.

The simulator uses standard Catanatron base rules with two players, without additional official two-player variant rules. v4 was trained with four players. An explicit `allow_fewer_players` transfer flag zero-pads the raw features for absent relative seats/opponents while retaining the original 264-feature order, offsets, scales and clipping. Public/own features and strategic calculations use only the actual two players; no phantom players are introduced. Default four-player encoding stays unchanged. These two-player rates cannot be compared directly with v4's earlier 32/100 against three F_static opponents.

## What depth means here

Direct v4 ranks all current legal actions by the original frozen SARSA Q. The searched v4 expands the top three Q-ranked root actions, with two sampled public-count beliefs per choice, continuing the actual opponent policy until the next v4 decision. It then backs up terminal win/loss or the original .05-own-VP potential reward plus discounted maximum Q (gamma .999 per simulated table action). An entire decision falls back to direct Q if any branch cannot reach its endpoint within 24 actions; zero such fallbacks occurred in these games. This is an approximate next-own-decision search, not two full turns or full minimax.

Native F is the installed, unmodified `ValueFunctionPlayer`: it executes each legal candidate once on a copy and evaluates its hand-coded state value. Its synthetic view has detached randomness, so candidate simulations cannot consume the real future dice/card stream. Native F is a different heuristic from F_static; changing F_static to F changes both evaluation logic and forward simulation, so that contrast is not a pure depth-only ablation. The two sides use different depth definitions and computational budgets. Internal candidate simulations inside F are omitted from v4's branch-action statistics.

Both search policies receive only their own known cards and public observations through synthetic beliefs. Opponent exact private cards and actual future randomness are not used. Real future draws may diverge after policies choose different moves, despite identical initial seeds.

## Recovery and verification

The initial run finished 399 games but failed to sample a live belief in one late-game decision of `both/game_0028` within 128 rejection attempts. The same deterministic sample sequence now permits 4096 attempts, preserving choices where any of the first 128 samples succeeded. The missing game was rerun from the same original seed; the 399 complete games were retained. The failed partial trace is preserved under `both/game_0028_failed_attempt`. A regression test reproduces the old failure and verifies the new sampler returns a live view without changing real state/RNG. No failed game is counted as a loss or omitted from the 100-game denominator.

All **400 games / {audit['actions_replayed']:,} actions** were independently replayed through the engine, checking predecision whitelisted observations, complete legal-action lists, selected actions, final state and winners. In 12 games (first, last and index 28 in all four conditions), **{audit['policy_decisions_rescored']:,} choices** were recomputed exactly, including bounded search. **{audit['privacy_checks']} privacy-invariance checks** and 43 distinct focused tests passed. Initial boards were checked against all prior recorded boards and normalization-source boards under both hash conventions. Final-state comparison excludes the independently generated UUID; all simulator state and game rule metadata match.

Measured execution time, including the initial failed batch and recovery: **{report['elapsed_seconds']:.1f} seconds**; independent replay/scoring audit: **{audit['elapsed_seconds']:.1f} seconds**. Preparation, debugging pause, tests and final documentation are not included. The recovery-only {report['recovery_execution_seconds']:.1f} seconds is not the full benchmark duration.

## Interpretation and saved artifacts

Search changes many decisions but did not improve measured wins in this pilot. An inaccurate Q continuation, beam omissions, sparse belief sampling and transfer from four-player training can limit or degrade search. This experiment does not isolate which factor is responsible, nor prove every deeper search would fail. The five paired comparisons are exploratory and unadjusted for multiple testing.

All traces, complete games, protocol, copied model/normalization, results, audit and source snapshots are in `runs/v4-vs-f-duel-search-100-v1`. Small reports, source and documentation are attached to MLflow experiment `CatanZero_V4_F_Duel_Search`, completed run `{tracking['run_id']}`; the original technical failure remains recorded as failed run `{args.original_run}`. No new trained model was produced; v4's existing registered model and aliases are preserved.
"""
    document = REPO / "docs/v4-vs-f-duel-search-100-v1.md"
    document.write_text(text, encoding="utf-8")
    (root / "README.md").write_text(text, encoding="utf-8")
    for name in ("tools/audit_v4_f_duel.py", "tools/finalize_v4_f_duel.py", "tests/test_duel_search.py"):
        dest = root / "implementation_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, dest)
    with mlflow.start_run(run_id=tracking["run_id"]):
        mlflow.log_metrics({"elapsed_seconds": report["elapsed_seconds"],
                            "audit_games_replayed": audit["games_replayed"],
                            "audit_actions_replayed": audit["actions_replayed"],
                            "audit_choices_rescored": audit["policy_decisions_rescored"],
                            "audit_privacy_checks": audit["privacy_checks"], "focused_tests_passed": 43})
        mlflow.set_tags({"audit": "passed", "source_weights_unchanged": "true", "technical_recovery": "one failed belief sample; complete denominator restored"})
        for file in (root / "report.json", root / "audit.json", root / "README.md", document):
            mlflow.log_artifact(str(file), "audit_and_documentation")
        mlflow.log_artifact(str(root / "report.json"), "evaluation")
        mlflow.log_artifacts(str(root / "implementation_snapshot"), "implementation_snapshot")
    final = client.get_run(tracking["run_id"])
    assert final.info.status == "FINISHED"
    print(json.dumps({"run_id": tracking['run_id'], "status": final.info.status,
                      "full_execution_seconds": report['elapsed_seconds'], "docs": str(document),
                      "v4_wins": {key: group['v4_wins'] for key,group in groups.items()}}))


if __name__ == "__main__":
    main()
