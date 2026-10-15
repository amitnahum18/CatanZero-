"""Counterfactual win-rate estimates under a public-information F continuation.

This is a bounded local teacher, not an optimal solver or calibrated expert.
Existing labels are preserved; uncertain and censored comparisons stay explicit.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import gzip
import json
from itertools import combinations
from math import comb, isfinite, sqrt
from multiprocessing import get_context
from pathlib import Path
import random
from statistics import NormalDist
from time import perf_counter

from catanatron.models.enums import ActionType
from catanatron.players.value import ValueFunctionPlayer
from catanatron.serialization import action_record_from_json, action_to_json, state_from_json, state_to_json
from catanatron import Color

from catanzero.cli import positive_int, write_json
from catanzero.dataset import PilotHeuristic, belief_game, observation, ordered_actions
from catanzero.simulation import SimulationConfig, SimulationRandomPlayer, create_game

VERSION = "public_belief_F_terminal_rollouts_v1"


@dataclass(frozen=True)
class SearchConfig:
    rollouts: int = 32
    candidates: int = 4
    max_actions: int = 3000
    max_turns: int = 1000
    screening_beliefs: int = 4
    max_seconds: float = 120.0

    def __post_init__(self):
        if any(not isfinite(value) or value <= 0 for value in asdict(self).values()) or self.candidates < 2:
            raise ValueError("positive search budgets and at least two candidates are required")


def wilson_interval(wins, trials, z=1.959963984540054):
    if trials == 0:
        return [0.0, 1.0]
    p = wins / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def paired_preferences(actions, outcomes):
    """Exact paired sign tests with correction for all ordered candidate pairs.

    Under the equal-win-rate null, discordant paired trials have equal signs.
    Censored comparisons are not tested; ties contribute no evidence.
    """
    threshold = 0.05 / (len(actions) * (len(actions) - 1))
    pairs = []
    for first, second in combinations(range(len(actions)), 2):
        if any(value is None for value in outcomes[first] + outcomes[second]):
            continue
        plus = sum(a > b for a, b in zip(outcomes[first], outcomes[second]))
        minus = sum(a < b for a, b in zip(outcomes[first], outcomes[second]))
        preferred, rejected = (first, second) if plus >= minus else (second, first)
        favorable, unfavorable = max(plus, minus), min(plus, minus)
        discordant = favorable + unfavorable
        p_value = sum(comb(discordant, k) for k in range(unfavorable + 1)) / 2 ** discordant if discordant else 1.0
        pairs.append({
            "preferred_action": action_to_json(actions[preferred]),
            "rejected_action": action_to_json(actions[rejected]),
            "favorable_trials": favorable, "unfavorable_trials": unfavorable,
            "tied_trials": len(outcomes[first]) - discordant,
            "win_rate_difference": (favorable - unfavorable) / len(outcomes[first]),
            "paired_one_sided_p_value": p_value, "corrected_threshold": threshold,
            "resolved": favorable > unfavorable and p_value < threshold,
        })
    return pairs


def live_belief(game, actor, seed, *, max_attempts=4096):
    """Condition synthetic hidden cards on the observed game still being live."""
    seeds = random.Random(seed)
    if max_attempts < 1:
        raise ValueError("positive belief sampling limit required")
    for _ in range(max_attempts):
        proxy = belief_game(game, actor, seeds.getrandbits(128))
        if proxy.winning_color() is None:
            return proxy
    raise ValueError("could not sample a live public-count belief")


def candidate_actions(game, selected, config, seed):
    actions = ordered_actions(game)
    if selected not in actions:
        raise ValueError("recorded choice is not legal in reconstructed state")
    if len(actions) <= config.candidates:
        return actions, "all_legal_actions"
    evaluator = PilotHeuristic(game)
    rng = random.Random(seed)
    totals = {action: 0.0 for action in actions}
    for _ in range(config.screening_beliefs):
        proxy = live_belief(game, selected.color, rng.getrandbits(128))
        for action in actions:
            totals[action] += evaluator.expected_value(proxy, action)
    ranked = sorted(actions, key=lambda action: -totals[action])
    # Preserve the recorded action, the top screened actions, and one randomly
    # selected remaining alternative so the heuristic is not the sole filter.
    candidates = [selected]
    for action in ranked:
        if action not in candidates and len(candidates) < config.candidates - 1:
            candidates.append(action)
    remaining = [a for a in actions if a not in candidates]
    if remaining:
        candidates.append(rng.choice(remaining))
    return sorted(candidates, key=lambda a: json.dumps(action_to_json(a))), "heuristic_screen_plus_random"


def rollout(game, action, seed, config, deadline=None):
    actor = action.color
    child = live_belief(game, actor, seed)
    # Separate environment randomness from all policy deliberation. Copies in
    # the engine share RNG references; no decision may advance this stream.
    child.random = child.state.random = random.Random(seed ^ 0xD1CE)
    child.state.action_records = []
    child.execute(action)
    policy_rng = random.Random(seed ^ 0xB311EF)
    policies = {color: ValueFunctionPlayer(color) for color in child.state.colors}
    steps = 1
    initial_turn = child.state.num_turns
    while child.winning_color() is None and steps < config.max_actions and child.state.num_turns - initial_turn < config.max_turns and (deadline is None or perf_counter() < deadline):
        actions = ordered_actions(child)
        current = child.state.current_color()
        if len(actions) == 1:
            decision = actions[0]
        else:
            view = live_belief(child, current, policy_rng.getrandbits(128))
            decision = policies[current].decide(view, actions)
        if decision not in actions:
            raise ValueError("continuation policy returned an illegal move")
        child.execute(decision)
        steps += 1
    winner = child.winning_color()
    return (None if winner is None else int(winner == actor)), steps


def evaluate_actions(game, selected, config, seed):
    started = perf_counter()
    deadline = started + config.max_seconds
    actions, screening = candidate_actions(game, selected, config, seed)
    rng = random.Random(seed ^ 0x5EA2C4)
    # Same initial belief and environment seeds across alternatives in a trial.
    # Branches consume different draws; this is not an identical future path.
    seeds = [rng.getrandbits(128) for _ in range(config.rollouts)]
    comparisons = []
    total_steps = 0
    # Bonferroni-adjusted approximate Wilson intervals, including censoring
    # bounds. The coverage is approximate, not an expert confidence guarantee.
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * len(actions)))
    trial_outcomes = [[] for _ in actions]
    attempted = [0 for _ in actions]
    # Interleave candidates so a time budget does not spend all effort on the
    # first alternative. Unevaluated allocated trials remain censored, not losses.
    for trial_seed in seeds:
        for index, action in enumerate(actions):
            if perf_counter() >= deadline:
                outcome, steps = None, 0
            else:
                outcome, steps = rollout(game, action, trial_seed, config, deadline)
                attempted[index] += 1
            trial_outcomes[index].append(outcome)
            total_steps += steps
    for index, action in enumerate(actions):
        outcomes = trial_outcomes[index]
        wins = sum(outcome == 1 for outcome in outcomes)
        truncated = outcomes.count(None)
        completed = config.rollouts - truncated
        lower = wilson_interval(wins, config.rollouts, z)[0]
        upper = wilson_interval(wins + truncated, config.rollouts, z)[1]
        comparisons.append({
            "action": action_to_json(action), "wins": wins,
            "losses": completed - wins, "truncated": truncated,
            "rollouts": config.rollouts,
            "rollouts_attempted": attempted[index],
            "not_attempted": config.rollouts - attempted[index],
            "trial_outcomes": outcomes,
            # Avoid silently conditioning estimates on completing the game.
            "win_rate": wins / config.rollouts if not truncated else None,
            "win_rate_bounds": [wins / config.rollouts, (wins + truncated) / config.rollouts],
            "interval_approx_95_familywise": [lower, upper],
        })
    complete = all(c["win_rate"] is not None for c in comparisons)
    best = max(comparisons, key=lambda c: c["win_rate_bounds"][0])
    others = [c for c in comparisons if c is not best]
    resolved = all(best["interval_approx_95_familywise"][0] > c["interval_approx_95_familywise"][1] for c in others)
    selected_result = next(c for c in comparisons if c["action"] == action_to_json(selected))
    regret = best["win_rate"] - selected_result["win_rate"] if complete else None
    pairs = paired_preferences(actions, trial_outcomes)
    return {
        "version": VERSION, "continuation_policy": "F on fresh public-count belief at each decision",
        "candidate_selection": screening, "legal_action_count": len(game.playable_actions),
        "all_legal_actions_evaluated": len(actions) == len(game.playable_actions),
        "candidates": comparisons, "best_estimated_action": best["action"],
        "ranking_resolved": resolved, "search_estimate_available": complete,
        "pairwise_comparisons": pairs,
        "resolved_pairwise_count": sum(pair["resolved"] for pair in pairs),
        "selected_win_rate": selected_result["win_rate"],
        "regret_among_candidates": regret,
        "quality_score": 1 - regret if regret is not None else None,
        "time_budget_exhausted": any(n < config.rollouts for n in attempted) or perf_counter() >= deadline,
        "simulated_actions": total_steps, "elapsed_seconds": perf_counter() - started,
    }


def sample_rows(source, limit, seed, sampling="stratified"):
    if (source / "FAILED.json").exists():
        raise ValueError("partial/failed source datasets must not be used")
    rng = random.Random(seed)
    groups = {}
    reservoir = []
    count = 0
    results = {r["game_index"]: r for r in (json.loads(line) for line in (source / "results.jsonl").read_text(encoding="utf-8").splitlines())}
    for result in results.values():
        with gzip.open(source / result["moves_file"], "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if not row["train_eligible"]:
                    continue
                if sampling == "critical":
                    view = row["inputs"]["observation"]
                    own = next(p for p in view["players"] if p["color"] == view["actor"])
                    rivals = [p for p in view["players"] if p["color"] != view["actor"]]
                    if own["actual_vp"] < 8 or max(p["VICTORY_POINTS"] for p in rivals) < 7:
                        continue
                group = (row["inputs"]["action"][1], row["inputs"]["observation"]["initial_phase"])
                seen, previous = groups.get(group, (0, None))
                groups[group] = (seen + 1, row if rng.randrange(seen + 1) == 0 else previous)
                count += 1
                if len(reservoir) < limit:
                    reservoir.append(row)
                else:
                    index = rng.randrange(count)
                    if index < limit:
                        reservoir[index] = row
    # Cover rare action/phase categories first; fill from a uniform reservoir.
    selected = [r for _, r in sorted(groups.values(), key=lambda item: item[0])[:limit]]
    identities = {(r["game_index"], r["action_index"]) for r in selected}
    for row in reservoir:
        identity = (row["game_index"], row["action_index"])
        if len(selected) < limit and identity not in identities:
            selected.append(row)
            identities.add(identity)
    if not selected:
        raise ValueError("source contains no eligible decisions")
    return sorted(selected, key=lambda r: (r["game_index"], r["action_index"])), results


def reconstruct(source, row):
    document = json.loads((source / "games" / f"game_{row['game_index']:04d}.json").read_text(encoding="utf-8"))
    game = create_game(SimulationConfig(players=len(document["colors"])), document["game"]["seed"])
    if state_to_json(game)["map"] != document["map"] or [c.value for c in game.state.colors] != document["colors"]:
        raise ValueError("saved board/seating does not match seeded reconstruction")
    for payload in document["action_records"][:row["action_index"]]:
        record = action_record_from_json(payload)
        action = record.action
        if action.action_type in (ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD):
            action = action._replace(value=None)
        game.execute(action, action_record=record)
    current = json.loads(json.dumps(observation(game, game.state.current_color())))
    if current != row["inputs"]["observation"]:
        raise ValueError("source observation differs from reconstructed position")
    legal = [action_to_json(a) for a in ordered_actions(game)]
    if json.loads(json.dumps(legal)) != row["inputs"]["legal_actions"]:
        raise ValueError("source legal actions differ from reconstructed position")
    return game


def label_job(job):
    document, row, config, seed, split = job
    players = [SimulationRandomPlayer(Color[c]) for c in document["colors"]]
    game = state_from_json(document, players)
    action = next(a for a in ordered_actions(game) if action_to_json(a) == row["inputs"]["action"])
    search = evaluate_actions(game, action, config, seed)
    outcome = row["targets"]["terminal_outcome"]
    weighted = 0.5 * search["quality_score"] + 0.5 * outcome if search["quality_score"] is not None and outcome is not None else None
    return {**row, "source_split": split, "search": search,
            "targets": {**row["targets"], "search_win_rate": search["selected_win_rate"],
                        "search_quality_score": search["quality_score"],
                        "search_weighted_score": weighted},
            "search_ranking_eligible": search["resolved_pairwise_count"] > 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Add local counterfactual terminal-rollout labels to sampled existing decisions.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=positive_int, default=24)
    parser.add_argument("--rollouts", type=positive_int, default=32)
    parser.add_argument("--candidates", type=positive_int, default=4)
    parser.add_argument("--workers", type=positive_int, default=8)
    parser.add_argument("--max-actions", type=positive_int, default=3000)
    parser.add_argument("--max-turns", type=positive_int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=120.0,
                        help="Per-position wall budget, including screening; checked between actions, not a hard process timeout.")
    parser.add_argument("--seed", type=int, default=70000)
    parser.add_argument("--sampling", choices=("stratified", "critical"), default="stratified",
                        help="Critical enriches close late-game decisions; do not use it as the only training distribution.")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be a new directory")
    config = SearchConfig(args.rollouts, args.candidates, args.max_actions, args.max_turns,
                          max_seconds=args.max_seconds)
    started = perf_counter()
    rows, results = sample_rows(args.source, args.samples, args.seed, args.sampling)
    jobs = []
    for index, row in enumerate(rows):
        game = reconstruct(args.source, row)
        jobs.append((state_to_json(game), row, config, args.seed + index, results[row["game_index"]]["split"]))
    args.output.mkdir(parents=True, exist_ok=False)
    workers = min(args.workers, len(jobs))
    write_json(args.output / "config.json", {
        "version": VERSION, "source_dataset": str(args.source.resolve()),
        "search": asdict(config), "seed": args.seed, "workers": workers,
        "sampling": args.sampling,
        "weighted_search_formula": "0.5 * (1 - regret_among_candidates) + 0.5 * original_terminal_outcome",
        "limitations": ["not expert or optimal play", "F continuation considers only one enemy in its heuristic",
                        "public-count beliefs, no history posterior", "candidate screening may omit the best legal move",
                        "interval coverage approximate; labels are not calibrated confidence", "no player-to-player trade offers"],
    })
    (args.output / "maps.jsonl").write_bytes((args.source / "maps.jsonl").read_bytes())
    summary = Counter()
    print(f"Search-labeling {len(jobs)} positions with {workers} workers", flush=True)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool, (args.output / "search.moves.jsonl").open("x", encoding="utf-8") as stream:
            for index, row in enumerate(pool.map(label_job, jobs)):
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                search = row["search"]
                summary["positions"] += 1
                summary["ranking_resolved"] += int(search["ranking_resolved"])
                summary["positions_with_resolved_pairs"] += int(search["resolved_pairwise_count"] > 0)
                summary["resolved_pairs"] += search["resolved_pairwise_count"]
                summary["fully_evaluated_positions"] += int(search["all_legal_actions_evaluated"])
                summary["simulated_actions"] += search["simulated_actions"]
                summary["rollouts"] += sum(c["rollouts"] for c in search["candidates"])
                summary["truncated_rollouts"] += sum(c["truncated"] for c in search["candidates"])
                print(f"[{index + 1}/{len(jobs)}] {row['inputs']['action'][1]} alternatives={len(search['candidates'])} seconds={search['elapsed_seconds']:.2f} unique_best={search['ranking_resolved']} resolved_pairs={search['resolved_pairwise_count']}", flush=True)
    except Exception as error:
        write_json(args.output / "FAILED.json", {"error": str(error), "partial": True})
        raise
    write_json(args.output / "summary.json", {**summary, "elapsed_seconds": perf_counter() - started, "workers": workers})
    print(f"Search labels: {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
