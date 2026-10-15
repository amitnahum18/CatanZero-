"""Bounded two-decision search using frozen linear Q values and public beliefs."""
import argparse
import gzip
import json
from pathlib import Path
import random
from time import perf_counter

from catanatron.serialization import action_to_json, map_to_json, state_to_json
from catanatron.players.value import ValueFunctionPlayer
from catanzero.dataset import observation, ordered_actions
from catanzero.features import FeatureEncoder, VERSION
from catanzero.online_linear import LinearActionValues
from catanzero.search_labels import live_belief
from catanzero.simulation import SimulationConfig, create_game
from catanzero.static_policies import ObservationPolicy, forbid_forward_simulation
from catanzero.tournament import atomic_json, canonical_hash


def backed_up_value(previous_vp, following_vp, steps, continuation, *, won=None, gamma=.999, vp_weight=.05):
    """Same semi-MDP reward/discount as the fitted SARSA model."""
    if steps < 1:
        raise ValueError("a branch must execute at least one action")
    if won is not None:
        return gamma**(steps-1)*int(won)-vp_weight*previous_vp
    discount = gamma**steps
    return discount*(vp_weight*following_vp+continuation)-vp_weight*previous_vp


class RegressionSearch:
    def __init__(self, model, normalizer, map_document, *, depth=1, beam=3, beliefs=2,
                 max_branch_actions=24, allow_fewer_players=False, opponent_policy="F_static"):
        if depth not in (1, 2) or min(beam, beliefs, max_branch_actions) < 1:
            raise ValueError("depth 1 or 2 and positive budgets required")
        if opponent_policy not in ("F_static", "F"):
            raise ValueError("unsupported search continuation opponent")
        self.scorer = LinearActionValues(model, normalizer)
        self.teacher = ObservationPolicy("F_static", map_document, {}, allow_fewer_players=allow_fewer_players)
        self.encoder = FeatureEncoder(map_document, version=normalizer.get("feature_version", VERSION),
                                      allow_fewer_players=allow_fewer_players)
        self.opponent_policy = opponent_policy
        self.depth, self.beam, self.beliefs, self.max_branch_actions = depth, beam, beliefs, max_branch_actions
        self.cache = {}
        self.stats = {"simulated_actions": 0, "leaf_action_predictions": 0, "leaf_cache_hits": 0,
                      "searched_decisions": 0, "changed_choices": 0, "budget_fallbacks": 0}

    def action_values(self, inputs):
        return [self.scorer.predict(self.encoder.encode({**inputs, "action": action}))
                for action in inputs["legal_actions"]]

    def leaf_value(self, game, actor):
        actions = ordered_actions(game)
        inputs = {"observation": observation(game, actor), "map_ref": 0,
                  "legal_actions": [action_to_json(a) for a in actions]}
        # Local to one real decision. Key excludes actual hidden cards and RNG.
        key = canonical_hash(inputs)
        if key in self.cache:
            self.stats["leaf_cache_hits"] += 1
            return self.cache[key]
        value = max(self.action_values(inputs))
        self.stats["leaf_action_predictions"] += len(actions)
        self.cache[key] = value
        return value

    def branch_value(self, proxy, action, actor, previous_vp, branch_seed):
        child = proxy.copy()
        # Engine copies share RNG objects. Every branch gets a detached stream.
        child.random = child.state.random = random.Random(branch_seed)
        child.state.action_records = []
        child.execute(action)
        steps = 1
        policy_rng = random.Random(branch_seed ^ 0xF01C)
        while child.winning_color() is None and child.state.current_color() != actor:
            if steps >= self.max_branch_actions:
                self.stats["simulated_actions"] += steps
                return None
            current = child.state.current_color()
            actions = ordered_actions(child)
            if len(actions) == 1:
                selected = actions[0]
            elif self.opponent_policy == "F":
                selected = public_f_choice(child, actions, policy_rng.getrandbits(128))
            else:
                inputs = {"observation": observation(child, current), "map_ref": 0,
                          "legal_actions": [action_to_json(a) for a in actions]}
                with forbid_forward_simulation():
                    selected_index, _ = self.teacher.choose(inputs)
                selected = actions[selected_index]
            child.execute(selected)
            steps += 1
        self.stats["simulated_actions"] += steps
        winner = child.winning_color()
        if winner is not None:
            return backed_up_value(previous_vp, 0, steps, 0, won=winner == actor)
        obs = observation(child, actor)
        own = next(p for p in obs["players"] if p["color"] == actor.value)
        return backed_up_value(previous_vp, own["actual_vp"], steps, self.leaf_value(child, actor))

    def choose(self, game, actions, seed):
        actor = game.state.current_color()
        inputs = {"observation": observation(game, actor), "map_ref": 0,
                  "legal_actions": [action_to_json(a) for a in actions]}
        self.cache = {}
        scores = self.action_values(inputs)
        ranked = sorted(range(len(actions)), key=lambda i: (-scores[i], i))
        baseline = ranked[0]
        detail = {"base_scores": scores, "search_scores": [None]*len(actions),
                  "expanded_indices": [], "depth": self.depth, "fallback": False}
        if self.depth == 1 or len(actions) == 1:
            return baseline, detail
        candidates = ranked[:self.beam]
        detail["expanded_indices"] = candidates
        own = next(p for p in inputs["observation"]["players"] if p["color"] == actor.value)
        previous_vp = own["actual_vp"]
        values = {i: [] for i in candidates}
        seeds = random.Random(seed)
        for _ in range(self.beliefs):
            belief_seed, branch_seed = seeds.getrandbits(128), seeds.getrandbits(128)
            proxy = live_belief(game, actor, belief_seed)
            # Same belief and sampling seed across candidates; divergent actions
            # can consume different draws, so futures are not identical.
            for i in candidates:
                values[i].append(self.branch_value(proxy, actions[i], actor, previous_vp, branch_seed))
        self.stats["searched_decisions"] += 1
        if any(v is None for samples in values.values() for v in samples):
            # Never compare a partial estimate or original Q to a complete backup.
            detail["fallback"] = True
            self.stats["budget_fallbacks"] += 1
            return baseline, detail
        for i, samples in values.items():
            detail["search_scores"][i] = sum(samples)/len(samples)
        selected = max(candidates, key=lambda i: (detail["search_scores"][i], -ranked.index(i)))
        self.stats["changed_choices"] += selected != baseline
        return selected, detail


def public_f_choice(game, actions, seed):
    """Unmodified native one-action F search on a synthetic public belief.

    Engine copies share RNG references. F may advance only this detached view's
    RNG while scoring; never pass the real game (or its actual future stream).
    """
    view = live_belief(game, game.state.current_color(), seed)
    selected = ValueFunctionPlayer(view.state.current_color()).decide(view, actions)
    if selected not in actions:
        raise ValueError("native F returned an illegal action")
    return selected


def run_match(config, depth, index, output):
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    seed = config["base_seed"]+index
    game = create_game(SimulationConfig(), seed)
    candidate = game.state.colors[index % 4]
    map_document = map_to_json(game.state.board.map)
    board_hash = canonical_hash(map_document)
    if board_hash in config["excluded_board_hashes"]:
        raise ValueError("test board overlaps source data")
    policy = RegressionSearch(json.loads(Path(config["model"]).read_text()),
                              json.loads(Path(config["normalizer"]).read_text()), map_document,
                              depth=depth, beam=config["beam"], beliefs=config["beliefs"],
                              max_branch_actions=config["max_branch_actions"])
    times = []
    opponent_guarded = 0
    reason = "running"
    metadata = {"depth": depth, "game_index": index, "seed": seed, "candidate_color": candidate.value,
                "candidate_seat": index % 4, "initial_map_hash": board_hash,
                "initial_stream_hash": canonical_hash(game.state.random.getstate())}
    with gzip.open(output / "decisions.jsonl.gz", "wt", encoding="utf-8", compresslevel=1) as trace:
        while game.winning_color() is None:
            if len(game.state.action_records) >= config["max_actions"]:
                reason = "action_limit"
                break
            if perf_counter()-started >= config["max_seconds"]:
                reason = "time_limit"
                break
            actor = game.state.current_color()
            actions = ordered_actions(game)
            if len(actions) == 1:
                selected = actions[0]
            elif actor == candidate:
                before = observation(game, actor)
                stream = game.state.random.getstate()
                tick = perf_counter()
                selected_index, detail = policy.choose(game, actions, seed*1000003+len(game.state.action_records))
                elapsed = perf_counter()-tick
                times.append(elapsed)
                assert stream == game.state.random.getstate() and before == observation(game, actor)
                selected = actions[selected_index]
                trace.write(json.dumps({"action_index": len(game.state.action_records), "observation": before,
                                        "legal_actions": [action_to_json(a) for a in actions],
                                        "selected_index": selected_index, "elapsed_seconds": elapsed, **detail})+"\n")
            else:
                inputs = {"observation": observation(game, actor), "map_ref": 0,
                          "legal_actions": [action_to_json(a) for a in actions]}
                with forbid_forward_simulation():
                    selected_index, _ = policy.teacher.choose(inputs)
                selected = actions[selected_index]
                opponent_guarded += 1
            game.execute(selected)
    winner = game.winning_color()
    result = {**metadata, "termination": "victory" if winner else reason,
              "candidate_won": winner == candidate if winner else None,
              "winner": winner.value if winner else None, "actions": len(game.state.action_records),
              "elapsed_seconds": perf_counter()-started, "decision_count": len(times),
              "decision_seconds_sum": sum(times), "decision_seconds_max": max(times, default=0),
              "opponent_guarded_decisions": opponent_guarded, "search_stats": policy.stats}
    atomic_json(output / "result.json", result)
    atomic_json(output / "game.json", state_to_json(game))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, choices=(1, 2), required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    result = run_match(json.loads(args.config.read_text()), args.depth, args.index, args.output)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
