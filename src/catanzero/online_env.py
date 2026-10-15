"""Persistent pinned-engine worker: real play, waiting for central decisions."""

import argparse
import gzip
from itertools import permutations
import json
import random
from pathlib import Path
import sys
from time import perf_counter

from catanatron import Game
from catanatron.registry import REGISTRY
import catanatron.players
from catanatron.serialization import action_to_json, map_to_json, state_to_json

from catanzero.dataset import observation, ordered_actions
from catanzero.search_labels import live_belief
from catanzero.simulation import SimulationConfig, create_game
from catanzero.static_policies import ObservationPolicy, forbid_forward_simulation
from catanzero.tournament import atomic_json, canonical_hash
from catanzero.online_linear import LinearActionValues
from catanzero.features import FeatureEncoder, VERSION
from catanzero.board_exclusions import board_hash_variants


class OnlineEnvironment:
    def __init__(self, config):
        self.config = config
        self.game = self.stream = self.cached = None

    def reset(self, index, output):
        if self.stream is not None:
            raise ValueError("cannot reset an unfinished game")
        self.index, self.output = index, Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = perf_counter()
        self.seed = self.config["base_seed"] + index
        self.players = self.config.get("players", 4)
        self.opponent_name = self.config.get("opponent_policy", "F_static")
        search_opponent = self.opponent_name.split(":", 1)[0].upper() in ("AB", "M")
        if self.opponent_name == "Mixed_linear_F_static":
            self.opponent_name = "Linear_SARSA" if index % 8 < 4 else "F_static"
        self.game = create_game(SimulationConfig(players=self.players), self.seed)
        self.learner = self.game.state.colors[index % self.players]
        # Native search opponents are installed in the engine so their own
        # rollout code uses the correct player object. The acting bot receives
        # only a public-belief copy below, never the real hidden hands.
        if search_opponent:
            preview = self.game
            by_color = {player.color: player for player in preview.state.players}
            by_color = {
                color: REGISTRY.build(self.opponent_name, color) if color != self.learner else player
                for color, player in by_color.items()
            }
            for order in permutations(preview.state.colors):
                candidate_game = Game([by_color[color] for color in order], seed=self.seed)
                if candidate_game.state.colors[index % self.players] == self.learner:
                    self.game = candidate_game
                    break
            else:
                raise AssertionError(f"paired {self.opponent_name} setup changed learner seat")
        self.map_document = map_to_json(self.game.state.board.map)
        self.map_hash = canonical_hash(self.map_document)
        self.initial_stream_hash = canonical_hash(self.game.state.random.getstate())
        if board_hash_variants(self.map_document).intersection(self.config.get("excluded_board_hashes", [])):
            raise ValueError("board overlaps earlier training data")
        self.teacher = ObservationPolicy("F_static", self.map_document, {}, allow_fewer_players=self.players < 4)
        self.learner_encoder = FeatureEncoder(self.map_document, version=self.config.get("feature_version", VERSION),
                                              allow_fewer_players=self.players < 4)
        self.linear_opponent = None
        if self.opponent_name == "Linear_SARSA":
            self.linear_opponent = LinearActionValues.load(self.config["opponent_model"], self.config["opponent_normalizer"])
        elif self.opponent_name not in ("F_static", "F", "Linear_SARSA") and not search_opponent:
            raise ValueError("Unsupported frozen opponent")
        self.opponent_rng = {color: random.Random(self.seed*1000003+i*101+71)
                             for i, color in enumerate(self.game.state.colors)}
        self.stream = gzip.open(self.output / "decisions.jsonl.gz", "wt", encoding="utf-8",
                                compresslevel=self.config.get("trace_compresslevel", 9))
        self.guarded, self.learner_decisions = 0, 0
        self.native_f_decisions, self.native_search_decisions = 0, 0
        return self.advance()

    def execute(self, selected, inputs, policy, *, model_version=None, scores=None, explored=False):
        record = {"action_index": len(self.game.state.action_records), "policy": policy,
                  "inputs": {**inputs, "action": action_to_json(selected)}, "forward_simulation_guard": policy != "F",
                  "model_version": model_version, "raw_scores": scores, "explored": explored}
        self.game.execute(selected)
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")

    def step(self, selected_index, version, scores, explored=False):
        if self.cached is None:
            raise ValueError("worker is not waiting for a learner decision")
        actions, inputs = self.cached
        if not 0 <= selected_index < len(actions) or len(scores) != len(actions):
            raise ValueError("illegal central action or missing scores")
        self.execute(actions[selected_index], inputs, self.config.get("learner_policy", "Linear_SARSA"), model_version=version,
                     scores=scores, explored=explored)
        self.learner_decisions += 1
        self.cached = None
        return self.advance()

    def advance(self):
        while True:
            winner = self.game.winning_color()
            if winner:
                return self.finish("victory")
            if len(self.game.state.action_records) >= self.config.get("max_actions", 2000):
                return self.finish("action_limit")
            if perf_counter()-self.started >= self.config.get("max_seconds", 120):
                return self.finish("time_limit")
            actor = self.game.state.current_color()
            actions = ordered_actions(self.game)
            inputs = {"map_ref": 0, "observation": observation(self.game, actor),
                      "legal_actions": [action_to_json(a) for a in actions]}
            rng_before = self.game.state.random.getstate()
            if actor != self.learner and self.opponent_name == "F":
                from catanzero.regression_search import public_f_choice
                decision_seed = self.seed*1000003+len(self.game.state.action_records)
                selected = public_f_choice(self.game, actions, decision_seed) if len(actions) > 1 else actions[0]
                assert self.game.state.random.getstate() == rng_before
                assert observation(self.game, actor) == inputs["observation"]
                self.native_f_decisions += 1
                self.execute(selected, inputs, "F")
                continue
            if actor != self.learner and self.opponent_name.split(":", 1)[0].upper() in ("AB", "M"):
                # Built-in search bots receive only information visible to the
                # acting player. Their copies and rollouts stay on this sampled
                # belief state and cannot alter the real dice/card stream.
                decision_seed = self.seed * 1_000_003 + len(self.game.state.action_records)
                view = live_belief(self.game, actor, decision_seed)
                selected = view.state.current_player().decide(view, actions)
                if selected not in actions:
                    raise ValueError(f"{self.opponent_name} returned an illegal move")
                if self.game.state.random.getstate() != rng_before:
                    raise ValueError(f"{self.opponent_name} search advanced the actual environment RNG")
                self.native_search_decisions += 1
                self.execute(selected, inputs, self.opponent_name)
                continue
            with forbid_forward_simulation() as attempts:
                if actor == self.learner:
                    features = [self.learner_encoder.encode({**inputs, "action": a}) for a in inputs["legal_actions"]]
                else:
                    if self.linear_opponent is None:
                        selected, scores = self.teacher.choose(inputs)
                    else:
                        scores = [self.linear_opponent.predict(self.teacher.encoder.encode({**inputs, "action": a}))
                                  for a in inputs["legal_actions"]]
                        best = max(scores)
                        selected = self.opponent_rng[actor].choice([i for i, score in enumerate(scores) if score == best])
            assert not attempts and self.game.state.random.getstate() == rng_before
            self.guarded += 1
            if actor == self.learner:
                self.cached = actions, inputs
                own = next(p for p in inputs["observation"]["players"] if p["color"] == actor.value)
                return {"kind": "decision", "game_index": self.index, "seed": self.seed,
                        "learner_color": actor.value, "action_index": len(self.game.state.action_records),
                        "vp": own["actual_vp"],
                        "leader_public_vp": max(p['VICTORY_POINTS'] for p in inputs['observation']['players'] if p['color'] != actor.value),
                        "features": features}
            self.execute(actions[selected], inputs, self.opponent_name, scores=scores)

    def finish(self, reason):
        winner = self.game.winning_color()
        self.stream.close()
        self.stream = None
        own = next(p for p in observation(self.game, self.learner)["players"] if p["color"] == self.learner.value)
        result = {"kind": "finished", "game_index": self.index, "seed": self.seed,
                  "learner_color": self.learner.value, "learner_seat": self.index % self.players,
                  "players": self.players, "initial_stream_hash": self.initial_stream_hash,
                  "learner_policy": self.config.get("learner_policy", "Linear_SARSA"), "initial_map_hash": self.map_hash,
                  "opponent_policy": self.opponent_name,
                  "termination": reason, "winner": winner.value if winner else None,
                  "learner_won": winner == self.learner if winner else None,
                  "actions": len(self.game.state.action_records), "action_index": len(self.game.state.action_records),
                  "vp": own["actual_vp"], "guarded_decisions": self.guarded,
                  "learner_decisions": self.learner_decisions, "forbidden_engine_calls": 0,
                  "native_f_decisions": self.native_f_decisions,
                  "native_search_decisions": self.native_search_decisions,
                  "elapsed_seconds": perf_counter()-self.started}
        assert self.guarded+self.native_f_decisions+self.native_search_decisions == result["actions"]
        atomic_json(self.output / "game.json", state_to_json(self.game))
        atomic_json(self.output / "result.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    env = OnlineEnvironment(json.loads(args.config.read_text()))
    for line in sys.stdin:
        command = json.loads(line)
        if command["op"] == "close":
            if env.stream is not None:
                env.stream.close()
            break
        if command["op"] == "reset":
            response = env.reset(command["index"], command["output"])
        elif command["op"] == "step":
            response = env.step(command["selected_index"], command["model_version"], command["scores"], command["explored"])
        else:
            raise ValueError("unknown environment command")
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
