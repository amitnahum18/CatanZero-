"""Small simulation API; all game rules and legal actions belong to Catanatron."""

from collections import Counter
from dataclasses import asdict, dataclass
import json
from time import perf_counter
from typing import Callable

from catanatron import Color, Game, RandomPlayer
from catanatron.models.enums import ActionRecord
from catanatron.serialization import action_to_json
from catanatron.state_functions import player_key

ENGINE_REVISION = "ecf931181b9a65bb4116a2153fb78c16f1438e00"
COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


class SimulationRandomPlayer(RandomPlayer):
    """Uniform legal-action sampling with stable ordering across Python processes.

    Some engine action generators iterate sets. Canonical ordering prevents
    Python's randomized string hashing from changing a seeded game.
    """

    def decide(self, game, playable_actions):
        ordered = sorted(playable_actions, key=lambda action: json.dumps(action_to_json(action)))
        return game.state.random.choice(ordered)


@dataclass(frozen=True)
class SimulationConfig:
    players: int = 4
    max_turns: int = 1000
    max_actions: int = 100_000

    def __post_init__(self):
        if self.players not in (2, 3, 4):
            raise ValueError("players must be 2, 3, or 4")
        if self.max_turns < 1 or self.max_actions < 1:
            raise ValueError("max_turns and max_actions must be positive")


@dataclass(frozen=True)
class GameResult:
    seed: int
    winner: str | None
    termination: str
    turns: int
    actions: int
    seating_order: list[str]
    victory_points: dict[str, int]
    public_victory_points: dict[str, int]
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def create_game(config: SimulationConfig, seed: int) -> Game:
    """Create a standard base board, with shuffled seats and a target of 10 VP."""
    return Game([SimulationRandomPlayer(color) for color in COLORS[: config.players]], seed=seed)


def run_game(
    config: SimulationConfig,
    seed: int,
    on_action: Callable[[Game, ActionRecord], None] | None = None,
) -> tuple[Game, GameResult]:
    """Run one game sequentially, stopping on victory or either safety limit.

    The callback runs after each action. Return the engine game as well as its
    summary so callers can inspect the board, history, or legal next actions.
    """
    started = perf_counter()
    game = create_game(config, seed)
    while (
        game.winning_color() is None
        and game.state.num_turns < config.max_turns
        and len(game.state.action_records) < config.max_actions
    ):
        record = game.play_tick()
        if on_action is not None:
            on_action(game, record)

    winner = game.winning_color()
    termination = (
        "victory"
        if winner is not None
        else "turn_limit"
        if game.state.num_turns >= config.max_turns
        else "action_limit"
    )
    for observer in game.observers:
        observer.after(game)

    def points(field: str) -> dict[str, int]:
        return {
            color.value: game.state.player_state[f"{player_key(game.state, color)}_{field}"]
            for color in game.state.colors
        }

    result = GameResult(
        seed=seed,
        winner=winner.value if winner is not None else None,
        termination=termination,
        turns=game.state.num_turns,
        actions=len(game.state.action_records),
        seating_order=[color.value for color in game.state.colors],
        victory_points=points("ACTUAL_VICTORY_POINTS"),
        public_victory_points=points("VICTORY_POINTS"),
        elapsed_seconds=round(perf_counter() - started, 6),
    )
    return game, result


def summarize(results: list[GameResult]) -> dict:
    if not results:
        raise ValueError("at least one game result is required")
    colors = results[0].seating_order
    wins = Counter(result.winner for result in results if result.winner is not None)
    completed = sum(result.termination == "victory" for result in results)
    return {
        "games": len(results),
        "completed_games": completed,
        "truncated_games": len(results) - completed,
        "wins": {color: wins[color] for color in colors},
        "termination_counts": dict(Counter(result.termination for result in results)),
        "mean_turns": sum(result.turns for result in results) / len(results),
        "mean_actions": sum(result.actions for result in results) / len(results),
        "simulation_seconds": sum(result.elapsed_seconds for result in results),
    }
