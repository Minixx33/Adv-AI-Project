"""
run_game.py — CLI entry point for the Werewolf game engine (spec §3.1).

Hierarchy / contents:
  main()            — argument parsing → config building → GameRunner / GridSearch
  _load_config()    — merges JSON config file with CLI args
  _validate_args()  — rejects invalid player counts and compositions

Usage (PowerShell examples, spec §5):

  # Default 100-game run with 8 players, Bayesian+Suspicion
  python run_game.py

  # 1000 games with 5 players, MCTS+Suspicion
  python run_game.py --games 1000 --players 5 --suspicion-agent mcts

  # Custom composition via JSON config
  python run_game.py --config configs/exp1.json

  # Grid search for optimal weights
  python run_game.py --grid-search --players 8 --output ./grid_results

  # Reproducible run
  python run_game.py --games 500 --seed 42 --verbose

  # Baseline comparison (no suspicion)
  python run_game.py --games 1000 --suspicion-agent none

CLI arguments (spec §3.1):
  --games           int      Games to run          (default: 100)
  --players         int      5, 8, or 10 only      (default: 8)
  --suspicion-agent str      heuristic/bayesian/mcts/none  (default: bayesian)
  --config          str      Path to JSON config file
  --grid-search     flag     Run grid search instead
  --output          str      Output directory       (default: ./results)
  --seed            int      Random seed
  --verbose         flag     Print game progress
"""

import argparse
import datetime
import json
import os
import sys

# Add Model_V2 dir to path so werewolf/runner/suspicion packages are importable
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from runner.game_runner import GameRunner
from runner.grid_search  import GridSearch
from werewolf.const      import VALID_PLAYER_COUNTS


def main() -> None:
    args = _parse_args()
    config = _load_config(args)
    _validate_args(config)

    if config.get("grid_search"):
        _run_grid_search(config)
    else:
        _run_games(config)


# ------------------------------------------------------------------ #
#  Argument parsing                                                   #
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Suspicion-Aware Werewolf AI — game engine CLI"
    )
    p.add_argument("--games",           type=int,  default=None,
                   help="Number of games to run (default: 100)")
    p.add_argument("--players",         type=int,  default=None,
                   help="Player count: 5, 8, or 10 (default: 8)")
    p.add_argument("--suspicion-agent", type=str,  default=None,
                   dest="suspicion_agent",
                   help="heuristic | bayesian | mcts | none (default: bayesian)")
    p.add_argument("--config",          type=str,  default=None,
                   help="Path to JSON config file (overrides other args)")
    p.add_argument("--grid-search",     action="store_true", default=False,
                   dest="grid_search",
                   help="Run two-stage grid search instead of normal games")
    p.add_argument("--output",          type=str,  default=None,
                   help="Output directory for logs (default: ./results)")
    p.add_argument("--seed",            type=int,  default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--verbose",         action="store_true", default=False,
                   help="Print detailed game progress to console")
    p.add_argument("--run-name",        type=str,  default=None,
                   dest="run_name",
                   help="Name for this run (used in output folder name)")
    return p.parse_args()


# ------------------------------------------------------------------ #
#  Config loading & validation                                         #
# ------------------------------------------------------------------ #

def _load_config(args: argparse.Namespace) -> dict:
    """
    Priority (highest to lowest):
      1. JSON config file (--config)
      2. CLI arguments
      3. Built-in defaults
    """
    defaults = {
        "games":               100,
        "players":             8,
        "suspicion_agent_type": "bayesian",
        "output_dir":          "./results",
        "seed":                None,
        "verbose":             False,
        "grid_search":         False,
        "max_talk_rounds":     5,
        "suspicion_weights": {
            "a1": 0.30, "a2": 0.15, "a3": 0.20, "a4": 0.20, "a5": 0.15,
            "w1": 0.6,  "w2": 0.4,
        },
        "mcts_simulations": 500,
        "mcts_depth":       5,
    }

    config = dict(defaults)

    # Layer 1: JSON config file
    if args.config:
        if not os.path.isfile(args.config):
            _die(f"Config file not found: {args.config}")
        with open(args.config, encoding="utf-8") as f:
            file_cfg = json.load(f)
        config.update(file_cfg)

    # Layer 2: CLI arguments (only if explicitly set, i.e. not None)
    if args.games is not None:
        config["games"] = args.games
    if args.players is not None:
        config["players"] = args.players
    if args.suspicion_agent is not None:
        config["suspicion_agent_type"] = args.suspicion_agent
    if args.output is not None:
        config["output_dir"] = args.output
    if args.seed is not None:
        config["seed"] = args.seed
    if args.verbose:
        config["verbose"] = True
    if args.grid_search:
        config["grid_search"] = True
    if args.run_name is not None:
        config["run_name"] = args.run_name

    # Layer 3: interactive prompts for the three key parameters
    # (only when running as __main__ and the value was not set by CLI or config)
    if _running_interactively():
        config = _prompt_interactive(config, args)

    return config


def _running_interactively() -> bool:
    """True when run as a script (not imported) with no relevant CLI flags set."""
    import __main__
    return hasattr(__main__, "__file__") and __main__.__file__ == __file__


def _prompt_interactive(config: dict, args: argparse.Namespace) -> dict:
    print("=== Werewolf Game Engine ===\n")

    # Run name
    if not config.get("run_name"):
        raw = input("Your name (for output folder): ").strip()
        config["run_name"] = raw if raw else "run"

    # Number of games
    if args.games is None:
        raw = input(f"Number of games to run [default: {config['games']}]: ").strip()
        if raw:
            try:
                config["games"] = int(raw)
            except ValueError:
                _die(f"Invalid number of games: '{raw}'")

    # Number of players
    if args.players is None:
        raw = input(f"Number of players (5 / 8 / 10) [default: {config['players']}]: ").strip()
        if raw:
            try:
                config["players"] = int(raw)
            except ValueError:
                _die(f"Invalid player count: '{raw}'")

    # Agent composition (only when not already set by --config file)
    if config.get("agent_composition") is None:
        comp = _prompt_composition(config["players"])
        if comp is not None:
            config["agent_composition"] = comp
            # Infer suspicion_agent_type from whatever *_with_suspicion keys exist
            _SUSP_KEY_TO_TYPE = {
                "bayesian_with_suspicion":  "bayesian",
                "heuristic_with_suspicion": "heuristic",
                "mcts_with_suspicion":      "mcts",
            }
            susp_types = [v for k, v in _SUSP_KEY_TO_TYPE.items() if comp.get(k, 0) > 0]
            config["suspicion_agent_type"] = susp_types[0] if susp_types else "none"
        else:
            # User skipped composition — fall back to old single-type prompt
            if args.suspicion_agent is None:
                options = "heuristic / bayesian / mcts / none"
                raw = input(
                    f"Suspicion agent ({options})"
                    f" [default: {config['suspicion_agent_type']}]: "
                ).strip().lower()
                if raw:
                    config["suspicion_agent_type"] = raw

    print()
    return config


# ── Agent types available in the interactive composition prompt ──────────────
_PROMPT_AGENT_TYPES = [
    # (display label,  config key,   supports suspicion module?)
    ("Random",         "random",         False),
    ("Heuristic",      "heuristic",      True),
    ("Bayesian",       "bayesian",       True),
    ("MCTS",           "mcts",           True),
    ("Logic",          "logic_based",    False),
]


def _prompt_composition(n_players: int):
    """
    Ask the user how many of each agent type they want.
    For types that support the Suspicion module, also asks how many should
    have it enabled.

    - Counts are capped to remaining slots; the user is warned when capped.
    - Any unfilled slots are automatically padded with Random agents.
    - Returns a composition dict, or None if the user skipped everything.
    """
    print(f"\nAgent composition  ({n_players} players total)")
    print("  Press Enter to keep 0.  Remaining slots are filled with Random.")

    composition: dict = {}
    total = 0
    any_entered = False

    for display, key, has_suspicion in _PROMPT_AGENT_TYPES:
        remaining = n_players - total
        if remaining <= 0:
            break

        raw = input(f"  {display:<12} [0, {remaining} remaining]: ").strip()
        if not raw:
            continue

        try:
            count = int(raw)
        except ValueError:
            print(f"    (invalid - treating as 0)")
            continue

        if count <= 0:
            continue

        # Cap to available slots
        if count > remaining:
            print(f"    Only {remaining} slot(s) left - capping to {remaining}.")
            count = remaining

        any_entered = True
        total += count

        if has_suspicion:
            raw_s = input(
                f"    -> of those, with Suspicion module? (0-{count}) [0]: "
            ).strip()
            try:
                n_susp = int(raw_s) if raw_s else 0
            except ValueError:
                n_susp = 0
            if n_susp < 0 or n_susp > count:
                print(f"    Invalid - clamping to range 0-{count}.")
                n_susp = max(0, min(count, n_susp))
            n_base = count - n_susp
            if n_susp > 0:
                composition[f"{key}_with_suspicion"] = n_susp
            if n_base > 0:
                composition[key] = n_base
        else:
            composition[key] = count

    if not any_entered:
        return None   # user skipped everything - fall back to defaults

    # Auto-fill leftover slots with Random agents
    leftover = n_players - total
    if leftover > 0:
        print(f"  {leftover} slot(s) unfilled - adding {leftover}x Random.")
        composition["random"] = composition.get("random", 0) + leftover

    # Show summary
    from runner.game_runner import _AGENT_DISPLAY_NAMES
    summary = "  |  ".join(
        f"{cnt}x {_AGENT_DISPLAY_NAMES.get(t, t)}"
        for t, cnt in composition.items() if cnt > 0
    )
    print(f"\n  Composition set:  {summary}\n")
    return composition


def _validate_args(config: dict) -> None:
    n = config.get("players", 8)
    if n not in VALID_PLAYER_COUNTS:
        _die(
            f"Invalid player count: {n}. "
            f"Must be one of {VALID_PLAYER_COUNTS}."
        )

    susp = config.get("suspicion_agent_type", "bayesian")
    valid_susp = {"heuristic", "bayesian", "mcts", "none"}
    if susp not in valid_susp:
        _die(
            f"Invalid --suspicion-agent value: '{susp}'. "
            f"Must be one of {sorted(valid_susp)}."
        )

    comp = config.get("agent_composition")
    if comp is not None:
        total = sum(comp.values())
        if total != n:
            _die(
                f"agent_composition sums to {total} but --players={n}. "
                f"Counts: {comp}"
            )

    if config.get("games", 1) < 1:
        _die("--games must be a positive integer.")


# ------------------------------------------------------------------ #
#  Runners                                                            #
# ------------------------------------------------------------------ #

def _apply_run_name(config: dict) -> None:
    """Replace output_dir with {base}/{run_name}_{YYYY-MM-DD_HH-MM-SS}."""
    base      = config.get("output_dir", "./results")
    run_name  = config.get("run_name", "run")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config["output_dir"] = os.path.join(base, f"{run_name}_{timestamp}")


def _run_games(config: dict) -> None:
    _apply_run_name(config)
    n_games    = config["games"]
    output_dir = config["output_dir"]
    susp_type  = config.get("suspicion_agent_type", "bayesian")
    n_players  = config["players"]

    print(
        f"Running {n_games} game(s) | {n_players} players | "
        f"suspicion: {susp_type} | output: {output_dir}"
    )

    runner = GameRunner(config)
    records, summary = runner.run_games(n_games)

    village_wins = sum(1 for r in summary if r["winner"] == "VILLAGE")
    print(
        f"\nDone. Village win rate: {village_wins}/{n_games} "
        f"= {village_wins/max(n_games,1):.3f}"
    )
    print(f"Results saved -> {output_dir}/")


def _run_grid_search(config: dict) -> None:
    _apply_run_name(config)
    n_games = config.get("games", 2000)
    output_dir = config["output_dir"]

    print(
        f"Grid search mode | {n_games} games/config | "
        f"output: {output_dir}"
    )

    gs = GridSearch(
        base_config=config,
        n_games=n_games,
        n_samples=config.get("grid_samples", 200),
        seed=config.get("seed"),
    )
    gs.run(output_dir)


# ------------------------------------------------------------------ #
#  Helper                                                             #
# ------------------------------------------------------------------ #

def _die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
