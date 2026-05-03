#!/usr/bin/env python
"""run_experiments.py — CLI entry point for the Werewolf AI comparison study.

This script runs a batch of self-contained Werewolf games, writes per-agent
per-game statistics to a CSV file, then prints a win-rate summary table.

Usage
-----
::

    # Run 1 000 games with default settings
    python run_experiments.py

    # Run 10 000 games
    python run_experiments.py -n 10000

    # Reproducible run with a fixed seed
    python run_experiments.py -n 100 --seed 42

    # Print a summary table from an existing CSV without running new games
    python run_experiments.py --summarise

    # Custom output path
    python run_experiments.py -n 5000 --output results/experiment_01.csv

Arguments
---------
-n / --n-games  : int  (default 1000)
    Number of independent games to simulate.
--seed          : int  (optional)
    Seed for Python's ``random`` module.  Omit for a non-deterministic run.
--output        : str  (default 'results/results.csv')
    Path for the CSV output file.
--summarise     : flag
    If set, print the win-rate table from ``--output`` and exit immediately
    without running new games.

Output
------
Each run produces a CSV with one row per agent per game, then prints a
win-rate summary table grouped by agent type and role.  See
``werewolf.runner.GameRunner`` for the full column specification.
"""
import argparse
import os
import sys

# Make sure the project root is on sys.path when invoked directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from werewolf.runner import GameRunner


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI.

    Returns:
        A configured ``ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        prog="run_experiments",
        description="Run Werewolf AI comparison experiments and log results to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-n", "--n-games",
        type=int,
        default=1000,
        metavar="N",
        help="Number of games to simulate (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed for reproducibility (omit for non-deterministic).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/results.csv",
        metavar="PATH",
        help="Output CSV file path (default: results/results.csv).",
    )
    parser.add_argument(
        "--summarise",
        action="store_true",
        help="Print win-rate table from the existing CSV and exit.",
    )
    return parser


def main() -> None:
    """Parse arguments and run the experiment or summary as requested."""
    args = _build_parser().parse_args()

    if args.summarise:
        GameRunner.summarise(args.output)
        return

    print(
        f"Running {args.n_games} games "
        f"(seed={args.seed}) -> {args.output}"
    )
    runner = GameRunner(
        output_path=args.output,
        n_games=100000,
        seed=args.seed,
    )
    runner.run_experiments()
    GameRunner.summarise(args.output)


if __name__ == "__main__":
    main()
