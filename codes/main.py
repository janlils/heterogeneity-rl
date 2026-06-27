"""
Jeden punkt wejścia do głównych zadań projektu.

Przykłady:
    python -m codes.main train-all --quick
    python -m codes.main train --algo ppo --quick
    python -m codes.main train --algo signal_rule --quick
"""

from __future__ import annotations

import argparse
from typing import List

from codes import train_all
from codes.experiment import run_ippo_cli, run_ppo_cli, run_sarsa_cli, run_signal_rule_cli

def _dispatch_train(algo: str, forwarded: List[str]) -> None:
    if algo == "sarsa":
        run_sarsa_cli(forwarded)
        return
    if algo == "ppo":
        run_ppo_cli(forwarded)
        return
    if algo == "ippo":
        run_ippo_cli(forwarded)
        return
    if algo == "signal_rule":
        run_signal_rule_cli(forwarded)
        return
    raise ValueError(f"Nieznany algorytm: {algo}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train-all", help="Uruchom pełny benchmark wszystkich algorytmów.")

    train_parser = sub.add_parser("train", help="Uruchom pojedynczy algorytm.")
    train_parser.add_argument("--algo", choices=["sarsa", "ppo", "ippo", "signal_rule"], required=True)

    args, forwarded = parser.parse_known_args(argv)

    if args.command == "train-all":
        train_all.main(forwarded)
        return
    if args.command == "train":
        _dispatch_train(args.algo, forwarded)
        return

    raise ValueError(f"Nieobsługiwane polecenie: {args.command}")


if __name__ == "__main__":
    main()
