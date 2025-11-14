"""Utility script to inspect and render a graph for a given conjecture.

Usage example::

    python draw_graph.py --id 185 --graph6 "??C?G"

If the conjecture identifier refers to a row in ``benchmark.csv`` the script
uses the official invariants; otherwise it looks up a ``conj_<id>`` scoring
function inside ``scores_function.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import networkx as nx
from matplotlib import pyplot as plt

from helpers.utility import (
    check_subclasses,
    evaluate_conjecture,
    load_conjectures,
    Conjecture,
)
from helpers import scores_function


def load_conjecture(identifier: str, csv_path: Path) -> Optional[object]:
    """Return the conjecture row (or custom score entry) for ``identifier``."""

    for row in load_conjectures(csv_path, as_dataclasses=True):
        if row.identifier == identifier:
            return row
    score_fn = getattr(scores_function, f"conj_{identifier}", None)
    if score_fn is None:
        return None
    return {
        "ID": identifier,
        "conjecture": f"Custom scoring function conj_{identifier}",
        "subclass": "",
        "score_function": score_fn,
    }


def evaluate_graph(
    graph: nx.Graph,
    identifier: str,
    csv_path: Path,
    min_size: int,
    max_size: int,
) -> Optional[float]:
    """Return the score for ``graph`` if the conjecture can be evaluated."""

    conjecture = load_conjecture(identifier, csv_path)
    if conjecture is None:
        raise SystemExit(f"Conjecture ID '{identifier}' not found")

    score_fn = conjecture.get("score_function") if isinstance(conjecture, dict) else None
    if score_fn is not None:
        if graph.number_of_nodes() < min_size or graph.number_of_nodes() > max_size:
            return None
        subclass = conjecture.get("subclass")
        if subclass and not check_subclasses(graph, subclass):
            return None
        return score_fn(graph, min_size, max_size)

    if isinstance(conjecture, Conjecture):
        result = evaluate_conjecture(
            graph,
            conjecture,
            min_size=min_size,
            max_size=max_size,
            ensure_subclass=True,
            margin=0.0,
        )
        if result is None:
            return None
        return result.score

    raise SystemExit(
        "Unable to evaluate this conjecture because it is neither a benchmark entry "
        "nor a custom scoring function."
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Draw a graph and evaluate its score")
    parser.add_argument("--id", required=True, help="Conjecture identifier")
    parser.add_argument("--graph6", required=True, help="Graph encoded in graph6 format")
    parser.add_argument("--min-size", type=int, default=1, help="Minimum order for eligibility")
    parser.add_argument("--max-size", type=int, default=100, help="Maximum order for eligibility")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/benchmark.csv"),
        help="Path to benchmark.csv",
    )
    args = parser.parse_args(argv)

    graph = nx.from_graph6_bytes(args.graph6.encode("ascii"))
    score = evaluate_graph(graph, args.id, args.csv, args.min_size, args.max_size)

    if score is None:
        print("Graph does not satisfy eligibility conditions.")
    else:
        status = "COUNTEREXAMPLE" if score < 0 else "NOT A COUNTEREXAMPLE"
        print(f"Score = {score:.6f} -> {status}")

    nx.draw(graph, with_labels=True, node_color="skyblue", edge_color="grey")
    plt.title(f"Conjecture {args.id} ({status if score is not None else 'INELIGIBLE'})")
    plt.show()


if __name__ == "__main__":
    # Exemple d’appel personnalisé
    custom_args = [
        "--id", "4447",
        "--graph6", "QrXc{xbWrERfWpfMrfSxsWp][yG",
        "--min-size", "6",
        "--max-size", "30",
        "--csv", "data/benchmark.csv",
    ]
    main(custom_args)
