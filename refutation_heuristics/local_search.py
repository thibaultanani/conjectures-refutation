"""Simplified local-search runner tailored for quick experimentation.

The module provides a lightweight hill-climbing routine that works directly with
rows loaded from ``benchmark.csv``.  It purposefully keeps the surface API small
so it can be used from notebooks or ad-hoc scripts while still offering
reasonable structure, type hints, and documentation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import numpy as np
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import networkx as nx
from matplotlib import pyplot as plt

_BLAS_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

for _var in _BLAS_ENV_VARS:
    os.environ.setdefault(_var, "1")

from helpers.utility import (
    check_subclasses,
    evaluation,
    generate_init_graph,
    get_invariants,
    load_conjectures,
    mutation_add_edge,
    mutation_add_vertex,
    mutation_remove_edge,
    mutation_remove_vertex,
    mutation_subdivision,
    mutation_contraction,
    mutation_replace_vertex_by_path,
    mutation_replace_vertex_by_star,
    mutation_replace_vertex_by_clique,
    mutation_replace_vertex_by_polyhedral,
    mutation_bipartition_neighborhood,
    ConjectureResult,
)
from helpers import invariants, scores_function

ConjectureMapping = Dict[str, Any]
CacheValue = Tuple[Optional[float], Optional[float], float]


@dataclass(slots=True)
class SearchParameters:
    """Tunable parameters controlling the hill-climbing search."""

    min_size: int = 6
    max_size: int = 30
    neighbor_count: int = 20
    max_mutations: int = 3
    time_limit: float = 60.0
    stagnation_limit: int = 100
    margin: float = 1e-3
    mutation_names: Tuple[str, ...] = tuple()
    seed: Optional[int] = None
    verbose: bool = True
    cache_size_limit: Optional[int] = None


@dataclass(slots=True)
class GraphEvaluation:
    """Container for the evaluation of a single graph."""

    graph: nx.Graph
    graph6: str
    score: float
    details: Optional[ConjectureResult] = None

    def is_counterexample(self, margin: float) -> bool:
        if self.details is not None:
            return self.details.score < -margin
        return self.score < -margin


@dataclass(slots=True)
class SearchOutcome:
    """Summary of a full hill-climbing run."""

    conjecture_id: str
    found_counterexample: bool
    best_evaluation: GraphEvaluation
    elapsed: float
    evaluations_total: int
    evaluations_eligible: int
    resets: int
    subclass_name: Optional[str] = None


MutationFunction = Callable[[nx.Graph], nx.Graph]


MUTATION_REGISTRY: Dict[str, MutationFunction] = {
    "add_edge": mutation_add_edge,
    "remove_edge": mutation_remove_edge,
    "add_vertex": mutation_add_vertex,
    "remove_vertex": mutation_remove_vertex,
    "subdivision": mutation_subdivision,
    "contraction": mutation_contraction,
    "replace_vertex_by_path": mutation_replace_vertex_by_path,
    "replace_vertex_by_star": mutation_replace_vertex_by_star,
    "replace_vertex_by_clique": mutation_replace_vertex_by_clique,
    "replace_vertex_by_polyhedral": mutation_replace_vertex_by_polyhedral,
    "bipartition_neighborhood": mutation_bipartition_neighborhood,
}

DEFAULT_MUTATION_NAMES: Tuple[str, ...] = tuple(MUTATION_REGISTRY.keys())


def resolve_mutation_functions(names: Sequence[str]) -> Tuple[MutationFunction, ...]:
    """Resolve mutation operator names to callables."""

    functions: list[MutationFunction] = []
    for name in names:
        try:
            functions.append(MUTATION_REGISTRY[name])
        except KeyError as exc:
            raise ValueError(f"Unknown mutation '{name}'") from exc
    if not functions:
        raise ValueError("At least one mutation must be specified.")
    return tuple(functions)


CSV_HEADER = [
    "identifier",
    "found_counterexample",
    "graph6",
    "score",
    "elapsed",
    "evaluations_total",
    "evaluations_eligible",
    "resets",
]


def prepare_output_directory(base: Path = Path("out")) -> Path:
    """Create (if necessary) the output directory for the current run."""

    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = 0
    while True:
        dirname = timestamp if suffix == 0 else f"{timestamp}_{suffix:02d}"
        candidate = base / dirname
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        suffix += 1


def _ensure_results_csv(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(CSV_HEADER)


def append_result_csv(path: Path, outcome: SearchOutcome) -> None:
    _ensure_results_csv(path)
    evaluation = outcome.best_evaluation
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                outcome.conjecture_id,
                outcome.found_counterexample,
                evaluation.graph6,
                f"{evaluation.score:.6f}",
                f"{outcome.elapsed:.3f}",
                outcome.evaluations_total,
                outcome.evaluations_eligible,
                outcome.resets,
            ]
        )


def write_summary_txt(
    path: Path,
    outcomes: Sequence[SearchOutcome],
    params: SearchParameters,
    cpus: int,
    base_seed: Optional[int],
    context_seeds: Sequence[Tuple[str, Optional[int]]],
    total_contexts: int,
) -> None:
    lines: list[str] = []
    lines.append("Local Search Experiment Summary")
    lines.append("=" * 32)
    lines.append(f"Timestamp      : {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Time limit     : {params.time_limit:.1f}s")
    lines.append(f"Graph order    : [{params.min_size}, {params.max_size}]")
    lines.append(f"Neighbours     : {params.neighbor_count}")
    lines.append(f"Max mutations  : {params.max_mutations}")
    lines.append(f"Stagnation cap : {params.stagnation_limit}")
    lines.append(f"Margin         : {params.margin:.1e}")
    lines.append(f"CPUs           : {cpus}")
    seed_label = base_seed if base_seed is not None else "random"
    lines.append(f"Seed           : {seed_label}")
    if context_seeds:
        lines.append(
            "Context seeds   : "
            + ", ".join(
                f"{identifier}:{seed if seed is not None else 'random'}"
                for identifier, seed in context_seeds
            )
        )
    mutation_names = params.mutation_names or DEFAULT_MUTATION_NAMES
    lines.append(f"Mutations      : {', '.join(mutation_names)}")
    lines.append("")

    success_count = sum(outcome.found_counterexample for outcome in outcomes)
    lines.append(
        f"Counterexamples found: {success_count}/{total_contexts}"
        + (f" (processed {len(outcomes)}/{total_contexts})" if total_contexts else "")
    )
    lines.append("")

    for outcome in outcomes:
        evaluation = outcome.best_evaluation
        lines.append(
            f"[ID {outcome.conjecture_id}] {'SUCCESS' if outcome.found_counterexample else 'FAIL'}"
        )
        lines.append(f"  Score          : {evaluation.score:.6f}")
        lines.append(f"  Time (s)       : {outcome.elapsed:.3f}")
        lines.append(f"  Evaluated      : {outcome.evaluations_total} graphs total")
        lines.append(
            f"  Eligible       : {outcome.evaluations_eligible} graphs within subclass"
        )
        lines.append(f"  Resets         : {outcome.resets}")
        if outcome.subclass_name:
            lines.append(f"  Subclass       : {outcome.subclass_name}")
        lines.append(f"  Graph6         : {evaluation.graph6}")
        lines.append("")

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).strip() + "\n")


def _maybe_log_outcome(outcome: SearchOutcome, verbose: bool) -> None:
    if not verbose:
        return
    status = "FOUND" if outcome.found_counterexample else "NONE"
    evaluation = outcome.best_evaluation
    print(
        f"ID={outcome.conjecture_id} status={status} total={outcome.evaluations_total} "
        f"eligible={outcome.evaluations_eligible} resets={outcome.resets} "
        f"score={evaluation.score:.6f} time={outcome.elapsed:.3f}s graph6={evaluation.graph6}"
    )


class GraphScoreCache:
    """Least-recently-used cache storing invariants and scores per graph."""

    def __init__(self, max_size: Optional[int] = None) -> None:
        self._data: "OrderedDict[str, CacheValue]" = OrderedDict()
        self._limit = max_size if max_size and max_size > 0 else None

    def get(self, key: str) -> Optional[CacheValue]:
        value = self._data.get(key)
        if value is not None:
            self._data.move_to_end(key)
        return value

    def set(self, key: str, value: CacheValue) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if self._limit is not None:
            while len(self._data) > self._limit:
                self._data.popitem(last=False)


def _derive_seed(identifier: str, base_seed: int) -> int:
    """Derive a stable 32-bit seed from ``identifier`` and ``base_seed``."""

    payload = f"{identifier}:{base_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _graph_to_key(graph: nx.Graph) -> str:
    """Return the canonical graph6 representation for ``graph``."""

    return nx.to_graph6_bytes(graph, header=False).decode("ascii").strip()


def _sample_valid_graph(
    conjecture: ConjectureMapping,
    min_size: int,
    max_size: int,
) -> Tuple[nx.Graph, str, CacheValue]:
    """Draw a random graph that satisfies the conjecture constraints."""

    subclass = conjecture.get("subclass") or None
    score_fn: Optional[Callable[[nx.Graph, int, int], Optional[float]]] = conjecture.get("score_function")

    while True:
        candidate = generate_init_graph(min_size, max_size)
        if subclass and not check_subclasses(candidate, subclass):
            continue
        if score_fn is not None:
            score = score_fn(candidate.copy(), min_size, max_size)
            if score is None:
                continue
            return candidate, _graph_to_key(candidate), (None, None, float(score))
        invariants = get_invariants(candidate, conjecture, min_size, max_size)
        if invariants is None:
            continue
        x_value, y_value = invariants
        score = evaluation(x_value, y_value, conjecture)
        return candidate, _graph_to_key(candidate), (x_value, y_value, score)


def _evaluate_graph(
    graph: nx.Graph,
    conjecture: ConjectureMapping,
    min_size: int,
    max_size: int,
) -> Optional[CacheValue]:
    """Evaluate ``graph`` for ``conjecture`` returning cached tuple when eligible."""

    score_fn: Optional[Callable[[nx.Graph, int, int], Optional[float]]] = conjecture.get("score_function")
    subclass = conjecture.get("subclass") or None
    if subclass and not check_subclasses(graph, subclass):
        return None

    if score_fn is not None:
        score = score_fn(graph.copy(), min_size, max_size)
        if score is None:
            return None
        return None, None, float(score)
    invariants = get_invariants(graph, conjecture, min_size, max_size)
    if invariants is None:
        return None
    x_value, y_value = invariants
    return x_value, y_value, evaluation(x_value, y_value, conjecture)


def mutate_graph(
    graph: nx.Graph,
    max_mutations: int,
    mutations: Sequence[Callable[[nx.Graph], nx.Graph]],
) -> nx.Graph:
    """Apply between one and ``max_mutations`` random mutations to ``graph``."""

    mutated = graph
    for _ in range(random.randint(1, max_mutations)):
        mutation = random.choice(mutations)
        mutated = mutation(mutated)
    return mutated


@dataclass(slots=True)
class SearchConfig:
    """Configuration parameters for the lightweight hill-climbing routine."""

    neighbour_count: int = 20
    min_size: int = 6
    max_size: int = 30
    max_mutations: int = 3
    time_limit: float = 60.0
    stagnation_limit: int = 10
    margin: float = 1e-3
    cache_size_limit: Optional[int] = None
    mutation_names: Optional[Tuple[str, ...]] = None
    verbose: bool = False
    seed: Optional[int] = None


@dataclass(slots=True)
class SearchResult:
    """Outcome of a hill-climbing run against a single conjecture."""

    has_counterexample: bool
    counterexample_g6: str
    score: float
    x_value: Optional[float]
    y_value: Optional[float]
    time: float
    total_evaluated: int
    total_rejected: int
    reset: int


def search_hill_climbing(conjecture: ConjectureMapping, config: SearchConfig) -> SearchResult:
    """Run a hill-climbing search for ``conjecture`` using ``config`` parameters."""

    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        invariants.set_cbc_solver_seed(config.seed)
    else:
        invariants.set_cbc_solver_seed(None)

    start_time = time.time()
    total_evaluated = 0
    total_rejected = 0
    cache = GraphScoreCache(config.cache_size_limit)

    mutation_names = config.mutation_names or DEFAULT_MUTATION_NAMES
    mutation_functions = resolve_mutation_functions(mutation_names)

    current_graph, current_key, (x_value, y_value, current_score) = _sample_valid_graph(
        conjecture,
        config.min_size,
        config.max_size,
    )
    cache.set(current_key, (x_value, y_value, current_score))
    total_evaluated += 1

    best_score = current_score
    best_invariants: Tuple[Optional[float], Optional[float]] = (x_value, y_value)
    best_graph_key = current_key

    iteration = 0
    no_improve_counter = 0
    reset_count = 0

    while time.time() - start_time < config.time_limit:
        iteration += 1
        neighbours = []

        for _ in range(config.neighbour_count):
            neighbour_graph = mutate_graph(current_graph, config.max_mutations, mutation_functions)
            total_evaluated += 1
            neighbour_key = _graph_to_key(neighbour_graph)

            cached_invariant = cache.get(neighbour_key)
            if cached_invariant is not None:
                x_neighbour, y_neighbour, neighbour_score = cached_invariant
            else:
                evaluated = _evaluate_graph(
                    neighbour_graph,
                    conjecture,
                    config.min_size,
                    config.max_size,
                )
                if evaluated is None:
                    total_rejected += 1
                    continue
                x_neighbour, y_neighbour, neighbour_score = evaluated
                cache.set(neighbour_key, (x_neighbour, y_neighbour, neighbour_score))

            if neighbour_score + config.margin < 0:
                elapsed = round(time.time() - start_time, 3)
                return SearchResult(
                    has_counterexample=True,
                    counterexample_g6=neighbour_key,
                    score=neighbour_score,
                    x_value=x_neighbour,
                    y_value=y_neighbour,
                    time=elapsed,
                    total_evaluated=total_evaluated,
                    total_rejected=total_rejected,
                    reset=reset_count,
                )

            neighbours.append((neighbour_graph, neighbour_key, neighbour_score, (x_neighbour, y_neighbour)))

        if not neighbours:
            no_improve_counter += 1
            continue

        best_neighbour_graph, best_neighbour_key, best_neighbour_score, best_neighbour_invariants = min(
            neighbours,
            key=lambda item: item[2],
        )

        if best_neighbour_score < current_score:
            current_graph = best_neighbour_graph
            current_score = best_neighbour_score
            best_score = best_neighbour_score
            best_invariants = best_neighbour_invariants
            best_graph_key = best_neighbour_key
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            if best_neighbour_score == current_score:
                current_graph = best_neighbour_graph
                current_score = best_neighbour_score

        if no_improve_counter >= config.stagnation_limit:
            reset_count += 1
            current_graph, current_key, (x_value, y_value, current_score) = _sample_valid_graph(
                conjecture,
                config.min_size,
                config.max_size,
            )
            cache.set(current_key, (x_value, y_value, current_score))
            no_improve_counter = 0

    elapsed = round(time.time() - start_time, 3)
    x_best, y_best = best_invariants
    return SearchResult(
        has_counterexample=False,
        counterexample_g6=best_graph_key,
        score=best_score,
        x_value=x_best,
        y_value=y_best,
        time=elapsed,
        total_evaluated=total_evaluated,
        total_rejected=total_rejected,
        reset=reset_count,
    )


def process_conjecture(
    conjecture: ConjectureMapping,
    config: SearchConfig,
) -> SearchResult:
    """Execute the search for a single conjecture and return the raw result."""

    if config.seed is not None:
        identifier = str(conjecture.get("ID", ""))
        derived_seed = _derive_seed(identifier, config.seed)
        effective_config = replace(config, seed=derived_seed)
    else:
        effective_config = config

    result = search_hill_climbing(conjecture, effective_config)
    return result


def process_all_conjectures(
    conjectures: Sequence[ConjectureMapping],
    output_dir: Path,
    config: SearchConfig,
    *,
    show_plot: bool = False,
    cpus: int = 1,
    context_seed_pairs: Optional[Sequence[Tuple[str, Optional[int]]]] = None,
) -> None:
    """Run the search on ``conjectures`` in parallel and persist the outcomes."""

    if not conjectures:
        raise ValueError("No conjectures provided.")

    run_dir = prepare_output_directory(output_dir)
    results_csv_path = run_dir / "results.csv"
    summary_path = run_dir / "summary.txt"
    base_params = SearchParameters(
        min_size=config.min_size,
        max_size=config.max_size,
        neighbor_count=config.neighbour_count,
        max_mutations=config.max_mutations,
        time_limit=config.time_limit,
        stagnation_limit=config.stagnation_limit,
        margin=config.margin,
        mutation_names=config.mutation_names or DEFAULT_MUTATION_NAMES,
        seed=config.seed,
        verbose=config.verbose,
        cache_size_limit=config.cache_size_limit,
    )

    _ensure_results_csv(results_csv_path)

    context_ids = [str(conj["ID"]) for conj in conjectures]
    effective_cpus = 1
    if cpus > 1:
        effective_cpus = min(cpus, cpu_count())
    elif cpus < 0:
        effective_cpus = cpu_count()

    if context_seed_pairs is None:
        context_seed_pairs = [(cid, None) for cid in context_ids]

    outcomes_by_id: Dict[str, SearchOutcome] = {}

    def _handle_completion(conjecture: ConjectureMapping, search_result: SearchResult) -> None:
        identifier = str(conjecture["ID"])
        outcome = _result_to_outcome(conjecture, search_result)
        outcomes_by_id[identifier] = outcome
        append_result_csv(results_csv_path, outcome)
        ordered_outcomes = [outcomes_by_id[cid] for cid in context_ids if cid in outcomes_by_id]
        write_summary_txt(
            summary_path,
            ordered_outcomes,
            base_params,
            effective_cpus,
            config.seed,
            context_seed_pairs,
            len(conjectures),
        )

        if config.verbose:
            status = "Counterexample" if search_result.has_counterexample else "No counterexample"
            print(
                f"[{identifier}] {status} - score={search_result.score:.6f}, "
                f"evaluated={search_result.total_evaluated}, resets={search_result.reset}, "
                f"time={search_result.time:.3f}s, graph6={search_result.counterexample_g6}",
            )

        if show_plot and effective_cpus == 1:
            graph = nx.from_graph6_bytes(search_result.counterexample_g6.encode("ascii"))
            plt.title(identifier)
            if search_result.has_counterexample:
                nx.draw(graph, with_labels=True)
            else:
                nx.draw(graph, node_color="orange", with_labels=True)
            plt.show()
            plt.close()

    if effective_cpus == 1:
        for conjecture in conjectures:
            search_result = process_conjecture(conjecture, config)
            _handle_completion(conjecture, search_result)
        print(f"Results written to {run_dir}")
        return

    payloads = [(conjecture, config) for conjecture in conjectures]

    with Pool(processes=effective_cpus) as pool:
        for conjecture, search_result in pool.imap_unordered(_worker_entry, payloads, chunksize=1):
            _handle_completion(conjecture, search_result)

    print(f"Results written to {run_dir}")


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the lightweight local-search runner."""

    parser = argparse.ArgumentParser(description="Lightweight local search for graph conjectures")
    parser.add_argument("ids", nargs="*", help="Conjecture identifiers to challenge")
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=Path("../data/identifiers.txt"),
        help="Text file listing conjecture identifiers (one per line)",
    )
    parser.add_argument("--input", type=Path, default=Path("../data/benchmark.csv"), help="CSV file holding conjectures")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out"),
        help="Base directory where run artefacts will be written",
    )
    parser.add_argument("--min-size", type=int, default=6, help="Minimum number of vertices")
    parser.add_argument("--max-size", type=int, default=30, help="Maximum number of vertices")
    parser.add_argument("--time-limit", type=float, default=60.0, help="Wall-clock time limit per conjecture")
    parser.add_argument("--neighbors", type=int, default=20, help="Number of neighbours explored per iteration")
    parser.add_argument("--max-mutations", type=int, default=3, help="Maximum number of mutations per neighbour")
    parser.add_argument("--stagnation", type=int, default=10, help="Iterations without improvement before a reset")
    parser.add_argument("--margin", type=float, default=1e-3, help="Score margin required to accept a counterexample")
    parser.add_argument("--cache-limit", type=int, default=None, help="Maximum number of cached evaluations (default: unlimited)")
    parser.add_argument(
        "--mutations",
        nargs="+",
        default=None,
        help="Mutation operators to use (available: " + ", ".join(DEFAULT_MUTATION_NAMES) + ")",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--cpus", type=int, default=1, help="Number of worker processes (<=1 disables multiprocessing)")
    parser.add_argument("--verbose", action="store_true", help="Print per-conjecture summaries")
    parser.add_argument("--plots", action="store_true", help="Display NetworkX visualisations for each outcome")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments()

    if not args.input.exists():
        raise SystemExit(f"Conjecture CSV not found: {args.input}")

    dataset = load_conjectures(str(args.input))
    if not dataset:
        raise SystemExit(f"No conjectures were loaded from {args.input}")

    by_identifier = {row.get("ID"): row for row in dataset if row.get("ID") is not None}
    requested_ids: list[str] = [str(identifier) for identifier in args.ids]

    if args.ids_file is not None:
        if not args.ids_file.exists():
            raise SystemExit(f"Identifier file not found: {args.ids_file}")
        with args.ids_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = line.strip()
                if value:
                    requested_ids.append(value)

    if not requested_ids:
        raise SystemExit("No conjecture identifiers provided (CLI ids or --ids-file).")

    seen: set[str] = set()
    unique_ids: list[str] = []
    for identifier in requested_ids:
        if identifier not in seen:
            seen.add(identifier)
            unique_ids.append(identifier)

    selected: list[ConjectureMapping] = []
    missing: list[str] = []
    for identifier in unique_ids:
        match = by_identifier.get(identifier)
        if match is not None:
            entry = dict(match)
            entry.setdefault("subclass", "")
            selected.append(entry)
            continue
        score_fn = getattr(scores_function, f"conj_{identifier}", None)
        if score_fn is None:
            missing.append(identifier)
            continue
        selected.append(
            {
                "ID": identifier,
                "conjecture": f"Custom scoring function conj_{identifier}",
                "subclass": "",
                "score_function": score_fn,
            }
        )

    if missing:
        missing_ids = ", ".join(sorted(missing))
        raise SystemExit(
            f"Unable to resolve conjecture ID(s): {missing_ids}. "
            f"Check that they exist in {args.input}."
        )

    cache_limit = args.cache_limit if args.cache_limit is None or args.cache_limit > 0 else None
    mutation_names = tuple(args.mutations) if args.mutations else None
    config = SearchConfig(
        neighbour_count=args.neighbors,
        min_size=args.min_size,
        max_size=args.max_size,
        max_mutations=args.max_mutations,
        time_limit=args.time_limit,
        stagnation_limit=args.stagnation,
        margin=args.margin,
        cache_size_limit=cache_limit,
        mutation_names=mutation_names,
        verbose=args.verbose,
        seed=args.seed,
    )

    output_path: Path = args.output
    output_path.mkdir(parents=True, exist_ok=True)

    if config.seed is not None:
        context_seed_pairs = [
            (str(conjecture["ID"]), _derive_seed(str(conjecture["ID"]), config.seed))
            for conjecture in selected
        ]
    else:
        context_seed_pairs = [(str(conjecture["ID"]), None) for conjecture in selected]

    process_all_conjectures(
        selected,
        output_path,
        config,
        show_plot=args.plots,
        cpus=args.cpus,
        context_seed_pairs=context_seed_pairs,
    )
def _result_to_outcome(conjecture: ConjectureMapping, result: SearchResult) -> SearchOutcome:
    """Convert a ``SearchResult`` into the canonical ``SearchOutcome`` structure."""

    graph = nx.from_graph6_bytes(result.counterexample_g6.encode("ascii"))
    evaluation = GraphEvaluation(
        graph=graph,
        graph6=result.counterexample_g6,
        score=result.score,
        details=None,
    )
    eligible = max(0, result.total_evaluated - result.total_rejected)
    elapsed = result.time
    return SearchOutcome(
        conjecture_id=str(conjecture["ID"]),
        found_counterexample=result.has_counterexample,
        best_evaluation=evaluation,
        elapsed=elapsed,
        evaluations_total=result.total_evaluated,
        evaluations_eligible=eligible,
        resets=result.reset,
        subclass_name=conjecture.get("subclass"),
    )


def _worker_entry(arguments: Tuple[ConjectureMapping, SearchConfig]) -> Tuple[ConjectureMapping, SearchResult]:
    """Entry point for multiprocessing workers."""

    conjecture, config = arguments
    result = process_conjecture(conjecture, config)
    return conjecture, result
