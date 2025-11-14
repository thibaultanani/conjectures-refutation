from multiprocessing import cpu_count
from pathlib import Path
from typing import List

from helpers import scores_function
from helpers.utility import load_conjectures

from refutation_heuristics.local_search import (
    SearchConfig,
    SearchParameters,
    _derive_seed,
    process_all_conjectures,
)


def main() -> None:
    identifiers = _load_identifiers(Path("data/identifiers.txt"))

    dataset = load_conjectures("data/benchmark.csv")
    if not dataset:
        raise SystemExit("No conjectures were loaded from benchmark.csv")

    by_identifier = {row.get("ID"): row for row in dataset if row.get("ID") is not None}
    selected: List[dict] = []
    missing: List[str] = []
    for identifier in identifiers:
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
        raise SystemExit(f"Unable to resolve conjecture ID(s): {missing_ids}")

    params = SearchParameters(
        min_size=6,
        max_size=30,
        neighbor_count=10,
        max_mutations=2,
        time_limit=60.0 * 20,
        stagnation_limit=10,
        margin=1e-3,
        mutation_names=(
            "add_edge",
            "remove_edge",
            "add_vertex",
            "remove_vertex",
            "subdivision",
            "contraction",
            "replace_vertex_by_path",
            "replace_vertex_by_star",
            "replace_vertex_by_clique",
            "replace_vertex_by_polyhedral",
            "bipartition_neighborhood",
        ),
        seed=42,
        verbose=True,
    )

    config = SearchConfig(
        neighbour_count=params.neighbor_count,
        min_size=params.min_size,
        max_size=params.max_size,
        max_mutations=params.max_mutations,
        time_limit=params.time_limit,
        stagnation_limit=params.stagnation_limit,
        margin=params.margin,
        cache_size_limit=getattr(params, "cache_size_limit", None),
        mutation_names=params.mutation_names or None,
        verbose=params.verbose,
        seed=params.seed,
    )

    if config.seed is not None:
        context_seed_pairs = [
            (identifier, _derive_seed(identifier, config.seed)) for identifier in identifiers
        ]
    else:
        context_seed_pairs = [(identifier, None) for identifier in identifiers]

    output_dir = Path("out")
    cpus = max(1, cpu_count() - 0)

    process_all_conjectures(
        selected,
        output_dir,
        config,
        show_plot=False,
        cpus=cpus,
        context_seed_pairs=context_seed_pairs,
    )


def _load_identifiers(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"Identifier file not found: {path}")
    identifiers: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                identifiers.append(value)
    if not identifiers:
        raise SystemExit(f"Identifier file {path} is empty")
    return identifiers


if __name__ == "__main__":
    main()
