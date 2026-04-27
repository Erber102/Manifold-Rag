"""Dimension sweep experiment.

Compare Euclidean / Poincaré / Product manifold heads across embedding
dimensions 8, 16, 32, 64 on all datasets (scifact, fiqa, wiki).
Each configuration is run with 3 seeds and results are aggregated as mean ± std.

Output: results/dimension_sweep_results.json
"""

import os
import sys
import json
import numpy as np

# Allow imports from experiments/
sys.path.insert(0, os.path.dirname(__file__))

from manifold_utils import (
    device, set_seed, EuclideanHead, PoincareHead, ProductHead,
    build_hard_neg_index, build_level_targets, load_query_types,
    train_head, encode_with_head, retrieve_with_head,
    eval_with_types, aggregate_seed_runs,
    load_dataset, encode_dataset, get_training_config, warm_start_head,
)
from sentence_transformers import SentenceTransformer

SEEDS = [42, 123, 456]
DIMS = [8, 16, 32, 64]
DATASETS = ["scifact", "fiqa", "wiki"]


def build_methods(input_dim, dim):
    """Build method factories for a given target dimension."""
    # Product split: (d//2, d//4, d//4)
    e_dim = dim // 2
    h_dim = dim // 4
    s_dim = dim - e_dim - h_dim  # handle odd splits
    return {
        f"euclidean_{dim}": lambda d=dim: EuclideanHead(input_dim, d),
        f"poincare_{dim}": lambda d=dim: PoincareHead(input_dim, d),
        f"product_{dim}": lambda ed=e_dim, hd=h_dim, sd=s_dim: ProductHead(input_dim, ed, hd, sd),
    }


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    input_dim = sbert.get_sentence_embedding_dimension()

    all_results = {}

    for dataset_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        data_path = os.path.join("data", dataset_name)
        data = load_dataset(data_path)
        data = encode_dataset(sbert, data, dataset_name)

        hard_neg_index = build_hard_neg_index(data["corpus"], data["doc_ids"])
        level_targets = build_level_targets(data["doc_ids"], data["corpus"])
        query_types = load_query_types(data_path)

        for dim in DIMS:
            methods = build_methods(input_dim, dim)
            per_seed_runs = {m: [] for m in methods}

            for seed in SEEDS:
                print(f"\n{'#'*50}")
                print(f"# dim={dim}, seed={seed}")
                print(f"{'#'*50}")
                set_seed(seed)

                # Train a Euclidean head at this dim for warm-starting
                trained_euclidean_head = None

                for method_name, head_fn in methods.items():
                    print(f"\n--- {method_name} (seed={seed}) ---")
                    head = head_fn()
                    is_poincare = "poincare" in method_name

                    # Warm-start from Euclidean (only if dim matches)
                    if trained_euclidean_head is not None and "euclidean" not in method_name:
                        warm_start_head(head, trained_euclidean_head, method_name)

                    cfg = get_training_config(method_name)
                    use_norm_reg = is_poincare or "product" in method_name

                    head = train_head(
                        head, data["corpus_emb"], data["train_query_emb"],
                        data["train_qrels"], data["train_query_ids"], data["doc_ids"],
                        epochs=cfg["epochs"], lr=cfg["lr"],
                        hard_neg_index=hard_neg_index, n_hard_per_sample=4, margin=0.5,
                        level_targets=level_targets if use_norm_reg else None,
                        lambda_reg=0.2,
                    )

                    if "euclidean" in method_name:
                        trained_euclidean_head = head

                    corpus_proj = encode_with_head(head, data["corpus_emb"])
                    query_proj = encode_with_head(head, data["test_query_emb"])
                    results = retrieve_with_head(head, corpus_proj, query_proj,
                                                 data["doc_ids"], data["test_query_ids"])

                    run_result = eval_with_types(results, data["test_qrels"],
                                                 data["test_query_ids"], query_types)
                    print(f"  NDCG@10={run_result['NDCG@10']:.4f}")

                    seed_key = f"{dataset_name}_{method_name}_seed{seed}"
                    all_results[seed_key] = run_result
                    per_seed_runs[method_name].append(run_result)

            # Aggregate across seeds for this dim
            for method_name, runs in per_seed_runs.items():
                agg_key = f"{dataset_name}_{method_name}"
                all_results[agg_key] = aggregate_seed_runs(runs)
                agg = all_results[agg_key]
                print(f"  AGG {method_name}: NDCG@10={agg['NDCG@10']:.4f} "
                      f"± {agg.get('NDCG@10_std', 0):.4f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    save_path = "results/dimension_sweep_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("DIMENSION SWEEP SUMMARY (NDCG@10 mean ± std)")
    print(f"{'='*80}")
    for dataset_name in DATASETS:
        print(f"\n  {dataset_name}:")
        print(f"    {'Method':<20} " + "".join(f"{'dim='+str(d):>18}" for d in DIMS))
        for method_base in ["euclidean", "poincare", "product"]:
            row = f"    {method_base:<20} "
            for dim in DIMS:
                key = f"{dataset_name}_{method_base}_{dim}"
                if key in all_results:
                    m = all_results[key]
                    row += f"{m['NDCG@10']:.4f}±{m.get('NDCG@10_std',0):.3f}".rjust(18)
                else:
                    row += "-".rjust(18)
            print(row)


if __name__ == "__main__":
    main()
