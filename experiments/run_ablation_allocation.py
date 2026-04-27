"""Ablation: Dimension allocation experiment.

Fixed total dim=64. Explore different (E, H, S) splits to find optimal
allocation across geometric components.

Allocations:
  (32, 16, 16) — default
  (16, 32, 16) — hyperbolic-heavy
  (16, 16, 32) — spherical-heavy
  (48, 8, 8)   — euclidean-heavy
  (22, 21, 21) — equal split

Run on all datasets with 3 seeds.
Output: results/ablation_allocation_results.json
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from manifold_utils import (
    device, set_seed, ProductHead,
    build_hard_neg_index, build_level_targets, load_query_types,
    train_head, encode_with_head, retrieve_with_head,
    eval_with_types, aggregate_seed_runs,
    load_dataset, encode_dataset,
)
from sentence_transformers import SentenceTransformer

SEEDS = [42, 123, 456]
DATASETS = ["scifact", "fiqa", "wiki"]
ALLOCATIONS = [
    (32, 16, 16),  # default
    (16, 32, 16),  # hyperbolic-heavy
    (16, 16, 32),  # spherical-heavy
    (48, 8, 8),    # euclidean-heavy
    (22, 21, 21),  # equal
]


def alloc_name(e, h, s):
    return f"product_E{e}_H{h}_S{s}"


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

        per_seed_runs = {alloc_name(*a): [] for a in ALLOCATIONS}

        for seed in SEEDS:
            print(f"\n{'#'*50}")
            print(f"# SEED = {seed}")
            print(f"{'#'*50}")
            set_seed(seed)

            for e_dim, h_dim, s_dim in ALLOCATIONS:
                name = alloc_name(e_dim, h_dim, s_dim)
                print(f"\n--- {name} (seed={seed}) ---")

                head = ProductHead(input_dim, euclid_dim=e_dim,
                                   hyper_dim=h_dim, sphere_dim=s_dim)

                head = train_head(
                    head, data["corpus_emb"], data["train_query_emb"],
                    data["train_qrels"], data["train_query_ids"], data["doc_ids"],
                    epochs=30, lr=1e-3,
                    hard_neg_index=hard_neg_index, n_hard_per_sample=4, margin=0.5,
                    level_targets=level_targets, lambda_reg=0.2,
                )

                corpus_proj = encode_with_head(head, data["corpus_emb"])
                query_proj = encode_with_head(head, data["test_query_emb"])
                results = retrieve_with_head(head, corpus_proj, query_proj,
                                             data["doc_ids"], data["test_query_ids"])

                run_result = eval_with_types(results, data["test_qrels"],
                                             data["test_query_ids"], query_types)
                print(f"  NDCG@10={run_result['NDCG@10']:.4f}")

                seed_key = f"{dataset_name}_{name}_seed{seed}"
                all_results[seed_key] = run_result
                per_seed_runs[name].append(run_result)

        # Aggregate
        for name, runs in per_seed_runs.items():
            agg_key = f"{dataset_name}_{name}"
            all_results[agg_key] = aggregate_seed_runs(runs)
            agg = all_results[agg_key]
            print(f"  AGG {name}: NDCG@10={agg['NDCG@10']:.4f} "
                  f"± {agg.get('NDCG@10_std', 0):.4f}")

    os.makedirs("results", exist_ok=True)
    save_path = "results/ablation_allocation_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    # Summary
    print(f"\n{'='*80}")
    print("ALLOCATION ABLATION SUMMARY (NDCG@10 mean ± std)")
    print(f"{'='*80}")
    for dataset_name in DATASETS:
        print(f"\n  {dataset_name}:")
        print(f"    {'Allocation':<25} {'NDCG@10':>16}")
        for e_dim, h_dim, s_dim in ALLOCATIONS:
            name = alloc_name(e_dim, h_dim, s_dim)
            key = f"{dataset_name}_{name}"
            if key in all_results:
                m = all_results[key]
                val = f"{m['NDCG@10']:.4f}±{m.get('NDCG@10_std',0):.3f}"
                print(f"    ({e_dim:>2},{h_dim:>2},{s_dim:>2}){'':<14} {val:>16}")


if __name__ == "__main__":
    main()
