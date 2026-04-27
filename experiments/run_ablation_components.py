"""Ablation: Component removal experiment.

Fixed total dim=64. Remove individual geometric components from the Product
manifold to measure each component's contribution.

Variants:
  product_full:    (E=32, H=16, S=16)
  euclidean_only:  (E=64, H=0,  S=0)
  poincare_only:   (H=64, E=0,  S=0)
  product_no_H:    (E=48, H=0,  S=16)
  product_no_S:    (E=48, H=16, S=0)
  product_no_E:    (E=0,  H=32, S=32)

Run on wiki dataset (where Product advantage is most pronounced).
Output: results/ablation_component_results.json
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from manifold_utils import (
    device, set_seed, EuclideanHead, PoincareHead, ProductHead,
    build_hard_neg_index, build_level_targets, load_query_types,
    train_head, encode_with_head, retrieve_with_head,
    eval_with_types, aggregate_seed_runs,
    load_dataset, encode_dataset, get_training_config,
)
from sentence_transformers import SentenceTransformer

SEEDS = [42, 123, 456]


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    input_dim = sbert.get_sentence_embedding_dimension()

    methods = {
        "product_full":    lambda: ProductHead(input_dim, euclid_dim=32, hyper_dim=16, sphere_dim=16),
        "euclidean_only":  lambda: EuclideanHead(input_dim, 64),
        "poincare_only":   lambda: PoincareHead(input_dim, 64),
        "product_no_H":    lambda: ProductHead(input_dim, euclid_dim=48, hyper_dim=0,  sphere_dim=16),
        "product_no_S":    lambda: ProductHead(input_dim, euclid_dim=48, hyper_dim=16, sphere_dim=0),
        "product_no_E":    lambda: ProductHead(input_dim, euclid_dim=0,  hyper_dim=32, sphere_dim=32),
    }

    data_path = os.path.join("data", "wiki")
    data = load_dataset(data_path)
    data = encode_dataset(sbert, data, "wiki")

    hard_neg_index = build_hard_neg_index(data["corpus"], data["doc_ids"])
    level_targets = build_level_targets(data["doc_ids"], data["corpus"])
    query_types = load_query_types(data_path)

    all_results = {}
    per_seed_runs = {m: [] for m in methods}

    for seed in SEEDS:
        print(f"\n{'#'*50}")
        print(f"# SEED = {seed}")
        print(f"{'#'*50}")
        set_seed(seed)

        for method_name, head_fn in methods.items():
            print(f"\n--- {method_name} (seed={seed}) ---")
            head = head_fn()

            is_poincare = "poincare" in method_name
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

            corpus_proj = encode_with_head(head, data["corpus_emb"])
            query_proj = encode_with_head(head, data["test_query_emb"])
            results = retrieve_with_head(head, corpus_proj, query_proj,
                                         data["doc_ids"], data["test_query_ids"])

            run_result = eval_with_types(results, data["test_qrels"],
                                         data["test_query_ids"], query_types)
            print(f"  NDCG@10={run_result['NDCG@10']:.4f}")

            seed_key = f"wiki_{method_name}_seed{seed}"
            all_results[seed_key] = run_result
            per_seed_runs[method_name].append(run_result)

    # Aggregate
    for method_name, runs in per_seed_runs.items():
        agg_key = f"wiki_{method_name}"
        all_results[agg_key] = aggregate_seed_runs(runs)
        agg = all_results[agg_key]
        print(f"  AGG {method_name}: NDCG@10={agg['NDCG@10']:.4f} "
              f"± {agg.get('NDCG@10_std', 0):.4f}")

    os.makedirs("results", exist_ok=True)
    save_path = "results/ablation_component_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    # Summary
    print(f"\n{'='*70}")
    print("COMPONENT ABLATION SUMMARY (wiki, NDCG@10 mean ± std)")
    print(f"{'='*70}")
    print(f"  {'Method':<20} {'Config':<18} {'NDCG@10':>16}")
    configs = {
        "product_full":   "(E=32, H=16, S=16)",
        "euclidean_only": "(E=64)",
        "poincare_only":  "(H=64)",
        "product_no_H":   "(E=48, S=16)",
        "product_no_S":   "(E=48, H=16)",
        "product_no_E":   "(H=32, S=32)",
    }
    for method_name in methods:
        key = f"wiki_{method_name}"
        if key in all_results:
            m = all_results[key]
            val = f"{m['NDCG@10']:.4f}±{m.get('NDCG@10_std',0):.3f}"
            print(f"  {method_name:<20} {configs[method_name]:<18} {val:>16}")


if __name__ == "__main__":
    main()
