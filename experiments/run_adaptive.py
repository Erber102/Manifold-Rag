"""Adaptive Product Head experiment.

Train and evaluate AdaptiveProductHead which learns query-dependent distance
weights (alpha, beta, gamma) via a small MLP on the raw SBERT embedding.

Compares adaptive_product_64 vs static product_64 on all datasets, with
special attention to per-type breakdown on wiki.

Also saves the learned weights per query for later visualization.
Output: results/adaptive_results.json, results/adaptive_weights.json
"""

import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from manifold_utils import (
    device, set_seed, ProductHead, AdaptiveProductHead,
    build_hard_neg_index, build_level_targets, load_query_types,
    train_head, encode_with_head, retrieve_with_head,
    eval_with_types, aggregate_seed_runs,
    load_dataset, encode_dataset,
)
from sentence_transformers import SentenceTransformer

SEEDS = [42, 123, 456]
DATASETS = ["scifact", "fiqa", "wiki"]


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    input_dim = sbert.get_sentence_embedding_dimension()

    methods = {
        "product_64": lambda: ProductHead(input_dim, 32, 16, 16),
        "adaptive_product_64": lambda: AdaptiveProductHead(input_dim, 32, 16, 16),
    }

    all_results = {}
    all_weights = {}  # dataset -> qa_type -> list of weight triples

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

        per_seed_runs = {m: [] for m in methods}

        for seed in SEEDS:
            print(f"\n{'#'*50}")
            print(f"# SEED = {seed}")
            print(f"{'#'*50}")
            set_seed(seed)

            for method_name, head_fn in methods.items():
                print(f"\n--- {method_name} (seed={seed}) ---")
                head = head_fn()
                is_adaptive = isinstance(head, AdaptiveProductHead)

                head = train_head(
                    head, data["corpus_emb"], data["train_query_emb"],
                    data["train_qrels"], data["train_query_ids"], data["doc_ids"],
                    epochs=30, lr=1e-3,
                    hard_neg_index=hard_neg_index, n_hard_per_sample=4, margin=0.5,
                    level_targets=level_targets, lambda_reg=0.2,
                    raw_query_emb=data["train_query_emb"] if is_adaptive else None,
                )

                corpus_proj = encode_with_head(head, data["corpus_emb"])
                query_proj = encode_with_head(head, data["test_query_emb"])

                # Compute adaptive weights for test queries
                query_weights = None
                if is_adaptive:
                    head.eval()
                    with torch.no_grad():
                        raw_test = torch.tensor(data["test_query_emb"],
                                                dtype=torch.float32, device=device)
                        query_weights = head.get_weights(raw_test).cpu()

                    # Save per-type weight statistics (last seed only for visualization)
                    if seed == SEEDS[-1]:
                        type_weights = {}
                        for i, qid in enumerate(data["test_query_ids"]):
                            qt = query_types.get(qid, "unknown")
                            if qt not in type_weights:
                                type_weights[qt] = []
                            w = query_weights[i].tolist()
                            type_weights[qt].append(w)

                        ds_weights = {}
                        for qt, ws in type_weights.items():
                            arr = np.array(ws)
                            ds_weights[qt] = {
                                "n": len(ws),
                                "mean_alpha": float(arr[:, 0].mean()),
                                "mean_beta": float(arr[:, 1].mean()),
                                "mean_gamma": float(arr[:, 2].mean()),
                                "std_alpha": float(arr[:, 0].std()),
                                "std_beta": float(arr[:, 1].std()),
                                "std_gamma": float(arr[:, 2].std()),
                                "raw_weights": arr.tolist(),
                            }
                        all_weights[dataset_name] = ds_weights

                results = retrieve_with_head(head, corpus_proj, query_proj,
                                             data["doc_ids"], data["test_query_ids"],
                                             query_weights=query_weights)

                run_result = eval_with_types(results, data["test_qrels"],
                                             data["test_query_ids"], query_types)
                print(f"  NDCG@10={run_result['NDCG@10']:.4f}")

                seed_key = f"{dataset_name}_{method_name}_seed{seed}"
                all_results[seed_key] = run_result
                per_seed_runs[method_name].append(run_result)

        # Aggregate
        for method_name, runs in per_seed_runs.items():
            agg_key = f"{dataset_name}_{method_name}"
            all_results[agg_key] = aggregate_seed_runs(runs)
            agg = all_results[agg_key]
            print(f"  AGG {method_name}: NDCG@10={agg['NDCG@10']:.4f} "
                  f"± {agg.get('NDCG@10_std', 0):.4f}")

    os.makedirs("results", exist_ok=True)

    save_path = "results/adaptive_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    # Save weights (without raw_weights for the main file, keep a separate detailed one)
    weights_summary = {}
    for ds, types in all_weights.items():
        weights_summary[ds] = {}
        for qt, info in types.items():
            weights_summary[ds][qt] = {k: v for k, v in info.items() if k != "raw_weights"}

    weights_path = "results/adaptive_weights.json"
    with open(weights_path, "w") as f:
        json.dump(all_weights, f, indent=2)
    print(f"Weights saved to {weights_path}")

    # Summary
    print(f"\n{'='*70}")
    print("ADAPTIVE vs STATIC PRODUCT (NDCG@10 mean ± std)")
    print(f"{'='*70}")
    for dataset_name in DATASETS:
        print(f"\n  {dataset_name}:")
        for method_name in methods:
            key = f"{dataset_name}_{method_name}"
            if key in all_results:
                m = all_results[key]
                print(f"    {method_name:<25} {m['NDCG@10']:.4f} ± {m.get('NDCG@10_std',0):.4f}")

    # Print weight analysis for wiki
    if "wiki" in all_weights:
        print(f"\n{'='*70}")
        print("ADAPTIVE WEIGHTS BY QA TYPE (wiki dataset)")
        print(f"{'='*70}")
        print(f"  {'qa_type':<28} {'α (euclid)':>12} {'β (hyper)':>12} {'γ (sphere)':>12}")
        for qt, info in sorted(all_weights["wiki"].items()):
            print(f"  {qt:<28} "
                  f"{info['mean_alpha']:.4f}±{info['std_alpha']:.3f} "
                  f"{info['mean_beta']:.4f}±{info['std_beta']:.3f} "
                  f"{info['mean_gamma']:.4f}±{info['std_gamma']:.3f}")


if __name__ == "__main__":
    main()
