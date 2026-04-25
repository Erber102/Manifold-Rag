"""
Deterministic baselines: raw SBERT (no fine-tuning) and BM25.
Results are appended to results/manifold_results.json so they sit
in the same table as the manifold methods.

Run once after the corpus is built; no need to re-run with run_retrieval.py.
"""
import os
import json
import numpy as np
from collections import defaultdict
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("rank_bm25 not installed — BM25 baseline skipped. pip install rank-bm25")


def load_query_types(data_path: str) -> dict:
    types = {}
    qfile = os.path.join(data_path, "queries.jsonl")
    if not os.path.exists(qfile):
        return types
    with open(qfile, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj.get("_id", "")
            qa_type = obj.get("metadata", {}).get("qa_type", "unknown")
            if qid:
                types[qid] = qa_type
    return types


def eval_with_types(results: dict, qrels: dict, query_ids: list,
                    query_types: dict) -> dict:
    evaluator = EvaluateRetrieval()
    ndcg, map_score, recall, _ = evaluator.evaluate(qrels, results, [1, 5, 10, 100])
    run = {
        "NDCG@10":    ndcg["NDCG@10"],
        "Recall@10":  recall["Recall@10"],
        "Recall@100": recall["Recall@100"],
        "MAP@10":     map_score["MAP@10"],
    }
    type_to_qids: dict = defaultdict(list)
    for qid in query_ids:
        type_to_qids[query_types.get(qid, "unknown")].append(qid)

    per_type: dict = {}
    print(f"  Per-type metrics:")
    print(f"    {'qa_type':<28} {'n':>4} {'NDCG@10':>9} {'R@10':>7} {'R@100':>7}")
    for qa_type in sorted(type_to_qids.keys()):
        qids      = type_to_qids[qa_type]
        sub_qrels = {q: qrels[q]   for q in qids if q in qrels}
        sub_res   = {q: results[q] for q in qids if q in results}
        if not sub_qrels:
            continue
        nd, mp, re, _ = evaluator.evaluate(sub_qrels, sub_res, [1, 5, 10, 100])
        per_type[qa_type] = {
            "n":          len(sub_qrels),
            "NDCG@10":    nd["NDCG@10"],
            "Recall@10":  re["Recall@10"],
            "Recall@100": re["Recall@100"],
            "MAP@10":     mp["MAP@10"],
        }
        print(f"    {qa_type:<28} {len(sub_qrels):>4} "
              f"{nd['NDCG@10']:>9.4f} {re['Recall@10']:>7.4f} {re['Recall@100']:>7.4f}")
    run["per_type"] = per_type
    return run


def main():
    datasets = ["wiki"]
    base_data_path = "data"
    all_results: dict = {}

    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Baselines on dataset: {dataset}")
        print(f"{'='*60}")

        data_path = os.path.join(base_data_path, dataset)
        corpus, _, _ = GenericDataLoader(data_path).load(split="train")
        _, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")

        doc_ids = list(corpus.keys())
        test_query_ids = list(test_queries.keys())
        doc_texts = [(corpus[d].get("title", "") + " " + corpus[d]["text"]).strip()
                     for d in doc_ids]
        test_query_texts = [test_queries[q] for q in test_query_ids]
        query_types = load_query_types(data_path)

        type_counts = {}
        for qid in test_query_ids:
            t = query_types.get(qid, "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"  Corpus: {len(doc_ids)} docs | Test: {len(test_query_ids)}")
        print(f"  Test QA breakdown: {type_counts}")

        # ── 1. Raw SBERT ────────────────────────────────────────────────
        print("\n--- sbert_raw (no fine-tuning) ---")
        print("Encoding corpus...")
        corpus_emb = sbert.encode(doc_texts, batch_size=256, show_progress_bar=True,
                                  convert_to_numpy=True)
        print("Encoding queries...")
        query_emb = sbert.encode(test_query_texts, batch_size=256, show_progress_bar=True,
                                 convert_to_numpy=True)

        c_norm = (corpus_emb / np.linalg.norm(corpus_emb, axis=1, keepdims=True)).astype(np.float32)
        q_norm = (query_emb  / np.linalg.norm(query_emb,  axis=1, keepdims=True)).astype(np.float32)

        raw_results: dict = {}
        for i, qid in enumerate(tqdm(test_query_ids, desc="Retrieving")):
            s = q_norm[i] @ c_norm.T
            top = np.argsort(s)[::-1][:100]
            raw_results[qid] = {doc_ids[idx]: float(s[idx]) for idx in top}

        raw_run = eval_with_types(raw_results, test_qrels, test_query_ids, query_types)
        print(f"  Overall: NDCG@10={raw_run['NDCG@10']:.4f}, "
              f"Recall@10={raw_run['Recall@10']:.4f}, Recall@100={raw_run['Recall@100']:.4f}")
        all_results[f"{dataset}_sbert_raw"] = raw_run

        # ── 2. BM25 ─────────────────────────────────────────────────────
        if HAS_BM25:
            print("\n--- bm25 ---")
            print("Building BM25 index...")
            tokenized = [t.lower().split() for t in tqdm(doc_texts, desc="Tokenizing")]
            bm25_model = BM25Okapi(tokenized)

            bm25_results: dict = {}
            for qid, qtext in tqdm(zip(test_query_ids, test_query_texts),
                                   total=len(test_query_ids), desc="Retrieving"):
                scores = bm25_model.get_scores(qtext.lower().split())
                top = np.argsort(scores)[::-1][:100]
                bm25_results[qid] = {doc_ids[idx]: float(scores[idx]) for idx in top}

            bm25_run = eval_with_types(bm25_results, test_qrels, test_query_ids, query_types)
            print(f"  Overall: NDCG@10={bm25_run['NDCG@10']:.4f}, "
                  f"Recall@10={bm25_run['Recall@10']:.4f}, Recall@100={bm25_run['Recall@100']:.4f}")
            all_results[f"{dataset}_bm25"] = bm25_run

    # ── Save — append into manifold_results.json ────────────────────────
    os.makedirs("results", exist_ok=True)
    save_path = "results/manifold_results.json"
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results_final = existing
    else:
        all_results_final = all_results

    with open(save_path, "w") as f:
        json.dump(all_results_final, f, indent=2)
    print(f"\nSaved to {save_path}")

    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'NDCG@10':>10} {'Recall@10':>10} {'Recall@100':>11}")
    print("-" * 53)
    for key in [f"{d}_{m}" for d in datasets for m in ["sbert_raw", "bm25"]]:
        if key in all_results_final:
            m = all_results_final[key]
            print(f"{key:<20} {m['NDCG@10']:>10.4f} {m['Recall@10']:>10.4f} {m['Recall@100']:>11.4f}")


if __name__ == "__main__":
    main()
