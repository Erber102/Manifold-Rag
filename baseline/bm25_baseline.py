import os
import json
import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from rank_bm25 import BM25Okapi
from tqdm import tqdm

def bm25_retrieve(corpus, queries, top_k=100):
    doc_ids = list(corpus.keys())
    # tokenize corpus
    tokenized = [
        (corpus[did].get("title", "") + " " + corpus[did]["text"]).lower().split()
        for did in tqdm(doc_ids, desc="Tokenizing corpus")
    ]
    
    bm25 = BM25Okapi(tokenized)
    
    results = {}
    for qid, query in tqdm(queries.items(), desc="Retrieving"):
        scores = bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        results[qid] = {doc_ids[i]: float(scores[i]) for i in top_indices}
    
    return results

datasets = ["scifact", "fiqa"]
results_all = {}

for dataset in datasets:
    print(f"\n{'='*50}")
    print(f"BM25 on {dataset}")
    print(f"{'='*50}")
    
    data_path = os.path.join("data", dataset)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    results = bm25_retrieve(corpus, queries, top_k=100)
    
    evaluator = EvaluateRetrieval()
    ndcg, map_score, recall, precision = evaluator.evaluate(qrels, results, [1, 5, 10, 100])
    
    results_all[dataset] = {
        "NDCG@10": ndcg["NDCG@10"],
        "Recall@10": recall["Recall@10"],
        "Recall@100": recall["Recall@100"],
        "MAP@10": map_score["MAP@10"],
    }
    
    print(f"\nResults:")
    for metric, value in results_all[dataset].items():
        print(f"  {metric}: {value:.4f}")

os.makedirs("results", exist_ok=True)
with open("results/bm25_results.json", "w") as f:
    json.dump(results_all, f, indent=2)

print("\nSaved to results/bm25_results.json")