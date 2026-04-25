import os
import json
import torch
import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def sbert_retrieve(corpus, queries, model, top_k=100, batch_size=128):
    doc_ids = list(corpus.keys())
    doc_texts = [
        (corpus[did].get("title", "") + " " + corpus[did]["text"]).strip()
        for did in doc_ids
    ]
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    print("Encoding corpus...")
    corpus_emb = model.encode(doc_texts, batch_size=batch_size, 
                               show_progress_bar=True, convert_to_numpy=True)
    print("Encoding queries...")
    query_emb = model.encode(query_texts, batch_size=batch_size,
                              show_progress_bar=True, convert_to_numpy=True)

    # Normalize for cosine similarity
    corpus_emb = corpus_emb / np.linalg.norm(corpus_emb, axis=1, keepdims=True)
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

    # Retrieve
    results = {}
    for i, qid in enumerate(tqdm(query_ids, desc="Retrieving")):
        scores = query_emb[i] @ corpus_emb.T
        top_indices = np.argsort(scores)[::-1][:top_k]
        results[qid] = {doc_ids[idx]: float(scores[idx]) for idx in top_indices}

    return results

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Model embedding dim: {model.get_sentence_embedding_dimension()}")

datasets = ["wiki", "scifact", "fiqa"]
results_all = {}

for dataset in datasets:
    print(f"\n{'='*50}")
    print(f"SBERT on {dataset}")
    print(f"{'='*50}")

    data_path = os.path.join("data", dataset)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    results = sbert_retrieve(corpus, queries, model)

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
save_path = "results/sbert_results.json"
if os.path.exists(save_path):
    with open(save_path, "r") as f:
        existing = json.load(f)
    existing.update(results_all)
    results_all = existing
with open(save_path, "w") as f:
    json.dump(results_all, f, indent=2)

print("\nSaved to results/sbert_results.json")