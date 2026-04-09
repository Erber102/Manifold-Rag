# save as: experiments/run_retrieval.py

import os
import sys
import json
import torch
import geoopt
import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ Projection Heads ============

class EuclideanHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.proj(x)
        return out / out.norm(dim=-1, keepdim=True)  # L2 normalize
    
    def distance(self, a, b):
        # cosine distance = 1 - cosine_similarity
        return 1 - (a * b).sum(dim=-1)

class PoincareHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim, c=1.0):
        super().__init__()
        self.proj = torch.nn.Linear(input_dim, output_dim)
        self.ball = geoopt.PoincareBall(c=c)
        self.scale = torch.nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        v = self.proj(x) * self.scale  # scale to avoid boundary collapse
        return self.ball.expmap0(v)
    
    def distance(self, a, b):
        return self.ball.dist(a, b)

class ProductHead(torch.nn.Module):
    def __init__(self, input_dim, euclid_dim=32, hyper_dim=16, sphere_dim=16, c=1.0):
        super().__init__()
        self.proj_e = torch.nn.Linear(input_dim, euclid_dim)
        self.proj_h = torch.nn.Linear(input_dim, hyper_dim)
        self.proj_s = torch.nn.Linear(input_dim, sphere_dim)
        self.ball = geoopt.PoincareBall(c=c)
        self.scale_h = torch.nn.Parameter(torch.tensor(0.1))
        # learnable weights for combining distances
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.beta = torch.nn.Parameter(torch.tensor(1.0))
        self.gamma = torch.nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        e = self.proj_e(x)
        e = e / e.norm(dim=-1, keepdim=True)
        h = self.ball.expmap0(self.proj_h(x) * self.scale_h)
        s = self.proj_s(x)
        s = s / s.norm(dim=-1, keepdim=True)
        return e, h, s
    
    def distance(self, a, b):
        e_a, h_a, s_a = a
        e_b, h_b, s_b = b
        d_e = ((e_a - e_b) ** 2).sum(dim=-1)
        d_h = self.ball.dist(h_a, h_b) ** 2
        d_s = torch.acos(torch.clamp((s_a * s_b).sum(dim=-1), -1+1e-7, 1-1e-7)) ** 2
        return self.alpha**2 * d_e + self.beta**2 * d_h + self.gamma**2 * d_s

# ============ Training ============

def train_head(head, corpus_emb, queries_emb, qrels, query_ids, doc_ids,
               epochs=20, lr=1e-3, n_negatives=7, batch_size=64):
    """Train projection head with triplet loss.

    IMPORTANT: `queries_emb`/`query_ids`/`qrels` here MUST come from the
    training split (disjoint from the evaluation split). `corpus_emb`/`doc_ids`
    may be the shared corpus since BEIR uses one corpus across splits.
    """
    head = head.to(device)

    # Use RiemannianAdam if model has manifold params, else regular Adam
    params = list(head.parameters())
    optimizer = geoopt.optim.RiemannianAdam(params, lr=lr)

    # Build training pairs: (query_idx, pos_doc_idx)
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
    query_id_to_idx = {qid: i for i, qid in enumerate(query_ids)}
    train_pairs = []
    for qid in qrels:
        if qid not in query_id_to_idx:
            continue
        qi = query_id_to_idx[qid]
        for did, rel in qrels[qid].items():
            if rel > 0 and did in doc_id_to_idx:
                train_pairs.append((qi, doc_id_to_idx[did]))
    if len(train_pairs) == 0:
        raise RuntimeError(
            "No training pairs built — check that the train-split qrels were "
            "passed in and that their query ids are present in `query_ids`."
        )
    print(f"  Built {len(train_pairs)} (query, positive doc) training pairs "
          f"from {len(qrels)} train queries.")
    
    corpus_t = torch.tensor(corpus_emb, dtype=torch.float32, device=device)
    queries_t = torch.tensor(queries_emb, dtype=torch.float32, device=device)
    
    head.train()
    for epoch in range(epochs):
        np.random.shuffle(train_pairs)
        total_loss = 0
        n_batches = 0
        
        for start in range(0, len(train_pairs), batch_size):
            batch = train_pairs[start:start+batch_size]
            q_idx = [p[0] for p in batch]
            p_idx = [p[1] for p in batch]
            
            # Random negatives
            n_idx = [np.random.choice(len(doc_ids), n_negatives, replace=False) 
                     for _ in batch]
            
            q_emb = queries_t[q_idx]
            p_emb = corpus_t[p_idx]
            
            optimizer.zero_grad()
            
            q_proj = head(q_emb)
            p_proj = head(p_emb)
            
            if isinstance(q_proj, tuple):  # Product manifold
                pos_dist = head.distance(q_proj, p_proj)
                neg_losses = []
                for i, neg_indices in enumerate(n_idx):
                    for ni in neg_indices:
                        n_emb = corpus_t[ni:ni+1]
                        n_proj = head(n_emb)
                        qi_proj = tuple(x[i:i+1] for x in q_proj)
                        neg_dist = head.distance(qi_proj, n_proj)
                        neg_losses.append(neg_dist)
                neg_dist_all = torch.cat(neg_losses).view(len(batch), n_negatives)
                loss = torch.clamp(pos_dist.unsqueeze(1) - neg_dist_all + 0.5, min=0).mean()
            else:
                pos_dist = head.distance(q_proj, p_proj)
                neg_losses = []
                for i, neg_indices in enumerate(n_idx):
                    n_emb = corpus_t[neg_indices]
                    n_proj = head(n_emb)
                    qi_proj = q_proj[i:i+1].expand_as(n_proj)
                    neg_dist = head.distance(qi_proj, n_proj)
                    neg_losses.append(neg_dist)
                neg_dist_all = torch.stack(neg_losses)
                loss = torch.clamp(pos_dist.unsqueeze(1) - neg_dist_all + 0.5, min=0).mean()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: loss={total_loss/n_batches:.4f}")
    
    return head

# ============ Encode + Retrieve ============

def encode_with_head(head, embeddings, batch_size=512):
    head.eval()
    all_proj = []
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            batch = torch.tensor(embeddings[start:start+batch_size], 
                               dtype=torch.float32, device=device)
            proj = head(batch)
            if isinstance(proj, tuple):
                all_proj.append(tuple(x.cpu() for x in proj))
            else:
                all_proj.append(proj.cpu())
    
    if isinstance(all_proj[0], tuple):
        n_components = len(all_proj[0])
        return tuple(torch.cat([p[i] for p in all_proj]) for i in range(n_components))
    else:
        return torch.cat(all_proj)

def retrieve_with_head(head, corpus_proj, query_proj, doc_ids, query_ids, top_k=100):
    head.eval()
    results = {}
    
    with torch.no_grad():
        for i, qid in enumerate(tqdm(query_ids, desc="Retrieving")):
            if isinstance(query_proj, tuple):
                q = tuple(x[i:i+1].to(device) for x in query_proj)
                # Batch distance computation
                dists = []
                for start in range(0, len(doc_ids), 1000):
                    end = min(start + 1000, len(doc_ids))
                    c_batch = tuple(x[start:end].to(device) for x in corpus_proj)
                    q_exp = tuple(x.expand(end-start, -1) for x in q)
                    d = head.distance(q_exp, c_batch)
                    dists.append(d.cpu())
                dists = torch.cat(dists)
            else:
                q = query_proj[i:i+1].to(device)
                dists = []
                for start in range(0, len(doc_ids), 1000):
                    end = min(start + 1000, len(doc_ids))
                    c_batch = corpus_proj[start:end].to(device)
                    q_exp = q.expand(end-start, -1)
                    d = head.distance(q_exp, c_batch)
                    dists.append(d.cpu())
                dists = torch.cat(dists)
            
            top_indices = torch.argsort(dists)[:top_k]
            results[qid] = {doc_ids[idx]: float(-dists[idx]) for idx in top_indices}
    
    return results

# ============ Main ============

def main():
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    input_dim = sbert.get_sentence_embedding_dimension()  # 384
    
    datasets = ["scifact", "fiqa"]
    methods = {
        "euclidean_64": lambda: EuclideanHead(input_dim, 64),
        "poincare_64": lambda: PoincareHead(input_dim, 64),
        "product_64": lambda: ProductHead(input_dim, 32, 16, 16),
    }
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        data_path = os.path.join("data", dataset)

        # --- Load BEIR train split (for head training) and test split
        # (for evaluation). Corpus is shared across splits in BEIR, so we
        # only encode it once. Query sets and qrels are disjoint.
        corpus, train_queries, train_qrels = GenericDataLoader(data_path).load(split="train")
        _, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")

        # Sanity-check that train and test query ids do not overlap.
        overlap = set(train_queries.keys()) & set(test_queries.keys())
        if overlap:
            raise RuntimeError(
                f"Train/test query id overlap detected ({len(overlap)} ids) — "
                f"this would reintroduce leakage. Example: {next(iter(overlap))}"
            )

        doc_ids = list(corpus.keys())
        train_query_ids = list(train_queries.keys())
        test_query_ids = list(test_queries.keys())
        doc_texts = [(corpus[d].get("title","") + " " + corpus[d]["text"]).strip() for d in doc_ids]
        train_query_texts = [train_queries[q] for q in train_query_ids]
        test_query_texts = [test_queries[q] for q in test_query_ids]

        print(f"  Corpus: {len(doc_ids)} docs | "
              f"Train queries: {len(train_query_ids)} | "
              f"Test queries: {len(test_query_ids)}")

        print("Encoding corpus with SBERT...")
        corpus_emb = sbert.encode(doc_texts, batch_size=128, show_progress_bar=True)
        print("Encoding train queries with SBERT...")
        train_query_emb = sbert.encode(train_query_texts, batch_size=128, show_progress_bar=True)
        print("Encoding test queries with SBERT...")
        test_query_emb = sbert.encode(test_query_texts, batch_size=128, show_progress_bar=True)

        for method_name, head_fn in methods.items():
            print(f"\n--- {method_name} ---")
            head = head_fn()

            print("Training projection head on BEIR train split...")
            head = train_head(head, corpus_emb, train_query_emb,
                              train_qrels, train_query_ids, doc_ids,
                              epochs=20, lr=1e-3)

            print("Encoding with projection head...")
            corpus_proj = encode_with_head(head, corpus_emb)
            query_proj = encode_with_head(head, test_query_emb)

            print("Retrieving (evaluating on test split)...")
            results = retrieve_with_head(head, corpus_proj, query_proj,
                                        doc_ids, test_query_ids)

            evaluator = EvaluateRetrieval()
            ndcg, map_score, recall, precision = evaluator.evaluate(
                test_qrels, results, [1, 5, 10, 100])
            
            key = f"{dataset}_{method_name}"
            all_results[key] = {
                "NDCG@10": ndcg["NDCG@10"],
                "Recall@10": recall["Recall@10"],
                "Recall@100": recall["Recall@100"],
                "MAP@10": map_score["MAP@10"],
            }
            print(f"Results: NDCG@10={ndcg['NDCG@10']:.4f}, "
                  f"Recall@10={recall['Recall@10']:.4f}")
    
    # Save all
    os.makedirs("results", exist_ok=True)
    with open("results/manifold_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'NDCG@10':>10} {'Recall@10':>10} {'Recall@100':>10}")
    print("-" * 60)
    for key, metrics in sorted(all_results.items()):
        print(f"{key:<25} {metrics['NDCG@10']:>10.4f} {metrics['Recall@10']:>10.4f} "
              f"{metrics['Recall@100']:>10.4f}")

if __name__ == "__main__":
    main()