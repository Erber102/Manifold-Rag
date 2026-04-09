# 本地运行加速版
import os
import json
import torch
import geoopt
import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")

# ============ Projection Heads ============

class EuclideanHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.proj(x)
        return out / out.norm(dim=-1, keepdim=True)

    def distance(self, a, b):
        return 1 - (a * b).sum(dim=-1)

    def batch_retrieve(self, query_proj, corpus_proj, device):
        scores = []
        c = corpus_proj.to(device)
        for i in range(0, query_proj.shape[0], 512):
            q = query_proj[i:i+512].to(device)
            scores.append((q @ c.T).cpu())
        return torch.cat(scores, dim=0)


class PoincareHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim, c=1.0):
        super().__init__()
        self.proj = torch.nn.Linear(input_dim, output_dim)
        self.ball = geoopt.PoincareBall(c=c)
        self.scale = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        v = self.proj(x) * self.scale
        return self.ball.expmap0(v)

    def distance(self, a, b):
        return self.ball.dist(a, b)

    def batch_retrieve(self, query_proj, corpus_proj, device, q_chunk=32, c_chunk=4096):
        n_q, n_c = query_proj.shape[0], corpus_proj.shape[0]
        scores = torch.zeros(n_q, n_c)
        c_dev = corpus_proj.to(device)
        for qi in range(0, n_q, q_chunk):
            qi_end = min(qi + q_chunk, n_q)
            qb = query_proj[qi:qi_end].to(device)
            for ci in range(0, n_c, c_chunk):
                ci_end = min(ci + c_chunk, n_c)
                cb = c_dev[ci:ci_end]
                bq, bc = qb.shape[0], cb.shape[0]
                q_exp = qb.unsqueeze(1).expand(bq, bc, -1).reshape(-1, qb.shape[-1])
                c_exp = cb.unsqueeze(0).expand(bq, bc, -1).reshape(-1, cb.shape[-1])
                d = self.ball.dist(q_exp, c_exp).reshape(bq, bc)
                scores[qi:qi_end, ci:ci_end] = -d.cpu()
        return scores


class ProductHead(torch.nn.Module):
    def __init__(self, input_dim, euclid_dim=32, hyper_dim=16, sphere_dim=16, c=1.0):
        super().__init__()
        self.proj_e = torch.nn.Linear(input_dim, euclid_dim)
        self.proj_h = torch.nn.Linear(input_dim, hyper_dim)
        self.proj_s = torch.nn.Linear(input_dim, sphere_dim)
        self.ball = geoopt.PoincareBall(c=c)
        self.scale_h = torch.nn.Parameter(torch.tensor(0.1))
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

    def batch_retrieve(self, query_proj, corpus_proj, device, q_chunk=32, c_chunk=4096):
        q_e, q_h, q_s = query_proj
        c_e, c_h, c_s = corpus_proj
        n_q, n_c = q_e.shape[0], c_e.shape[0]

        c_e_d = c_e.to(device)
        dist_e = torch.zeros(n_q, n_c)
        for i in range(0, n_q, 512):
            j = min(i+512, n_q)
            dist_e[i:j] = (2 - 2 * q_e[i:j].to(device) @ c_e_d.T).cpu()

        c_s_d = c_s.to(device)
        dist_s = torch.zeros(n_q, n_c)
        for i in range(0, n_q, 512):
            j = min(i+512, n_q)
            dots = torch.clamp(q_s[i:j].to(device) @ c_s_d.T, -1+1e-7, 1-1e-7)
            dist_s[i:j] = (torch.acos(dots) ** 2).cpu()

        c_h_d = c_h.to(device)
        dist_h = torch.zeros(n_q, n_c)
        for qi in range(0, n_q, q_chunk):
            qi_end = min(qi + q_chunk, n_q)
            qb = q_h[qi:qi_end].to(device)
            for ci in range(0, n_c, c_chunk):
                ci_end = min(ci + c_chunk, n_c)
                cb = c_h_d[ci:ci_end]
                bq, bc = qb.shape[0], cb.shape[0]
                q_exp = qb.unsqueeze(1).expand(bq, bc, -1).reshape(-1, qb.shape[-1])
                c_exp = cb.unsqueeze(0).expand(bq, bc, -1).reshape(-1, cb.shape[-1])
                d = self.ball.dist(q_exp, c_exp).reshape(bq, bc)
                dist_h[qi:qi_end, ci:ci_end] = (d ** 2).cpu()

        a2 = (self.alpha**2).item()
        b2 = (self.beta**2).item()
        g2 = (self.gamma**2).item()
        return -(a2 * dist_e + b2 * dist_h + g2 * dist_s)


# ============ Training ============

def train_head(head, corpus_emb, queries_emb, qrels, query_ids, doc_ids,
               epochs=20, lr=1e-3, n_negatives=7, batch_size=256):
    head = head.to(device)
    optimizer = geoopt.optim.RiemannianAdam(head.parameters(), lr=lr)

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
        raise RuntimeError("No training pairs built...")

    print(f"  Built {len(train_pairs)} training pairs. batch_size={batch_size}")

    corpus_t = torch.tensor(corpus_emb, dtype=torch.float32, device=device)
    queries_t = torch.tensor(queries_emb, dtype=torch.float32, device=device)
    n_docs = len(doc_ids)

    head.train()
    for epoch in range(epochs):
        np.random.shuffle(train_pairs)
        total_loss = 0
        n_batches = 0

        for start in range(0, len(train_pairs), batch_size):
            batch = train_pairs[start:start+batch_size]
            B = len(batch)
            q_idx = [p[0] for p in batch]
            p_idx = [p[1] for p in batch]

            q_emb = queries_t[q_idx]
            p_emb = corpus_t[p_idx]

            neg_idx = np.random.randint(0, n_docs, (B, n_negatives))
            n_emb = corpus_t[neg_idx.flatten()]

            optimizer.zero_grad()
            q_proj = head(q_emb)
            p_proj = head(p_emb)
            n_proj = head(n_emb)

            if isinstance(q_proj, tuple):
                pos_dist = head.distance(q_proj, p_proj)
                n_comp = len(q_proj)
                q_exp = tuple(
                    x.unsqueeze(1).expand(-1, n_negatives, -1).reshape(-1, x.shape[-1])
                    for x in q_proj
                )
                n_shaped = tuple(
                    n_proj[k].reshape(B * n_negatives, -1) for k in range(n_comp)
                )
                neg_dist = head.distance(q_exp, n_shaped).reshape(B, n_negatives)
            else:
                pos_dist = head.distance(q_proj, p_proj)
                d = q_proj.shape[-1]
                q_exp = q_proj.unsqueeze(1).expand(-1, n_negatives, -1).reshape(-1, d)
                n_shaped = n_proj.reshape(B * n_negatives, d)
                neg_dist = head.distance(q_exp, n_shaped).reshape(B, n_negatives)

            loss = torch.clamp(pos_dist.unsqueeze(1) - neg_dist + 0.5, min=0).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: loss={total_loss/n_batches:.4f}")

    return head


# ============ Encode + Retrieve ============

def encode_with_head(head, embeddings, batch_size=2048):
    head.eval()
    all_proj = []
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            batch = torch.tensor(embeddings[start:start+batch_size], dtype=torch.float32, device=device)
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
    print("  Computing distances (vectorized)...")
    with torch.no_grad():
        scores = head.batch_retrieve(query_proj, corpus_proj, device)

    print("  Sorting top-k...")
    topk = torch.topk(scores, top_k, dim=1)
    top_indices = topk.indices
    top_scores = topk.values

    results = {}
    for i, qid in enumerate(query_ids):
        results[qid] = {
            doc_ids[idx]: float(top_scores[i, j])
            for j, idx in enumerate(top_indices[i].tolist())
        }
    return results


# ============ Main ============

def main():
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    input_dim = sbert.get_sentence_embedding_dimension()

    datasets = ["scifact", "fiqa"]
    methods = {
        "euclidean_64": lambda: EuclideanHead(input_dim, 64),
        "poincare_64": lambda: PoincareHead(input_dim, 64),
        "product_64": lambda: ProductHead(input_dim, 32, 16, 16),
    }

    all_results = {}
    base_data_path = "datasets"

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        data_path = os.path.join(base_data_path, dataset)

        corpus, train_queries, train_qrels = GenericDataLoader(data_path).load(split="train")
        _, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")

        doc_ids = list(corpus.keys())
        train_query_ids = list(train_queries.keys())
        test_query_ids = list(test_queries.keys())
        doc_texts = [(corpus[d].get("title","") + " " + corpus[d]["text"]).strip() for d in doc_ids]
        train_query_texts = [train_queries[q] for q in train_query_ids]
        test_query_texts = [test_queries[q] for q in test_query_ids]

        print(f"  Corpus: {len(doc_ids)} docs | Train: {len(train_query_ids)} | Test: {len(test_query_ids)}")

        print("Encoding corpus with SBERT...")
        corpus_emb = sbert.encode(doc_texts, batch_size=256, show_progress_bar=True)
        print("Encoding train queries...")
        train_query_emb = sbert.encode(train_query_texts, batch_size=256, show_progress_bar=True)
        print("Encoding test queries...")
        test_query_emb = sbert.encode(test_query_texts, batch_size=256, show_progress_bar=True)

        for method_name, head_fn in methods.items():
            print(f"\n--- {method_name} ---")
            head = head_fn()

            print("Training projection head...")
            head = train_head(head, corpus_emb, train_query_emb,
                              train_qrels, train_query_ids, doc_ids,
                              epochs=20, lr=1e-3)

            print("Encoding with projection head...")
            corpus_proj = encode_with_head(head, corpus_emb)
            query_proj = encode_with_head(head, test_query_emb)

            print("Retrieving...")
            results = retrieve_with_head(head, corpus_proj, query_proj, doc_ids, test_query_ids)

            evaluator = EvaluateRetrieval()
            ndcg, map_score, recall, precision = evaluator.evaluate(test_qrels, results, [1, 5, 10, 100])

            key = f"{dataset}_{method_name}"
            all_results[key] = {
                "NDCG@10": ndcg["NDCG@10"],
                "Recall@10": recall["Recall@10"],
                "Recall@100": recall["Recall@100"],
                "MAP@10": map_score["MAP@10"],
            }
            print(f"Results: NDCG@10={ndcg['NDCG@10']:.4f}, Recall@10={recall['Recall@10']:.4f}")

    os.makedirs("results", exist_ok=True)
    save_path = "results/manifold_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ 结果已保存至: {save_path}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'NDCG@10':>10} {'Recall@10':>10} {'Recall@100':>10}")
    print("-" * 60)
    for key, metrics in sorted(all_results.items()):
        print(f"{key:<25} {metrics['NDCG@10']:>10.4f} {metrics['Recall@10']:>10.4f} {metrics['Recall@100']:>10.4f}")


if __name__ == "__main__":
    main()