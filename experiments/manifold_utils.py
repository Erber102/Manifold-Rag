"""Shared utilities for Manifold-RAG experiments.

Extracts common model definitions, training, encoding, retrieval, and
evaluation logic from run_retrieval.py so that dimension-sweep, ablation,
and adaptive experiments can reuse them without duplication.
"""

import os
import json
import random as _random
import torch
import geoopt
import numpy as np
from collections import defaultdict
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============ Seed ============

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    _random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        self.scale = torch.nn.Parameter(torch.tensor(0.3))

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
    """Product manifold head supporting zero-dim components for ablation."""

    def __init__(self, input_dim, euclid_dim=32, hyper_dim=16, sphere_dim=16, c=1.0):
        super().__init__()
        self.euclid_dim = euclid_dim
        self.hyper_dim = hyper_dim
        self.sphere_dim = sphere_dim

        if euclid_dim > 0:
            self.proj_e = torch.nn.Linear(input_dim, euclid_dim)
        if hyper_dim > 0:
            self.proj_h = torch.nn.Linear(input_dim, hyper_dim)
            self.ball = geoopt.PoincareBall(c=c)
            self.scale_h = torch.nn.Parameter(torch.tensor(0.3))
        if sphere_dim > 0:
            self.proj_s = torch.nn.Linear(input_dim, sphere_dim)

        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.beta = torch.nn.Parameter(torch.tensor(1.0))
        self.gamma = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        e = self.proj_e(x) if self.euclid_dim > 0 else None
        if e is not None:
            e = e / e.norm(dim=-1, keepdim=True)

        if self.hyper_dim > 0:
            h = self.ball.expmap0(self.proj_h(x) * self.scale_h)
        else:
            h = None

        if self.sphere_dim > 0:
            s = self.proj_s(x)
            s = s / s.norm(dim=-1, keepdim=True)
        else:
            s = None

        return e, h, s

    def distance(self, a, b):
        e_a, h_a, s_a = a
        e_b, h_b, s_b = b
        d = torch.zeros(e_a.shape[0] if e_a is not None else
                        h_a.shape[0] if h_a is not None else
                        s_a.shape[0], device=self._get_device(a))

        if self.euclid_dim > 0 and e_a is not None:
            d = d + self.alpha**2 * ((e_a - e_b) ** 2).sum(dim=-1)
        if self.hyper_dim > 0 and h_a is not None:
            d = d + self.beta**2 * self.ball.dist(h_a, h_b) ** 2
        if self.sphere_dim > 0 and s_a is not None:
            d = d + self.gamma**2 * torch.acos(
                torch.clamp((s_a * s_b).sum(dim=-1), -1+1e-7, 1-1e-7)) ** 2
        return d

    def _get_device(self, tup):
        for x in tup:
            if x is not None:
                return x.device
        return device

    def batch_retrieve(self, query_proj, corpus_proj, device, q_chunk=32, c_chunk=4096):
        q_e, q_h, q_s = query_proj
        c_e, c_h, c_s = corpus_proj

        # Determine n_q, n_c from whichever component exists
        for t in [q_e, q_h, q_s]:
            if t is not None:
                n_q = t.shape[0]
                break
        for t in [c_e, c_h, c_s]:
            if t is not None:
                n_c = t.shape[0]
                break

        total_dist = torch.zeros(n_q, n_c)

        if self.euclid_dim > 0 and q_e is not None:
            c_e_d = c_e.to(device)
            dist_e = torch.zeros(n_q, n_c)
            for i in range(0, n_q, 512):
                j = min(i+512, n_q)
                dist_e[i:j] = (2 - 2 * q_e[i:j].to(device) @ c_e_d.T).cpu()
            total_dist += (self.alpha**2).item() * dist_e

        if self.sphere_dim > 0 and q_s is not None:
            c_s_d = c_s.to(device)
            dist_s = torch.zeros(n_q, n_c)
            for i in range(0, n_q, 512):
                j = min(i+512, n_q)
                dots = torch.clamp(q_s[i:j].to(device) @ c_s_d.T, -1+1e-7, 1-1e-7)
                dist_s[i:j] = (torch.acos(dots) ** 2).cpu()
            total_dist += (self.gamma**2).item() * dist_s

        if self.hyper_dim > 0 and q_h is not None:
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
            total_dist += (self.beta**2).item() * dist_h

        return -total_dist


class AdaptiveProductHead(torch.nn.Module):
    """Product head with query-adaptive distance weights."""

    def __init__(self, input_dim, euclid_dim=32, hyper_dim=16, sphere_dim=16, c=1.0):
        super().__init__()
        self.proj_e = torch.nn.Linear(input_dim, euclid_dim)
        self.proj_h = torch.nn.Linear(input_dim, hyper_dim)
        self.proj_s = torch.nn.Linear(input_dim, sphere_dim)
        self.ball = geoopt.PoincareBall(c=c)
        self.scale_h = torch.nn.Parameter(torch.tensor(0.3))

        # Query-adaptive weight network
        self.weight_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3),
            torch.nn.Softplus(),  # ensure positive weights
        )

    def forward(self, x):
        e = self.proj_e(x)
        e = e / e.norm(dim=-1, keepdim=True)
        h = self.ball.expmap0(self.proj_h(x) * self.scale_h)
        s = self.proj_s(x)
        s = s / s.norm(dim=-1, keepdim=True)
        return e, h, s

    def get_weights(self, query_emb):
        """Input: raw SBERT embedding. Output: (batch, 3) positive weights."""
        return self.weight_net(query_emb)

    def distance(self, a, b, weights=None):
        e_a, h_a, s_a = a
        e_b, h_b, s_b = b
        d_e = ((e_a - e_b) ** 2).sum(dim=-1)
        d_h = self.ball.dist(h_a, h_b) ** 2
        d_s = torch.acos(torch.clamp((s_a * s_b).sum(dim=-1), -1+1e-7, 1-1e-7)) ** 2
        if weights is not None:
            return weights[:, 0] * d_e + weights[:, 1] * d_h + weights[:, 2] * d_s
        return d_e + d_h + d_s

    def batch_retrieve(self, query_proj, corpus_proj, device,
                       query_weights=None, q_chunk=32, c_chunk=4096):
        q_e, q_h, q_s = query_proj
        c_e, c_h, c_s = corpus_proj
        n_q, n_c = q_e.shape[0], c_e.shape[0]

        # Euclidean component distances
        c_e_d = c_e.to(device)
        dist_e = torch.zeros(n_q, n_c)
        for i in range(0, n_q, 512):
            j = min(i+512, n_q)
            dist_e[i:j] = (2 - 2 * q_e[i:j].to(device) @ c_e_d.T).cpu()

        # Spherical component distances
        c_s_d = c_s.to(device)
        dist_s = torch.zeros(n_q, n_c)
        for i in range(0, n_q, 512):
            j = min(i+512, n_q)
            dots = torch.clamp(q_s[i:j].to(device) @ c_s_d.T, -1+1e-7, 1-1e-7)
            dist_s[i:j] = (torch.acos(dots) ** 2).cpu()

        # Hyperbolic component distances
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

        # Apply per-query weights
        if query_weights is not None:
            w = query_weights.cpu()
            total = (w[:, 0:1] * dist_e +
                     w[:, 1:2] * dist_h +
                     w[:, 2:3] * dist_s)
        else:
            total = dist_e + dist_h + dist_s
        return -total


# ============ Hard Negative Index ============

def build_hard_neg_index(corpus: dict, doc_ids: list) -> dict:
    sample = corpus[doc_ids[0]]
    if not sample.get("metadata", {}).get("hierarchy"):
        return {}

    section_to_idxs = defaultdict(list)
    idx_to_top = {}
    cat_idxs = []

    for idx, did in enumerate(doc_ids):
        meta = corpus[did].get("metadata", {})
        hier = meta.get("hierarchy", [])
        dtype = meta.get("type", "")
        if dtype == "category":
            cat_idxs.append(idx)
            continue
        if dtype != "section" or len(hier) < 2:
            continue
        section_key = hier[-1].lower().strip()
        section_to_idxs[section_key].append(idx)
        idx_to_top[idx] = hier[0]

    hard_neg_index = {}
    for idxs in section_to_idxs.values():
        if len(idxs) < 2:
            continue
        for idx in idxs:
            top = idx_to_top.get(idx, "")
            hard = [i for i in idxs if idx_to_top.get(i, "") != top]
            if hard:
                hard_neg_index[idx] = hard

    if len(cat_idxs) >= 2:
        for idx in cat_idxs:
            hard_neg_index[idx] = [i for i in cat_idxs if i != idx]

    print(f"  Hard neg index: {len(hard_neg_index)} docs have hard negatives "
          f"({len(section_to_idxs)} section titles, {len(cat_idxs)} category docs)")
    return hard_neg_index


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


def build_level_targets(doc_ids: list, corpus: dict) -> np.ndarray:
    targets = np.zeros(len(doc_ids), dtype=np.float32)
    for i, did in enumerate(doc_ids):
        meta = corpus[did].get("metadata", {})
        level = meta.get("level", 3)
        dtype = meta.get("type", "")
        if dtype == "category" or level <= 1:
            targets[i] = 0.20
        elif level == 2:
            targets[i] = 0.45
        else:
            targets[i] = 0.70
    return targets


# ============ Evaluation ============

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
        qids = type_to_qids[qa_type]
        sub_qrels = {q: qrels[q] for q in qids if q in qrels}
        sub_res = {q: results[q] for q in qids if q in results}
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


def aggregate_seed_runs(runs: list[dict]) -> dict:
    if not runs:
        return {}
    if len(runs) == 1:
        return runs[0]
    metric_keys = ["NDCG@10", "Recall@10", "Recall@100", "MAP@10"]
    agg: dict = {}
    for k in metric_keys:
        vals = [r[k] for r in runs if k in r]
        if vals:
            agg[k] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
    if all("per_type" in r for r in runs):
        all_types = set().union(*[r["per_type"].keys() for r in runs])
        agg["per_type"] = {}
        for t in all_types:
            tagg: dict = {}
            ns = [r["per_type"][t]["n"] for r in runs if t in r["per_type"]]
            if ns:
                tagg["n"] = ns[0]
            for k in metric_keys:
                vals = [r["per_type"][t][k] for r in runs
                        if t in r["per_type"] and k in r["per_type"][t]]
                if vals:
                    tagg[k] = float(np.mean(vals))
                    tagg[f"{k}_std"] = float(np.std(vals))
            agg["per_type"][t] = tagg
    return agg


# ============ Training ============

def train_head(head, corpus_emb, queries_emb, qrels, query_ids, doc_ids,
               epochs=20, lr=1e-3, n_negatives=7, batch_size=256,
               hard_neg_index=None, n_hard_per_sample=4, margin=0.5,
               level_targets=None, lambda_reg=0.1,
               raw_query_emb=None):
    """Train a projection head.

    raw_query_emb: if provided and head is AdaptiveProductHead, used to
                   compute per-query adaptive weights during training.
    """
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

    is_adaptive = isinstance(head, AdaptiveProductHead)
    if is_adaptive and raw_query_emb is not None:
        raw_queries_t = torch.tensor(raw_query_emb, dtype=torch.float32, device=device)
    else:
        raw_queries_t = None

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

            neg_idx = np.zeros((B, n_negatives), dtype=np.int64)
            for b_i, (_, pi_b) in enumerate(batch):
                negs: list = []
                if hard_neg_index and pi_b in hard_neg_index:
                    pool = hard_neg_index[pi_b]
                    n_take = min(n_hard_per_sample, len(pool))
                    negs = list(np.random.choice(pool, n_take, replace=False))
                while len(negs) < n_negatives:
                    r = int(np.random.randint(0, n_docs))
                    if r != pi_b and r not in negs:
                        negs.append(r)
                neg_idx[b_i] = negs[:n_negatives]
            n_emb = corpus_t[neg_idx.flatten()]

            optimizer.zero_grad()
            q_proj = head(q_emb)
            p_proj = head(p_emb)
            n_proj = head(n_emb)

            if isinstance(q_proj, tuple):
                # Adaptive: compute weights from raw query embeddings
                if is_adaptive and raw_queries_t is not None:
                    weights = head.get_weights(raw_queries_t[q_idx])
                    pos_dist = head.distance(q_proj, p_proj, weights=weights)
                    n_comp = len(q_proj)
                    q_exp = tuple(
                        x.unsqueeze(1).expand(-1, n_negatives, -1).reshape(-1, x.shape[-1])
                        for x in q_proj
                    )
                    n_shaped = tuple(
                        n_proj[k].reshape(B * n_negatives, -1) for k in range(n_comp)
                    )
                    w_exp = weights.unsqueeze(1).expand(-1, n_negatives, -1).reshape(-1, 3)
                    neg_dist = head.distance(q_exp, n_shaped, weights=w_exp).reshape(B, n_negatives)
                else:
                    pos_dist = head.distance(q_proj, p_proj)
                    n_comp = len(q_proj)
                    q_exp = tuple(
                        x.unsqueeze(1).expand(-1, n_negatives, -1).reshape(-1, x.shape[-1])
                        for x in q_proj if x is not None
                    )
                    n_shaped = tuple(
                        n_proj[k].reshape(B * n_negatives, -1)
                        for k in range(n_comp) if n_proj[k] is not None
                    )
                    # Re-wrap with None positions for distance()
                    q_exp_full = []
                    n_shaped_full = []
                    idx = 0
                    for k in range(n_comp):
                        if q_proj[k] is not None:
                            q_exp_full.append(
                                q_proj[k].unsqueeze(1).expand(-1, n_negatives, -1).reshape(-1, q_proj[k].shape[-1]))
                            n_shaped_full.append(n_proj[k].reshape(B * n_negatives, -1))
                        else:
                            q_exp_full.append(None)
                            n_shaped_full.append(None)
                    neg_dist = head.distance(tuple(q_exp_full), tuple(n_shaped_full)).reshape(B, n_negatives)
            else:
                pos_dist = head.distance(q_proj, p_proj)
                d = q_proj.shape[-1]
                q_exp = q_proj.unsqueeze(1).expand(-1, n_negatives, -1).reshape(-1, d)
                n_shaped = n_proj.reshape(B * n_negatives, d)
                neg_dist = head.distance(q_exp, n_shaped).reshape(B, n_negatives)

            loss = torch.clamp(pos_dist.unsqueeze(1) - neg_dist + margin, min=0).mean()

            # Hierarchy norm regularization for Poincaré / Product heads
            if level_targets is not None and hasattr(head, 'ball'):
                p_tgts = torch.tensor(
                    [level_targets[pi_b] for _, pi_b in batch],
                    dtype=torch.float32, device=device,
                )
                if isinstance(p_proj, tuple):
                    h_proj = p_proj[1]
                    if h_proj is not None:
                        p_norms = h_proj.norm(dim=-1)
                        loss = loss + lambda_reg * ((p_norms - p_tgts) ** 2).mean()
                else:
                    p_norms = p_proj.norm(dim=-1)
                    loss = loss + lambda_reg * ((p_norms - p_tgts) ** 2).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
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
            batch = torch.tensor(embeddings[start:start+batch_size],
                                 dtype=torch.float32, device=device)
            proj = head(batch)
            if isinstance(proj, tuple):
                all_proj.append(tuple(x.cpu() if x is not None else None for x in proj))
            else:
                all_proj.append(proj.cpu())

    if isinstance(all_proj[0], tuple):
        n_components = len(all_proj[0])
        result = []
        for i in range(n_components):
            parts = [p[i] for p in all_proj if p[i] is not None]
            result.append(torch.cat(parts) if parts else None)
        return tuple(result)
    else:
        return torch.cat(all_proj)


def retrieve_with_head(head, corpus_proj, query_proj, doc_ids, query_ids,
                       top_k=100, query_weights=None):
    head.eval()
    print("  Computing distances (vectorized)...")
    with torch.no_grad():
        if isinstance(head, AdaptiveProductHead) and query_weights is not None:
            scores = head.batch_retrieve(query_proj, corpus_proj, device,
                                         query_weights=query_weights)
        else:
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


# ============ Data Loading ============

def load_dataset(data_path: str):
    """Load dataset, returning all needed components."""
    corpus, train_queries, train_qrels = GenericDataLoader(data_path).load(split="train")
    _, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")

    doc_ids = list(corpus.keys())
    train_query_ids = list(train_queries.keys())
    test_query_ids = list(test_queries.keys())
    doc_texts = [(corpus[d].get("title", "") + " " + corpus[d]["text"]).strip()
                 for d in doc_ids]
    train_query_texts = [train_queries[q] for q in train_query_ids]
    test_query_texts = [test_queries[q] for q in test_query_ids]

    print(f"  Corpus: {len(doc_ids)} docs | Train: {len(train_query_ids)} | Test: {len(test_query_ids)}")

    return {
        "corpus": corpus,
        "doc_ids": doc_ids,
        "doc_texts": doc_texts,
        "train_query_ids": train_query_ids,
        "train_query_texts": train_query_texts,
        "train_qrels": train_qrels,
        "test_query_ids": test_query_ids,
        "test_query_texts": test_query_texts,
        "test_qrels": test_qrels,
    }


CACHE_DIR = os.path.join("results", "embeddings")


def encode_dataset(sbert, data, dataset_name: str):
    """Encode corpus and queries with SBERT, using .npy cache.

    Cached files are stored in results/embeddings/<dataset>_<split>.npy.
    On subsequent runs the SBERT encode step is skipped entirely.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    corpus_path       = os.path.join(CACHE_DIR, f"{dataset_name}_corpus.npy")
    train_query_path  = os.path.join(CACHE_DIR, f"{dataset_name}_train_queries.npy")
    test_query_path   = os.path.join(CACHE_DIR, f"{dataset_name}_test_queries.npy")

    if os.path.exists(corpus_path):
        print(f"  Loading cached corpus embeddings from {corpus_path}")
        data["corpus_emb"] = np.load(corpus_path)
    else:
        print("  Encoding corpus with SBERT...")
        data["corpus_emb"] = sbert.encode(data["doc_texts"], batch_size=256, show_progress_bar=True)
        np.save(corpus_path, data["corpus_emb"])
        print(f"  Cached corpus embeddings to {corpus_path}")

    if os.path.exists(train_query_path):
        print(f"  Loading cached train query embeddings from {train_query_path}")
        data["train_query_emb"] = np.load(train_query_path)
    else:
        print("  Encoding train queries with SBERT...")
        data["train_query_emb"] = sbert.encode(data["train_query_texts"], batch_size=256, show_progress_bar=True)
        np.save(train_query_path, data["train_query_emb"])
        print(f"  Cached train query embeddings to {train_query_path}")

    if os.path.exists(test_query_path):
        print(f"  Loading cached test query embeddings from {test_query_path}")
        data["test_query_emb"] = np.load(test_query_path)
    else:
        print("  Encoding test queries with SBERT...")
        data["test_query_emb"] = sbert.encode(data["test_query_texts"], batch_size=256, show_progress_bar=True)
        np.save(test_query_path, data["test_query_emb"])
        print(f"  Cached test query embeddings to {test_query_path}")

    return data


def get_training_config(method_name: str) -> dict:
    """Return epochs/lr config based on method name."""
    is_poincare = "poincare" in method_name
    if is_poincare:
        return {"epochs": 50, "lr": 1e-4}
    elif "product" in method_name or "adaptive" in method_name:
        return {"epochs": 30, "lr": 1e-3}
    else:
        return {"epochs": 20, "lr": 1e-3}


def warm_start_head(head, trained_euclidean_head, method_name: str):
    """Copy weights from a trained Euclidean head to initialize another head."""
    if trained_euclidean_head is None:
        return
    with torch.no_grad():
        ew = trained_euclidean_head.proj.weight
        eb = trained_euclidean_head.proj.bias

        if "poincare" in method_name:
            print("  Warm-starting Poincaré from Euclidean weights...")
            head.proj.weight.copy_(ew[:head.proj.out_features])
            head.proj.bias.copy_(eb[:head.proj.out_features])

        elif "product" in method_name or "adaptive" in method_name:
            print("  Warm-starting Product/Adaptive head from Euclidean weights...")
            e_dim = head.proj_e.out_features if hasattr(head, 'proj_e') else 0
            h_dim = head.proj_h.out_features if hasattr(head, 'proj_h') else 0
            s_dim = head.proj_s.out_features if hasattr(head, 'proj_s') else 0

            total_needed = e_dim + h_dim + s_dim
            available = ew.shape[0]

            if e_dim > 0 and e_dim <= available:
                head.proj_e.weight.copy_(ew[:e_dim])
                head.proj_e.bias.copy_(eb[:e_dim])
            if h_dim > 0 and e_dim + h_dim <= available:
                head.proj_h.weight.copy_(ew[e_dim:e_dim+h_dim])
                head.proj_h.bias.copy_(eb[e_dim:e_dim+h_dim])
            if s_dim > 0 and e_dim + h_dim + s_dim <= available:
                head.proj_s.weight.copy_(ew[e_dim+h_dim:e_dim+h_dim+s_dim])
                head.proj_s.bias.copy_(eb[e_dim+h_dim:e_dim+h_dim+s_dim])
