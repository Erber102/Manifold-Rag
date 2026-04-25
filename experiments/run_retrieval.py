# 本地运行加速版
import os
import json
import torch
import geoopt
import numpy as np
from collections import defaultdict
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")

# Seeds to run sequentially.  Multi-seed lets us check whether per-method
# differences (especially Product winning on disambiguation) are real or noise.
SEEDS = [42, 123]


def set_seed(seed: int) -> None:
    import random as _random
    torch.manual_seed(seed)
    np.random.seed(seed)
    _random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggregate_seed_runs(runs: list[dict]) -> dict:
    """Mean (and std) across per-seed result dicts.  Preserves per_type."""
    if not runs:
        return {}
    if len(runs) == 1:
        return runs[0]
    metric_keys = ["NDCG@10", "Recall@10", "Recall@100", "MAP@10"]
    agg: dict = {}
    for k in metric_keys:
        vals = [r[k] for r in runs if k in r]
        if vals:
            agg[k]            = float(np.mean(vals))
            agg[f"{k}_std"]   = float(np.std(vals))
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
                    tagg[k]          = float(np.mean(vals))
                    tagg[f"{k}_std"] = float(np.std(vals))
            agg["per_type"][t] = tagg
    return agg

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
    def __init__(self, input_dim, euclid_dim=32, hyper_dim=16, sphere_dim=16, c=1.0):
        super().__init__()
        self.proj_e = torch.nn.Linear(input_dim, euclid_dim)
        self.proj_h = torch.nn.Linear(input_dim, hyper_dim)
        self.proj_s = torch.nn.Linear(input_dim, sphere_dim)
        self.ball = geoopt.PoincareBall(c=c)
        self.scale_h = torch.nn.Parameter(torch.tensor(0.3))
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


# ============ Hard Negative Index ============

def build_hard_neg_index(corpus: dict, doc_ids: list) -> dict:
    """
    Group section chunks by their section title (= last element of hierarchy).
    For each chunk, hard negatives = chunks with the same section title but
    a different top-level hierarchy category.
    Also: category docs across different top-level categories are mutual hard
    negatives (natural for hierarchy_navigation QA).
    Returns {} when corpus has no hierarchy metadata.
    """
    from collections import defaultdict

    sample = corpus[doc_ids[0]]
    if not sample.get("metadata", {}).get("hierarchy"):
        return {}

    section_to_idxs = defaultdict(list)
    idx_to_top = {}
    cat_idxs = []

    for idx, did in enumerate(doc_ids):
        meta  = corpus[did].get("metadata", {})
        hier  = meta.get("hierarchy", [])
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

    # Section-level hard negatives (same title, different branch)
    for idxs in section_to_idxs.values():
        if len(idxs) < 2:
            continue
        for idx in idxs:
            top  = idx_to_top.get(idx, "")
            hard = [i for i in idxs if idx_to_top.get(i, "") != top]
            if hard:
                hard_neg_index[idx] = hard

    # Category-level hard negatives (different category docs)
    if len(cat_idxs) >= 2:
        for idx in cat_idxs:
            hard_neg_index[idx] = [i for i in cat_idxs if i != idx]

    print(f"  Hard neg index: {len(hard_neg_index)} docs have hard negatives "
          f"({len(section_to_idxs)} section titles, {len(cat_idxs)} category docs)")
    return hard_neg_index


def load_query_types(data_path: str) -> dict:
    """Read qa_type from queries.jsonl metadata, indexed by query _id."""
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
    """Evaluate results and break down metrics by qa_type. Returns a result dict."""
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


def build_level_targets(doc_ids: list, corpus: dict) -> np.ndarray:
    """
    Target Poincaré norm for each corpus doc based on hierarchy level.
    level=1 (category) → 0.20  (near origin = abstract)
    level=2 (article)  → 0.45
    level>=3 (section) → 0.70  (near boundary = specific)
    """
    targets = np.zeros(len(doc_ids), dtype=np.float32)
    for i, did in enumerate(doc_ids):
        meta  = corpus[did].get("metadata", {})
        level = meta.get("level", 3)
        dtype = meta.get("type", "")
        if dtype == "category" or level <= 1:
            targets[i] = 0.20
        elif level == 2:
            targets[i] = 0.45
        else:
            targets[i] = 0.70
    return targets


# ============ Training ============

def train_head(head, corpus_emb, queries_emb, qrels, query_ids, doc_ids,
               epochs=20, lr=1e-3, n_negatives=7, batch_size=256,
               hard_neg_index=None, n_hard_per_sample=4, margin=0.5,
               level_targets=None, lambda_reg=0.1):
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

            neg_idx = np.zeros((B, n_negatives), dtype=np.int64)
            for b_i, (_, pi_b) in enumerate(batch):
                negs: list = []
                if hard_neg_index and pi_b in hard_neg_index:
                    pool   = hard_neg_index[pi_b]
                    n_take = min(n_hard_per_sample, len(pool))
                    negs   = list(np.random.choice(pool, n_take, replace=False))
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

            loss = torch.clamp(pos_dist.unsqueeze(1) - neg_dist + margin, min=0).mean()

            # Hierarchy norm regularization for Poincaré / Product heads
            if level_targets is not None and hasattr(head, 'ball'):
                p_tgts = torch.tensor(
                    [level_targets[pi_b] for _, pi_b in batch],
                    dtype=torch.float32, device=device,
                )
                if isinstance(p_proj, tuple):
                    # ProductHead: regularize hyperbolic component (index 1)
                    p_norms = p_proj[1].norm(dim=-1)
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

    datasets = ["wiki"]
    methods = {
        "euclidean_64": lambda: EuclideanHead(input_dim, 64),
        "poincare_64": lambda: PoincareHead(input_dim, 64),
        "product_64": lambda: ProductHead(input_dim, 32, 16, 16),
    }

    all_results = {}
    base_data_path = "data"

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

        print("Building hard negative index...")
        hard_neg_index = build_hard_neg_index(corpus, doc_ids)
        level_targets = build_level_targets(doc_ids, corpus)
        query_types  = load_query_types(data_path)
        type_counts  = {}
        for qid in test_query_ids:
            t = query_types.get(qid, "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"  Test QA breakdown by type: {type_counts}")
        print(f"  Level target distribution: "
              f"cat={( level_targets < 0.3).sum()}, "
              f"art={(( level_targets >= 0.3) & (level_targets < 0.6)).sum()}, "
              f"sec={(level_targets >= 0.6).sum()}")

        print("Encoding corpus with SBERT...")
        corpus_emb = sbert.encode(doc_texts, batch_size=256, show_progress_bar=True)
        print("Encoding train queries...")
        train_query_emb = sbert.encode(train_query_texts, batch_size=256, show_progress_bar=True)
        print("Encoding test queries...")
        test_query_emb = sbert.encode(test_query_texts, batch_size=256, show_progress_bar=True)

        # Track per-seed runs so we can aggregate at the end of the dataset.
        per_seed_runs: dict[str, list[dict]] = {m: [] for m in methods}

        for seed in SEEDS:
            print(f"\n{'#'*60}")
            print(f"# SEED = {seed}")
            print(f"{'#'*60}")
            set_seed(seed)

            trained_euclidean_head = None

            for method_name, head_fn in methods.items():
                print(f"\n--- {method_name} (seed={seed}) ---")
                head = head_fn()
                is_poincare = "poincare" in method_name

                # Warm-start: copy Euclidean proj weights into Poincaré / Product
                if trained_euclidean_head is not None:
                    with torch.no_grad():
                        ew = trained_euclidean_head.proj.weight  # (64, 384)
                        eb = trained_euclidean_head.proj.bias    # (64,)

                        if is_poincare:
                            print("Warm-starting Poincaré from Euclidean weights...")
                            head.proj.weight.copy_(ew)
                            head.proj.bias.copy_(eb)

                        elif "product" in method_name:
                            print("Warm-starting Product head from Euclidean weights...")
                            e = head.proj_e.out_features
                            h = head.proj_h.out_features
                            s = head.proj_s.out_features
                            head.proj_e.weight.copy_(ew[:e])
                            head.proj_e.bias.copy_(eb[:e])
                            head.proj_h.weight.copy_(ew[e:e+h])
                            head.proj_h.bias.copy_(eb[e:e+h])
                            head.proj_s.weight.copy_(ew[e+h:e+h+s])
                            head.proj_s.bias.copy_(eb[e+h:e+h+s])

                print("Training projection head...")
                use_norm_reg = is_poincare or "product" in method_name
                if is_poincare:
                    n_epochs = 50
                elif "product" in method_name:
                    n_epochs = 30
                else:
                    n_epochs = 20
                head = train_head(head, corpus_emb, train_query_emb,
                                  train_qrels, train_query_ids, doc_ids,
                                  epochs=n_epochs,
                                  lr=1e-4 if is_poincare else 1e-3,
                                  hard_neg_index=hard_neg_index,
                                  n_hard_per_sample=4,
                                  margin=0.5,
                                  level_targets=level_targets if use_norm_reg else None,
                                  lambda_reg=0.2)

                if method_name == "euclidean_64":
                    trained_euclidean_head = head

                print("Encoding with projection head...")
                corpus_proj = encode_with_head(head, corpus_emb)
                query_proj = encode_with_head(head, test_query_emb)

                print("Retrieving...")
                results = retrieve_with_head(head, corpus_proj, query_proj, doc_ids, test_query_ids)

                seed_key = f"{dataset}_{method_name}_seed{seed}"
                run_result = eval_with_types(results, test_qrels, test_query_ids, query_types)
                print(f"  Overall: NDCG@10={run_result['NDCG@10']:.4f}, Recall@10={run_result['Recall@10']:.4f}")
                all_results[seed_key] = run_result
                per_seed_runs[method_name].append(run_result)

        # ── Aggregate across seeds ───────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"AGGREGATED across seeds {SEEDS} for dataset {dataset}")
        print(f"{'='*60}")
        for method_name, runs in per_seed_runs.items():
            agg_key = f"{dataset}_{method_name}"
            all_results[agg_key] = aggregate_seed_runs(runs)
            agg = all_results[agg_key]
            print(f"  {method_name:<15}: NDCG@10={agg['NDCG@10']:.4f}"
                  f" ± {agg.get('NDCG@10_std', 0):.4f}, "
                  f"R@10={agg['Recall@10']:.4f} ± {agg.get('Recall@10_std', 0):.4f}")

    os.makedirs("results", exist_ok=True)
    save_path = "results/manifold_results.json"
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results = existing
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ 结果已保存至: {save_path}")

    print(f"\n{'='*70}")
    print("SUMMARY (aggregated across seeds; ± = std)")
    print(f"{'='*70}")
    print(f"{'Method':<28} {'NDCG@10':>16} {'Recall@10':>16} {'Recall@100':>16}")
    print("-" * 76)
    # Only show aggregated (un-suffixed) keys, not per-seed entries
    agg_keys = [k for k in sorted(all_results.keys())
                if "_seed" not in k and "per_type" in all_results[k]]
    for key in agg_keys:
        m = all_results[key]
        ndcg_s = f"{m['NDCG@10']:.4f}±{m.get('NDCG@10_std', 0):.3f}"
        r10_s  = f"{m['Recall@10']:.4f}±{m.get('Recall@10_std', 0):.3f}"
        r100_s = f"{m['Recall@100']:.4f}±{m.get('Recall@100_std', 0):.3f}"
        print(f"{key:<28} {ndcg_s:>16} {r10_s:>16} {r100_s:>16}")

    # Cross-method per-type comparison (aggregated only)
    print(f"\n{'='*80}")
    print("PER-QA-TYPE NDCG@10 mean±std (rows = qa_type, cols = method)")
    print(f"{'='*80}")
    if agg_keys:
        all_types = sorted({t for k in agg_keys for t in all_results[k]["per_type"].keys()})
        col_w = 22
        header = f"{'qa_type':<28}" + "".join(f"{k.rsplit('_',1)[-1]:>{col_w}}" for k in agg_keys)
        print(header)
        print("-" * len(header))
        for t in all_types:
            row = f"{t:<28}"
            for k in agg_keys:
                pt = all_results[k]["per_type"].get(t)
                if pt:
                    cell = f"{pt['NDCG@10']:.4f}±{pt.get('NDCG@10_std', 0):.3f}"
                else:
                    cell = "-"
                row += f"{cell:>{col_w}}"
            print(row)


if __name__ == "__main__":
    main()