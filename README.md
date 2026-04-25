# Manifold-RAG

Retrieval-Augmented Generation (RAG) fundamentally relies on vector similarity search to ground Large Language Models. However, conventional dense retrieval systems map textual data exclusively into Euclidean spaces, which severely limits their capacity to capture the diverse topological structures—such as strict hierarchies, cyclical relationships, and continuous semantic traits—inherent in complex real-world knowledge bases.

We propose **Manifold-RAG**, a retrieval framework utilizing mixed-curvature product manifolds. Documents and queries are projected into a combined space comprising hyperbolic (Poincaré ball), spherical, and Euclidean components, allowing the model to simultaneously capture tree-like ontologies, mutually exclusive entities, and flat semantic similarities. Manifold-specific features such as embedding norms in hyperbolic space are used to heuristically filter documents by their level of abstraction prior to ranking.

## Dataset

We construct a **hierarchical Wikipedia corpus** (`data/wiki/`) from scratch using 8 diverse seed categories (Biology, Physics, Mathematics, Computer Science, History, Philosophy, Chemistry, Economics). The corpus contains ~1,000 articles organized into a 4-level category tree (category → subcategory → article → section).

Evaluation queries (`data/wiki/queries.jsonl`) are 965 QA pairs of three types designed to probe different aspects of hierarchical retrieval:
- **generalize** — section content → parent category document
- **hierarchy_navigation** — article content → its category document
- **hierarchy_disambiguation** — hierarchy path only (no article name) → correct category among structurally similar ones

The dataset is in [BEIR](https://github.com/beir-cellar/beir) format:
```
data/wiki/
├── corpus.jsonl         # ~1,000 documents with hierarchy metadata
├── queries.jsonl        # 965 QA pairs (with qa_type metadata)
└── qrels/
    ├── train.tsv
    └── test.tsv
```

## Results

Multi-seed evaluation (seeds 42 and 123), std < 0.007 across all methods:

| Method | Overall NDCG@10 | disambiguation NDCG@10 | navigation NDCG@10 | generalize NDCG@10 |
|--------|:-:|:-:|:-:|:-:|
| BM25 | — | — | — | — |
| SBERT (no fine-tuning) | — | — | — | — |
| Euclidean-64 | 0.5777 ± 0.003 | 0.452 ± 0.001 | 0.580 ± 0.002 | 0.611 ± 0.006 |
| Poincaré-64 | 0.4921 ± 0.001 | 0.284 ± 0.003 | 0.488 ± 0.002 | 0.551 ± 0.002 |
| **Product-64** | 0.5417 ± 0.001 | **0.543 ± 0.007** | 0.470 ± 0.003 | 0.565 ± 0.001 |

**Key finding:** Product manifold significantly outperforms Euclidean on `hierarchy_disambiguation` (+9.1 NDCG@10, signal-to-noise ratio 13:1), confirming that mixed-curvature spaces better capture same-level concept separation.

Full results are in `results/manifold_results.json`.

## How to Run

**Requirements:**
```bash
pip install -r requirements.txt
```

**Step 1 — Build the dataset** (optional, `data/wiki/` is already included):
```bash
python wiki_mine.py           # crawl Wikipedia corpus
python generate_hard_qa.py    # generate QA pairs
```

**Step 2 — Run baselines** (fast, ~2 min):
```bash
python experiments/run_baselines.py
```

**Step 3 — Run manifold methods** (trains Euclidean / Poincaré / Product heads):
```bash
python experiments/run_retrieval.py
```

Results are saved to `results/manifold_results.json`.

## Report

LaTeX source: `report/report.tex`  
Compiled PDF: `report/report.pdf`  
Overleaf: https://www.overleaf.com/1269669347tpsgkkrftqpq#5a6f9b

## Contribution

**Boning Shao** — RAG baseline implementation (`baseline/`), evaluation pipeline, per-type metrics

**Tianyi Huang** — Dataset construction (`wiki_mine.py`, `generate_hard_qa.py`, `data/wiki/`), hard negative mining, multi-seed evaluation framework (`experiments/run_retrieval.py`, `experiments/run_baselines.py`)

**Yicheng Peng** — Manifold-RAG pipeline: Poincaré ball embeddings via geoopt, Product manifold head, hyperbolic distance-based retrieval, norm regularization
