# Abstract
Retrieval-Augmented Generation (RAG) fundamentally relies on vector similarity search to ground Large Language Models. However, conventional dense retrieval systems map textual data exclusively into Euclidean spaces, which severely limits their capacity to capture the diverse topological structures, such as strict hierarchies, cyclical relationships, and continuous semantic traits—inherent in complex real-world knowledge bases. To address this, we propose Manifold-RAG, a novel retrieval framework utilizing mixed-curvature product manifolds. Specifically, we project document chunks and user queries into a combined representation space comprising hyperbolic, spherical, and Euclidean components, allowing the model to simultaneously capture tree-like ontologies, mutually exclusive entities, and flat semantic similarities. Furthermore, we move beyond relying solely on distance metrics by incorporating manifold-specific topological features, such as embedding norms in hyperbolic space—to heuristically filter documents based on their level of abstraction prior to ranking. We anticipate that Manifold-RAG will significantly outperform traditional Euclidean baselines (e.g., BM25 and standard Sentence-BERT) on structurally complex datasets, yielding substantial improvements in Recall@K and Mean Reciprocal Rank (MRR). Additionally, we expect these rich geometric representations to achieve high retrieval accuracy at lower embedding dimensions, preserving computational efficiency. By aligning the representation space's geometry with the underlying topology of the knowledge domain, this work paves the way for more accurate, structurally-aware RAG systems, ultimately mitigating LLM hallucinations in specialized domains.

# Dataset
https://drive.google.com/drive/folders/1Z8zRD4gKy6gkChgl3_MT4VqD-uLpi0i8?usp=sharing

# Contribution
Boning Shao — RAG baseline implementation and evaluation

Tianyi Huang — Dataset collection (Wikipedia category tree with hierarchical labels) and evaluation query-document pair construction

Yicheng Peng — Manifold-RAG pipeline: Poincaré ball embeddings via geoopt, hyperbolic distance-based retrieval, and manifold-specific filtering
