"""Generate all visualizations for the Manifold-RAG report.

Reads JSON result files from results/ and produces PDF figures:
  (a) Dimension sweep line plot      → results/dimension_sweep.pdf
  (b) Poincaré disk visualization    → results/poincare_visualization.pdf
  (c) Ablation component bar chart   → results/ablation_components.pdf
  (d) Adaptive weight analysis       → results/adaptive_weights_by_type.pdf
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ================================================================
# (a) Dimension Sweep Line Plot
# ================================================================

def plot_dimension_sweep():
    path = os.path.join(RESULTS_DIR, "dimension_sweep_results.json")
    if not os.path.exists(path):
        print(f"  Skipping dimension sweep plot: {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    dims = [8, 16, 32, 64]
    datasets = ["scifact", "fiqa", "wiki"]
    methods = ["euclidean", "poincare", "product"]
    colors = {"euclidean": "#2196F3", "poincare": "#F44336", "product": "#4CAF50"}
    markers = {"euclidean": "o", "poincare": "s", "product": "D"}
    labels = {"euclidean": "Euclidean", "poincare": "Poincaré", "product": "Product"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax_idx, dataset in enumerate(datasets):
        ax = axes[ax_idx]
        for method in methods:
            means, stds = [], []
            for d in dims:
                key = f"{dataset}_{method}_{d}"
                if key in data:
                    means.append(data[key]["NDCG@10"])
                    stds.append(data[key].get("NDCG@10_std", 0))
                else:
                    means.append(np.nan)
                    stds.append(0)

            means = np.array(means)
            stds = np.array(stds)
            ax.errorbar(dims, means, yerr=stds,
                        label=labels[method], color=colors[method],
                        marker=markers[method], linewidth=2, markersize=7,
                        capsize=3)

        ax.set_title(dataset.upper(), fontweight="bold")
        ax.set_xlabel("Embedding Dimension")
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        if ax_idx == 0:
            ax.set_ylabel("NDCG@10")
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Dimension Sweep: NDCG@10 vs Embedding Dimension", fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "dimension_sweep.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"  Saved {out}")
    plt.close(fig)


# ================================================================
# (b) Poincaré Disk Embedding Visualization
# ================================================================

def plot_poincare_disk():
    """Visualize Poincaré embeddings of wiki corpus on a 2D disk."""
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        import torch
        import geoopt
        from manifold_utils import (
            PoincareHead, set_seed, device,
            load_dataset, encode_dataset,
            build_hard_neg_index, build_level_targets,
            train_head, encode_with_head,
        )
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"  Skipping Poincaré visualization: {e}")
        return

    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    input_dim = sbert.get_sentence_embedding_dimension()

    data_path = os.path.join("data", "wiki")
    data = load_dataset(data_path)
    data = encode_dataset(sbert, data, "wiki")

    hard_neg_index = build_hard_neg_index(data["corpus"], data["doc_ids"])
    level_targets = build_level_targets(data["doc_ids"], data["corpus"])

    set_seed(42)
    head = PoincareHead(input_dim, 64)
    head = train_head(
        head, data["corpus_emb"], data["train_query_emb"],
        data["train_qrels"], data["train_query_ids"], data["doc_ids"],
        epochs=50, lr=1e-4,
        hard_neg_index=hard_neg_index, n_hard_per_sample=4, margin=0.5,
        level_targets=level_targets, lambda_reg=0.2,
    )

    corpus_proj = encode_with_head(head, data["corpus_emb"])  # (N, 64)

    # PCA to 2D
    from sklearn.decomposition import PCA
    emb_np = corpus_proj.numpy()
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb_np)

    # Project onto Poincaré disk (scale to fit inside unit disk)
    norms = np.linalg.norm(emb_2d, axis=1, keepdims=True)
    max_norm = norms.max()
    if max_norm > 0:
        emb_2d = emb_2d / (max_norm * 1.05)  # small margin inside disk

    # Get top-level category for each doc
    categories = []
    for did in data["doc_ids"]:
        meta = data["corpus"][did].get("metadata", {})
        hier = meta.get("hierarchy", [])
        categories.append(hier[0] if hier else "Unknown")

    unique_cats = sorted(set(categories))
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    cat_indices = [cat_to_idx[c] for c in categories]

    # Get levels for marker sizing
    levels = []
    for did in data["doc_ids"]:
        meta = data["corpus"][did].get("metadata", {})
        levels.append(meta.get("level", 3))
    levels = np.array(levels)
    sizes = np.where(levels <= 1, 60, np.where(levels == 2, 30, 12))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit disk boundary
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1.5, alpha=0.3)

    cmap = plt.cm.Set2 if len(unique_cats) <= 8 else plt.cm.tab20
    for i, cat in enumerate(unique_cats):
        mask = np.array(cat_indices) == i
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   s=sizes[mask], c=[cmap(i / max(len(unique_cats)-1, 1))]*mask.sum(),
                   label=cat, alpha=0.7, edgecolors="none")

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_title("Poincaré Disk Embeddings (Wiki Corpus, PCA to 2D)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.8)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    # Add note about point sizes
    ax.text(0.98, 0.02, "Size: category > article > section",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            style="italic", alpha=0.6)

    out = os.path.join(RESULTS_DIR, "poincare_visualization.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"  Saved {out}")
    plt.close(fig)


# ================================================================
# (c) Ablation Component Bar Chart
# ================================================================

def plot_ablation_components():
    path = os.path.join(RESULTS_DIR, "ablation_component_results.json")
    if not os.path.exists(path):
        print(f"  Skipping ablation components plot: {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    variants = [
        ("product_full",   "Full\n(E32,H16,S16)"),
        ("product_no_H",   "No Hyper\n(E48,S16)"),
        ("product_no_S",   "No Sphere\n(E48,H16)"),
        ("product_no_E",   "No Euclid\n(H32,S32)"),
        ("euclidean_only", "Euclidean\nOnly (64)"),
        ("poincare_only",  "Poincaré\nOnly (64)"),
    ]

    means, stds, labels = [], [], []
    for key, label in variants:
        full_key = f"wiki_{key}"
        if full_key in data:
            means.append(data[full_key]["NDCG@10"])
            stds.append(data[full_key].get("NDCG@10_std", 0))
        else:
            means.append(0)
            stds.append(0)
        labels.append(label)

    colors = ["#4CAF50", "#FF9800", "#FF9800", "#FF9800", "#2196F3", "#F44336"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=colors, edgecolor="white", linewidth=1.5, alpha=0.85)

    # Add value annotations
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("NDCG@10")
    ax.set_title("Component Ablation (Wiki Dataset, dim=64)", fontweight="bold")
    ax.set_ylim(0, max(means) * 1.15)

    out = os.path.join(RESULTS_DIR, "ablation_components.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"  Saved {out}")
    plt.close(fig)


# ================================================================
# (d) Adaptive Weights by QA Type
# ================================================================

def plot_adaptive_weights():
    path = os.path.join(RESULTS_DIR, "adaptive_weights.json")
    if not os.path.exists(path):
        print(f"  Skipping adaptive weights plot: {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    if "wiki" not in data:
        print("  Skipping adaptive weights plot: no wiki data")
        return

    wiki = data["wiki"]
    qa_types = sorted(wiki.keys())

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(qa_types))
    width = 0.25

    alphas = [wiki[qt]["mean_alpha"] for qt in qa_types]
    betas = [wiki[qt]["mean_beta"] for qt in qa_types]
    gammas = [wiki[qt]["mean_gamma"] for qt in qa_types]
    alpha_std = [wiki[qt]["std_alpha"] for qt in qa_types]
    beta_std = [wiki[qt]["std_beta"] for qt in qa_types]
    gamma_std = [wiki[qt]["std_gamma"] for qt in qa_types]

    ax.bar(x - width, alphas, width, yerr=alpha_std, label=r"$\alpha$ (Euclidean)",
           color="#2196F3", capsize=3, alpha=0.85)
    ax.bar(x, betas, width, yerr=beta_std, label=r"$\beta$ (Hyperbolic)",
           color="#F44336", capsize=3, alpha=0.85)
    ax.bar(x + width, gammas, width, yerr=gamma_std, label=r"$\gamma$ (Spherical)",
           color="#FF9800", capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([qt.replace("_", "\n") for qt in qa_types], fontsize=9)
    ax.set_ylabel("Learned Weight")
    ax.set_title("Adaptive Distance Weights by QA Type (Wiki Dataset)", fontweight="bold")
    ax.legend()

    out = os.path.join(RESULTS_DIR, "adaptive_weights_by_type.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"  Saved {out}")
    plt.close(fig)


# ================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Generating plots...")
    print("\n(a) Dimension sweep line plot")
    plot_dimension_sweep()

    print("\n(b) Poincaré disk visualization")
    plot_poincare_disk()

    print("\n(c) Ablation component bar chart")
    plot_ablation_components()

    print("\n(d) Adaptive weights by QA type")
    plot_adaptive_weights()

    print("\nDone.")


if __name__ == "__main__":
    main()
