"""Generate LaTeX tables from experiment results.

Reads JSON files from results/ and outputs formatted LaTeX tables to report/tables/.
Tables:
  1. Main results table (all datasets, all methods)
  2. Dimension sweep table
  3. Component ablation table
  4. Allocation ablation table
  5. Adaptive vs static comparison
"""

import os
import json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "tables")


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(val, std=None, bold=False):
    """Format a metric value with optional std and bold."""
    if val is None:
        return "---"
    s = f"{val:.3f}"
    if std is not None and std > 0:
        s += f" {{\\scriptsize$\\pm${std:.3f}}}"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def find_best(rows, metric_key="NDCG@10"):
    """Return the method name with the highest mean metric."""
    best_val, best_name = -1, None
    for name, data in rows.items():
        if data and metric_key in data:
            if data[metric_key] > best_val:
                best_val = data[metric_key]
                best_name = name
    return best_name


# ================================================================
# Table 1: Main Results
# ================================================================

def generate_main_table():
    data = load_json("manifold_results.json")
    if not data:
        print("  Skipping main table: manifold_results.json not found")
        return

    datasets = ["wiki", "scifact", "fiqa"]
    methods = [
        ("bm25", "BM25"),
        ("sbert_raw", "SBERT (frozen)"),
        ("euclidean_64", "Euclidean-64"),
        ("poincare_64", "Poincar\\'e-64"),
        ("product_64", "Product-64"),
    ]

    lines = [
        "\\begin{table}[t]",
        "\\caption{Main retrieval results across datasets. Best trained method in \\textbf{bold}.}",
        "\\label{tab:main_results}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{l" + "ccc" * len(datasets) + "}",
        "\\toprule",
    ]

    # Header
    header = " & ".join(["Method"] +
                        [f"\\multicolumn{{3}}{{c}}{{{ds.upper()}}}" for ds in datasets])
    lines.append(header + " \\\\")

    subheader = " & ".join([""] +
                           ["NDCG@10 & R@10 & R@100"] * len(datasets))
    lines.append(subheader + " \\\\")
    lines.append("\\midrule")

    for method_key, method_label in methods:
        row = [method_label]
        for ds in datasets:
            key = f"{ds}_{method_key}"
            if key in data:
                m = data[key]
                # Find best trained method for bolding
                trained = {mk: data.get(f"{ds}_{mk}") for mk, _ in methods[2:]}
                best = find_best(trained)
                is_best = (method_key == best)

                row.append(fmt(m.get("NDCG@10"), m.get("NDCG@10_std"), bold=is_best))
                row.append(fmt(m.get("Recall@10"), m.get("Recall@10_std")))
                row.append(fmt(m.get("Recall@100"), m.get("Recall@100_std")))
            else:
                row.extend(["---"] * 3)
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

    out = os.path.join(TABLES_DIR, "main_results.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {out}")


# ================================================================
# Table 2: Dimension Sweep
# ================================================================

def generate_dimension_sweep_table():
    data = load_json("dimension_sweep_results.json")
    if not data:
        print("  Skipping dimension sweep table: not found")
        return

    dims = [8, 16, 32, 64]
    datasets = ["wiki", "scifact", "fiqa"]
    methods = ["euclidean", "poincare", "product"]
    method_labels = {
        "euclidean": "Euclidean",
        "poincare": "Poincar\\'e",
        "product": "Product",
    }

    lines = [
        "\\begin{table}[t]",
        "\\caption{NDCG@10 across embedding dimensions. Best method per column in \\textbf{bold}.}",
        "\\label{tab:dimension_sweep}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{ll" + "c" * len(dims) + "}",
        "\\toprule",
        "Dataset & Method & " + " & ".join([f"$d={d}$" for d in dims]) + " \\\\",
        "\\midrule",
    ]

    for ds in datasets:
        # Find best method per dim
        best_per_dim = {}
        for d in dims:
            best_val, best_m = -1, None
            for m in methods:
                key = f"{ds}_{m}_{d}"
                if key in data and data[key]["NDCG@10"] > best_val:
                    best_val = data[key]["NDCG@10"]
                    best_m = m
            best_per_dim[d] = best_m

        for i, m in enumerate(methods):
            row_prefix = f"\\multirow{{3}}{{*}}{{{ds.upper()}}}" if i == 0 else ""
            row = [row_prefix, method_labels[m]]
            for d in dims:
                key = f"{ds}_{m}_{d}"
                if key in data:
                    is_best = (best_per_dim[d] == m)
                    row.append(fmt(data[key]["NDCG@10"],
                                   data[key].get("NDCG@10_std"), bold=is_best))
                else:
                    row.append("---")
            lines.append(" & ".join(row) + " \\\\")

        if ds != datasets[-1]:
            lines.append("\\midrule")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

    out = os.path.join(TABLES_DIR, "dimension_sweep.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {out}")


# ================================================================
# Table 3: Component Ablation
# ================================================================

def generate_ablation_component_table():
    data = load_json("ablation_component_results.json")
    if not data:
        print("  Skipping component ablation table: not found")
        return

    variants = [
        ("product_full",   "Full (E32, H16, S16)"),
        ("product_no_H",   "No Hyperbolic (E48, S16)"),
        ("product_no_S",   "No Spherical (E48, H16)"),
        ("product_no_E",   "No Euclidean (H32, S32)"),
        ("euclidean_only", "Euclidean Only (64)"),
        ("poincare_only",  "Poincar\\'e Only (64)"),
    ]

    # Find best
    rows = {}
    for key, _ in variants:
        full_key = f"wiki_{key}"
        rows[key] = data.get(full_key)
    best = find_best(rows)

    lines = [
        "\\begin{table}[t]",
        "\\caption{Component ablation on Wiki dataset (dim=64). Best in \\textbf{bold}.}",
        "\\label{tab:ablation_components}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Variant & NDCG@10 & R@10 & R@100 & MAP@10 \\\\",
        "\\midrule",
    ]

    for key, label in variants:
        m = rows.get(key)
        if m:
            is_best = (key == best)
            row = [label,
                   fmt(m["NDCG@10"], m.get("NDCG@10_std"), bold=is_best),
                   fmt(m["Recall@10"], m.get("Recall@10_std")),
                   fmt(m["Recall@100"], m.get("Recall@100_std")),
                   fmt(m["MAP@10"], m.get("MAP@10_std"))]
        else:
            row = [label] + ["---"] * 4
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

    out = os.path.join(TABLES_DIR, "ablation_components.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {out}")


# ================================================================
# Table 4: Allocation Ablation
# ================================================================

def generate_ablation_allocation_table():
    data = load_json("ablation_allocation_results.json")
    if not data:
        print("  Skipping allocation ablation table: not found")
        return

    allocations = [
        (32, 16, 16, "Default"),
        (16, 32, 16, "Hyper-heavy"),
        (16, 16, 32, "Sphere-heavy"),
        (48, 8, 8,   "Euclid-heavy"),
        (22, 21, 21, "Equal"),
    ]
    datasets = ["wiki", "scifact", "fiqa"]

    lines = [
        "\\begin{table}[t]",
        "\\caption{NDCG@10 for different (E,H,S) dimension allocations (total=64). Best per dataset in \\textbf{bold}.}",
        "\\label{tab:ablation_allocation}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lc" + "c" * len(datasets) + "}",
        "\\toprule",
        "Allocation & (E,H,S) & " + " & ".join([ds.upper() for ds in datasets]) + " \\\\",
        "\\midrule",
    ]

    # Find best per dataset
    best_per_ds = {}
    for ds in datasets:
        best_val, best_name = -1, None
        for e, h, s, _ in allocations:
            key = f"{ds}_product_E{e}_H{h}_S{s}"
            if key in data and data[key]["NDCG@10"] > best_val:
                best_val = data[key]["NDCG@10"]
                best_name = f"{e},{h},{s}"
        best_per_ds[ds] = best_name

    for e, h, s, label in allocations:
        name = f"{e},{h},{s}"
        row = [label, f"({e},{h},{s})"]
        for ds in datasets:
            key = f"{ds}_product_E{e}_H{h}_S{s}"
            if key in data:
                is_best = (best_per_ds[ds] == name)
                row.append(fmt(data[key]["NDCG@10"],
                               data[key].get("NDCG@10_std"), bold=is_best))
            else:
                row.append("---")
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

    out = os.path.join(TABLES_DIR, "ablation_allocation.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {out}")


# ================================================================
# Table 5: Adaptive vs Static
# ================================================================

def generate_adaptive_table():
    data = load_json("adaptive_results.json")
    if not data:
        print("  Skipping adaptive table: not found")
        return

    datasets = ["wiki", "scifact", "fiqa"]
    methods = [
        ("product_64", "Product (static)"),
        ("adaptive_product_64", "Product (adaptive)"),
    ]

    lines = [
        "\\begin{table}[t]",
        "\\caption{Static vs adaptive Product head (dim=64). Best in \\textbf{bold}.}",
        "\\label{tab:adaptive}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{l" + "cc" * len(datasets) + "}",
        "\\toprule",
    ]

    header = " & ".join(["Method"] +
                        [f"\\multicolumn{{2}}{{c}}{{{ds.upper()}}}" for ds in datasets])
    lines.append(header + " \\\\")
    subheader = " & ".join([""] + ["NDCG@10 & R@10"] * len(datasets))
    lines.append(subheader + " \\\\")
    lines.append("\\midrule")

    for method_key, method_label in methods:
        row = [method_label]
        for ds in datasets:
            key = f"{ds}_{method_key}"
            if key in data:
                m = data[key]
                # Find best
                rows_ds = {mk: data.get(f"{ds}_{mk}") for mk, _ in methods}
                best = find_best(rows_ds)
                is_best = (method_key == best)
                row.append(fmt(m["NDCG@10"], m.get("NDCG@10_std"), bold=is_best))
                row.append(fmt(m["Recall@10"], m.get("Recall@10_std")))
            else:
                row.extend(["---"] * 2)
        lines.append(" & ".join(row) + " \\\\")

    # Add per-type breakdown for wiki if available
    wiki_data = {mk: data.get(f"wiki_{mk}") for mk, _ in methods}
    has_per_type = all(v and "per_type" in v for v in wiki_data.values())
    if has_per_type:
        lines.append("\\midrule")
        lines.append("\\multicolumn{" + str(1 + 2*len(datasets)) + "}{l}"
                      "{\\textit{Wiki per-type NDCG@10:}} \\\\")
        all_types = set()
        for v in wiki_data.values():
            if v and "per_type" in v:
                all_types.update(v["per_type"].keys())
        for qt in sorted(all_types):
            row = [f"\\quad {qt}"]
            for ds in datasets:
                if ds != "wiki":
                    row.extend(["", ""])
                    continue
                for mk, _ in methods:
                    m = data.get(f"{ds}_{mk}", {})
                    pt = m.get("per_type", {}).get(qt)
                    if pt:
                        row.append(fmt(pt["NDCG@10"], pt.get("NDCG@10_std")))
                    else:
                        row.append("---")
                row.append("")  # skip R@10 column for per-type
            # Simplified: just show NDCG@10 for each method on wiki
            row_simple = [f"\\quad {qt}"]
            for mk, _ in methods:
                m = data.get(f"wiki_{mk}", {})
                pt = m.get("per_type", {}).get(qt)
                if pt:
                    row_simple.append(fmt(pt["NDCG@10"], pt.get("NDCG@10_std")))
                else:
                    row_simple.append("---")
            # Pad remaining columns
            while len(row_simple) < 1 + 2*len(datasets):
                row_simple.append("")
            lines.append(" & ".join(row_simple) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

    out = os.path.join(TABLES_DIR, "adaptive.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {out}")


# ================================================================

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)

    print("Generating LaTeX tables...")
    generate_main_table()
    generate_dimension_sweep_table()
    generate_ablation_component_table()
    generate_ablation_allocation_table()
    generate_adaptive_table()
    print("\nDone. Tables are in report/tables/")


if __name__ == "__main__":
    main()
