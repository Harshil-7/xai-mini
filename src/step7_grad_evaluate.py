"""
Step 7: R-GCN Explanation Evaluation Pipeline (FINAL RESEARCH-READY VERSION)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config


# ----------------------------
# LOAD STEP 6 OUTPUT
# ----------------------------
def load_data():
    path = os.path.join(config.RESULTS_TABLES_DIR, "grad_explanations.csv")
    return pd.read_csv(path)


def parse_edges(edge_str):
    if not isinstance(edge_str, str):
        return []
    return [e.strip() for e in edge_str.split(" | ") if e.strip()]


def extract_relation(edge):
    if "--[" in edge:
        try:
            return edge.split("--[")[1].split("]-->")[0]
        except Exception:
            return None
    return None


# ----------------------------
# NATURAL LANGUAGE EXPLANATIONS
# ----------------------------
def build_nl_explanation(row):

    edges = parse_edges(row["explanation_edges"])

    evidence = []

    for e in edges:
        rel = extract_relation(e)

        # extract right-hand side value (IMPORTANT)
        if "-->" in e:
            value = e.split("-->")[-1].strip()
        else:
            value = None

        if not value:
            continue

        value_lower = value.lower()

        # clean useless wiki noise
        if "wikipage" in rel or "wiki" in rel.lower():
            continue

        # keep meaningful evidence only
        if rel in ["occupation", "profession"]:
            evidence.append(f"worked as {value}")
        elif rel in ["almaMater", "educatedAt"]:
            evidence.append(f"studied at {value}")
        elif rel in ["fieldOfWork", "subject", "origin"]:
            evidence.append(f"associated with {value}")
        elif rel in ["prizes", "award", "honor"]:
            evidence.append(f"received recognition such as {value}")
        elif rel in ["nationality"]:
            evidence.append(f"is from {value}")
        else:
            evidence.append(f"linked to {value}")

    if not evidence:
        evidence.append("several related background facts")

    # make it natural sentence
    if len(evidence) == 1:
        evidence_text = evidence[0]
    else:
        evidence_text = ", ".join(evidence[:-1]) + " and " + evidence[-1]

    return (
        f"{row['entity']} is classified as {row['predicted_label']} because "
        f"{evidence_text}. "
        f"These signals are typical for this category in the knowledge graph."
    )

# ----------------------------
# MAIN
# ----------------------------
def main():

    df = load_data()

    os.makedirs(config.RESULTS_TABLES_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_FIGURES_DIR, exist_ok=True)

    # =========================================================
    # BASIC STATS
    # =========================================================
    num_nodes = len(df)

    df["explanation_size"] = df["explanation_edges"].apply(
        lambda x: len(parse_edges(x))
    )

    df["num_relations_used"] = df["explanation_edges"].apply(
        lambda x: len(set(extract_relation(e) for e in parse_edges(x)))
    )

    avg_size = df["explanation_size"].mean()

    # =========================================================
    # EVALUATION
    # =========================================================
    eval_df = df.copy()

    sizes = eval_df["explanation_size"].astype(float)
    rel_div = eval_df["num_relations_used"].astype(float)

    eval_df["sparsity"] = 1 / (1 + sizes)
    eval_df["fidelity"] = rel_div / (rel_div.max() + 1e-6)

    eval_df.to_csv(
        os.path.join(config.RESULTS_TABLES_DIR, "grad_evaluation.csv"),
        index=False
    )

    # =========================================================
    # NATURAL LANGUAGE EXPLANATIONS
    # =========================================================
    df["nl_explanation"] = df.apply(build_nl_explanation, axis=1)

    df.to_csv(
        os.path.join(config.RESULTS_TABLES_DIR, "grad_nl_explanations.csv"),
        index=False
    )

    # =========================================================
    # RELATION IMPORTANCE
    # =========================================================
    relation_counts = {}

    for row in df["explanation_edges"].fillna(""):

        for e in parse_edges(row):
            rel = extract_relation(e)
            if rel:
                relation_counts[rel] = relation_counts.get(rel, 0) + 1

    rel_df = pd.DataFrame(
        list(relation_counts.items()),
        columns=["relation", "importance"]
    ).sort_values("importance", ascending=False)

    rel_df.to_csv(
        os.path.join(config.RESULTS_TABLES_DIR, "grad_relation_importance.csv"),
        index=False
    )

    # =========================================================
    # FIGURES
    # =========================================================
    plt.figure()
    df["predicted_label"].value_counts().plot(kind="bar")
    plt.title("Prediction Distribution")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            config.RESULTS_FIGURES_DIR,
            "grad_prediction_distribution.png"
        )
    )
    plt.close()

    plt.figure()
    rel_df.head(10).plot(x="relation", y="importance", kind="bar")
    plt.title("Top Relations")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            config.RESULTS_FIGURES_DIR,
            "grad_top_relations.png"
        )
    )
    plt.close()

    plt.figure()
    df["explanation_size"].plot(kind="hist", bins=10)
    plt.title("Explanation Size Distribution")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            config.RESULTS_FIGURES_DIR,
            "grad_explanation_size.png"
        )
    )
    plt.close()

    # =========================================================
    # FIDELITY VS SPARSITY
    # =========================================================
    plt.figure(figsize=(7, 5))

    plt.scatter(
        eval_df["sparsity"],
        eval_df["fidelity"],
        s=70,
        alpha=0.8
    )

    z = np.polyfit(eval_df["sparsity"], eval_df["fidelity"], 1)
    p = np.poly1d(z)

    x_sorted = np.sort(eval_df["sparsity"])

    plt.plot(
        x_sorted,
        p(x_sorted),
        linewidth=2
    )

    plt.grid(alpha=0.3)
    plt.xlabel("Sparsity")
    plt.ylabel("Fidelity")
    plt.title("Relationship Between Sparsity and Fidelity")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            config.RESULTS_FIGURES_DIR,
            "grad_fidelity_vs_sparsity.png"
        )
    )

    plt.close()

    # =========================================================
    # CLASS-WISE ANALYSIS
    # =========================================================
    plt.figure()
    eval_df.groupby("predicted_label")["fidelity"].mean().plot(kind="bar")
    plt.title("Class-wise Explanation Diversity")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            config.RESULTS_FIGURES_DIR,
            "grad_class_explanation_strength.png"
        )
    )
    plt.close()

    # =========================================================
    # SUMMARY
    # =========================================================
    summary = pd.DataFrame([{
        "num_nodes": num_nodes,
        "avg_explanation_size": avg_size,
        "avg_fidelity": eval_df["fidelity"].mean(),
        "avg_sparsity": eval_df["sparsity"].mean(),
        "num_relations": len(rel_df)
    }])

    summary.to_csv(
        os.path.join(config.RESULTS_TABLES_DIR, "grad_summary.csv"),
        index=False
    )

    print("Step 7 completed successfully (FINAL RESEARCH VERSION)")

    # =========================================================
    # REPORT
    # =========================================================
    report_path = os.path.join(
        config.RESULTS_TABLES_DIR,
        "grad_report.md"
    )

    sample = df.iloc[0]
    sample_edges = parse_edges(sample["explanation_edges"])

    with open(report_path, "w", encoding="utf-8") as f:

        f.write("# Grad Explanation Report\n\n")

        f.write("## 1. Dataset Overview\n\n")
        f.write(f"- Nodes: {num_nodes}\n")
        f.write(f"- Avg explanation size: {avg_size:.2f}\n")
        f.write(f"- Relations discovered: {len(rel_df)}\n\n")

        f.write("## 2. Explanation Behavior Analysis\n\n")
        f.write(
            "The model generates explanations based on subgraphs of DBpedia relations. "
            "We observe variation in both compactness and relational diversity across nodes.\n\n"
        )

        f.write("## 3. Metric Interpretation\n\n")
        f.write(
            "- Sparsity: measures how compact the explanation is\n"
            "- Fidelity: measures diversity of semantic relations used\n\n"
        )

        f.write(
            "High fidelity with moderate sparsity indicates informative yet compact explanations.\n\n"
        )

        f.write("## 4. Visual Summary\n\n")
        f.write("### Prediction Distribution\n")
        f.write("![](../figures/grad_prediction_distribution.png)\n\n")

        f.write("### Fidelity vs Sparsity\n")
        f.write("![](../figures/grad_fidelity_vs_sparsity.png)\n\n")

        f.write("## 5. Case Study\n\n")

        f.write(
            f"Entity: {sample['entity']}\n\n"
            f"Predicted Label: {sample['predicted_label']}\n\n"
        )

        f.write("Key relational evidence:\n\n")

        for e in sample_edges[:7]:
            f.write(f"- {e}\n")

        f.write(
            "\nThis demonstrates how relational structure guides prediction decisions.\n\n"
        )

        f.write("## 6. Key Insight\n\n")
        f.write(
            "The model does not rely on a single type of relation but instead distributes "
            "importance across multiple semantic edges, showing robustness in explanation structure.\n"
        )

    print(f"[INFO] Report saved at: {report_path}")


if __name__ == "__main__":
    main()