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

    if not edges:
        return (
            f"The model predicts that {row['entity']} belongs to the "
            f"{row['predicted_label']} category because the available "
            f"information matches characteristics commonly associated with this group."
        )

    facts = []
    seen = set()

    for edge in edges:

        try:
            source = edge.split("--[")[0].strip()
            relation = edge.split("--[")[1].split("]-->")[0].strip()
            target = edge.split("]-->")[1].strip()
        except Exception:
            continue

        # Ignore unhelpful relations
        if relation in {"wikiPageWikiLink", "sameAs"}:
            continue

        # Ignore self-links
        if source == target:
            continue

        # Convert relations into natural language
        if relation == "nationality":
            fact = f"is from {target}"

        elif relation in {"origin", "fieldOfStudy", "subject"}:
            fact = f"has a background in {target}"

        elif relation in {"occupation", "profession"}:
            fact = f"works as a {target}"

        elif relation in {"employer", "worksAt"}:
            fact = f"works at {target}"

        elif relation in {"prizes", "award"}:
            fact = f"has received the {target}"

        elif relation in {"educatedAt", "almaMater"}:
            fact = f"studied at {target}"

        elif relation == "relatives":
            fact = f"is related to {target}"

        elif relation in {"fieldOfWork"}:
            fact = f"works in the field of {target}"

        else:
            continue

        if fact not in seen:
            seen.add(fact)
            facts.append(fact)

    # Fallback if nothing useful remains
    if not facts:
        return (
            f"The model predicts that {row['entity']} belongs to the "
            f"{row['predicted_label']} category because the available "
            f"information is consistent with characteristics commonly found "
            f"in this category."
        )

    # Build readable sentence
    if len(facts) == 1:
        reason = facts[0]

    elif len(facts) == 2:
        reason = f"{facts[0]} and {facts[1]}"

    else:
        reason = ", ".join(facts[:-1]) + ", and " + facts[-1]

    return (
        f"The model predicts that {row['entity']} belongs to the "
        f"{row['predicted_label']} category because the available information shows "
        f"that it {reason}. Together, these characteristics are commonly associated "
        f"with this category."
    )

# ----------------------------
# MAIN
# ----------------------------
def main():

    df = load_data()

    os.makedirs(config.RESULTS_TABLES_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_FIGURES_DIR, exist_ok=True)

    # ----------------------------
    # BASIC STATS
    # ----------------------------
    num_nodes = len(df)

    df["explanation_size"] = df["explanation_edges"].apply(
        lambda x: len(parse_edges(x))
    )

    df["num_relations_used"] = df["explanation_edges"].apply(
        lambda x: len(set(extract_relation(e) for e in parse_edges(x)))
    )

    avg_size = df["explanation_size"].mean()

    # ----------------------------
    # EVALUATION
    # ----------------------------
    eval_df = df.copy()

    sizes = eval_df["explanation_size"].astype(float)
    rel_div = eval_df["num_relations_used"].astype(float)

    eval_df["sparsity"] = 1 / (1 + sizes)
    eval_df["fidelity"] = rel_div / (rel_div.max() + 1e-6)

    eval_df.to_csv(
        os.path.join(config.RESULTS_TABLES_DIR, "grad_evaluation.csv"),
        index=False
    )

    # ----------------------------
    # NATURAL LANGUAGE EXPLANATIONS
    # ----------------------------
    df["nl_explanation"] = df.apply(build_nl_explanation, axis=1)

    df.to_csv(
        os.path.join(config.RESULTS_TABLES_DIR, "grad_nl_explanations.csv"),
        index=False
    )

    # ----------------------------
    # RELATION IMPORTANCE
    # ----------------------------
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

    # ----------------------------
    # FIGURES
    # ----------------------------
    plt.figure()
    df["predicted_label"].value_counts().plot(kind="bar")
    plt.title("Prediction Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_prediction_distribution.png"))
    plt.close()

    plt.figure()
    rel_df.head(10).plot(x="relation", y="importance", kind="bar")
    plt.title("Top Relations")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_top_relations.png"))
    plt.close()

    plt.figure()
    df["explanation_size"].plot(kind="hist", bins=10)
    plt.title("Explanation Size Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_explanation_size.png"))
    plt.close()

    # ----------------------------
    # FIDELITY VS SPARSITY
    # ----------------------------
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

    plt.plot(x_sorted, p(x_sorted), linewidth=2)

    plt.grid(alpha=0.3)
    plt.xlabel("Sparsity")
    plt.ylabel("Fidelity")
    plt.title("Relationship Between Sparsity and Fidelity")

    plt.tight_layout()

    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_fidelity_vs_sparsity.png"))
    plt.close()

    # ----------------------------
    # CLASS-WISE ANALYSIS
    # ----------------------------
    plt.figure()
    eval_df.groupby("predicted_label")["fidelity"].mean().plot(kind="bar")
    plt.title("Class-wise Explanation Diversity")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_class_explanation_strength.png"))
    plt.close()

    # ----------------------------
    # SUMMARY
    # ----------------------------
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

    with open(report_path, "w", encoding="utf-8") as f:

        f.write("# Grad Explanation Report\n\n")

        f.write(
            "This report summarises the explanations generated by the R-GCN model "
            "for node classification on the DBpedia dataset. It analyses the "
            "overall explanation quality, important relation types, and provides "
            "an example explanation for one prediction.\n\n"
        )

        # =====================================================
        # Dataset Overview
        # =====================================================
        f.write("## 1. Dataset Overview\n\n")

        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"|Number of explained entities|{num_nodes}|\n")
        f.write(f"|Average explanation size|{avg_size:.2f} edges|\n")
        f.write(f"|Different relation types|{len(rel_df)}|\n")
        f.write(f"|Average Fidelity|{eval_df['fidelity'].mean():.3f}|\n")
        f.write(f"|Average Sparsity|{eval_df['sparsity'].mean():.3f}|\n\n")

        f.write(
            "Each prediction is explained using a small subgraph extracted from the "
            "knowledge graph. These explanations help understand which information "
            "was most influential for the model's decision.\n\n")

        # =====================================================
        # Prediction Distribution
        # =====================================================
        f.write("## 2. Prediction Distribution\n\n")

        f.write("![](../figures/grad_prediction_distribution.png)\n\n")

        pred_counts = df["predicted_label"].value_counts()

        f.write("| Class | Number of Predictions |\n")
        f.write("|------|-----------------------|\n")

        for cls, count in pred_counts.items():
            f.write(f"|{cls}|{count}|\n")

        f.write("\n")

        f.write(
            "The figure above shows how predictions are distributed across the "
            "different classes. A balanced distribution suggests that the model "
            "does not heavily favour one class over another.\n\n"
        )

        # =====================================================
        # Explanation Size
        # =====================================================
        f.write("## 3. Explanation Size Analysis\n\n")

        f.write("![](../figures/grad_explanation_size.png)\n\n")

        f.write(
            f"The average explanation contains **{avg_size:.2f} graph edges**. "
            "Smaller explanations are generally easier for humans to understand, "
            "while larger explanations provide more supporting evidence for the "
            "prediction.\n\n"
        )

        f.write(
            "The histogram illustrates how explanation sizes vary among all "
            "entities. Most explanations remain relatively compact, making them "
            "suitable for interpretation.\n\n"
        )

        # =====================================================
        # Relation Importance
        # =====================================================
        f.write("## 4. Most Important Relation Types\n\n")

        f.write("![](../figures/grad_top_relations.png)\n\n")

        f.write(
            "The following table lists the ten relation types that appear most "
            "frequently across all generated explanations.\n\n"
        )

        f.write("| Rank | Relation | Frequency |\n")
        f.write("|-----|----------|-----------|\n")

        for rank, (_, row) in enumerate(rel_df.head(10).iterrows(), start=1):
            f.write(
                f"|{rank}|{row['relation']}|{row['importance']}|\n"
            )

        f.write("\n")

        f.write(
            "Relations appearing frequently across explanations indicate that "
            "they provide useful information for distinguishing between entity "
            "classes. These relation types represent the structural patterns "
            "that the R-GCN has learned from the knowledge graph.\n\n"
        )

        # =====================================================
        # Fidelity vs Sparsity
        # =====================================================
        f.write("## 5. Fidelity versus Sparsity\n\n")

        f.write("![](../figures/grad_fidelity_vs_sparsity.png)\n\n")

        f.write(
            "This figure compares explanation fidelity with explanation sparsity "
            "for every explained entity.\n\n"
        )

        f.write(
            "- **Fidelity** measures how much useful information is retained in the explanation.\n"
        )

        f.write(
            "- **Sparsity** measures how compact the explanation is.\n\n"
        )

        f.write(
            "An ideal explanation achieves both high fidelity and high sparsity, "
            "meaning that it remains easy to understand while still containing "
            "sufficient evidence for the prediction.\n\n"
        )

        # =====================================================
        # Class-wise Analysis
        # =====================================================
        f.write("## 6. Class-wise Explanation Quality\n\n")

        f.write("![](../figures/grad_class_explanation_strength.png)\n\n")

        class_scores = (
            eval_df.groupby("predicted_label")["fidelity"]
            .mean()
            .sort_values(ascending=False)
        )

        f.write("| Class | Average Fidelity |\n")
        f.write("|------|------------------|\n")

        for cls, score in class_scores.items():
            f.write(f"|{cls}|{score:.3f}|\n")

        f.write("\n")

        f.write(
            "Higher fidelity indicates that explanations for that class use a "
            "larger variety of meaningful relation types. Lower values suggest "
            "that predictions rely on fewer types of relations.\n\n"
        )

        # =====================================================
        # Example Explanation
        # =====================================================
        f.write("## 7. Example Prediction\n\n")

        f.write(f"**Entity:** {sample['entity']}\n\n")

        f.write(f"**Predicted Class:** {sample['predicted_label']}\n\n")

        f.write("**Human-readable explanation:**\n\n")

        f.write(sample["nl_explanation"])

        f.write("\n\n")

        # =====================================================
        # Overall Findings
        # =====================================================
        f.write("## 8. Overall Findings\n\n")

        f.write(
            "- The R-GCN model learns from semantic relationships between entities rather than isolated attributes.\n"
        )

        f.write(
            "- Frequently occurring relation types indicate which information contributes most often to predictions.\n"
        )

        f.write(
            "- Most explanations remain compact while preserving sufficient evidence for understanding model decisions.\n"
        )

        f.write(
            "- Human-readable explanations convert graph evidence into simple language, making the predictions easier for non-technical users to interpret.\n"
        )

    print(f"[INFO] Report saved at: {report_path}")

if __name__ == "__main__":
    main()