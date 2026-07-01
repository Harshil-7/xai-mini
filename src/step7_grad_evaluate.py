import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

import config


# =========================================================
# LOAD STEP 6 OUTPUT
# =========================================================
def load_step6():
    path = os.path.join(config.RESULTS_TABLES_DIR, "grad_explanations.csv")
    return pd.read_csv(path)


# =========================================================
# EDGE PARSER
# =========================================================
def parse_edges(edge_str):
    if not isinstance(edge_str, str):
        return []
    return [e.strip() for e in edge_str.split(" | ") if e.strip()]


def extract_relation(edge):
    try:
        return edge.split("[")[1].split("]")[0]
    except:
        return None


# =========================================================
# MAIN
# =========================================================
def main():

    print("\n========== STEP 7 STARTED ==========\n")

    df = load_step6()

    os.makedirs(config.RESULTS_TABLES_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_FIGURES_DIR, exist_ok=True)

    eval_rows = []
    nl_rows = []
    edge_rows = []
    relation_counter = Counter()

    # =====================================================
    # PROCESS
    # =====================================================
    for _, row in df.iterrows():

        entity = row["entity"]
        label = row["predicted_label"]
        edges = parse_edges(row.get("top_edges", ""))

        eval_rows.append({
            "entity": entity,
            "predicted_label": label,
            "num_edges": len(edges)
        })

        nl_rows.append({
            "entity": entity,
            "predicted_label": label,
            "explanation": f"{entity} is classified as {label}."
        })

        for e in edges:
            edge_rows.append({"entity": entity, "edge": e})
            rel = extract_relation(e)
            if rel:
                relation_counter[rel] += 1

    # =====================================================
    # DATAFRAMES
    # =====================================================
    eval_df = pd.DataFrame(eval_rows)
    nl_df = pd.DataFrame(nl_rows)
    edge_df = pd.DataFrame(edge_rows)

    # =====================================================
    # TABLES
    # =====================================================
    eval_df.to_csv(os.path.join(config.RESULTS_TABLES_DIR, "grad_eval_summary.csv"), index=False)
    nl_df.to_csv(os.path.join(config.RESULTS_TABLES_DIR, "grad_nl_explanations.csv"), index=False)
    edge_df.to_csv(os.path.join(config.RESULTS_TABLES_DIR, "grad_edge_explanations.csv"), index=False)

    # =====================================================
    # FIGURE 1: prediction distribution
    # =====================================================
    plt.figure()
    eval_df["predicted_label"].value_counts().plot(kind="bar")
    plt.title("Prediction Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_prediction_distribution.png"))
    plt.close()

    # =====================================================
    # FIGURE 2: explanation size
    # =====================================================
    plt.figure()
    eval_df["num_edges"].hist(bins=20)
    plt.title("Explanation Size Distribution")
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_explanation_size.png"))
    plt.close()

    # =====================================================
    # FIGURE 3: relation importance
    # =====================================================
    plt.figure()
    pd.Series(relation_counter).sort_values(ascending=False).head(10).plot(kind="bar")
    plt.title("Relation Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_relation_importance.png"))
    plt.close()

    # =====================================================
    # FIGURE 4: structural complexity
    # =====================================================
    plt.figure()
    eval_df["num_edges"].plot(kind="hist", bins=20)
    plt.title("Structural Complexity")
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_structural_complexity.png"))
    plt.close()

    # =====================================================
    # FIGURE 5: class-wise strength
    # =====================================================
    plt.figure()
    eval_df.groupby("predicted_label")["num_edges"].mean().plot(kind="bar")
    plt.title("Class-wise Explanation Strength")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_class_explanation_strength.png"))
    plt.close()

    # =====================================================
    # FIGURE 6: entity coverage
    # =====================================================
    plt.figure()
    eval_df["entity"].value_counts().head(15).plot(kind="bar")
    plt.title("Entity Coverage")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_entity_coverage.png"))
    plt.close()

    # =====================================================
    # REPORT.MD GENERATION
    # =====================================================
    report_path = os.path.join(config.RESULTS_TABLES_DIR, "grad_report.md")

    with open(report_path, "w", encoding="utf-8") as f:

        f.write("# Grad Explanation Report\n\n")

        f.write("## Overview\n")
        f.write(f"- Nodes: {len(df)}\n")
        f.write(f"- Avg explanation size: {eval_df['num_edges'].mean():.2f}\n\n")

        f.write("---\n\n")

        f.write("## Figures\n\n")
        f.write("### Prediction Distribution\n![](../figures/grad_prediction_distribution.png)\n\n")
        f.write("### Explanation Size\n![](../figures/grad_explanation_size.png)\n\n")
        f.write("### Relation Importance\n![](../figures/grad_relation_importance.png)\n\n")
        f.write("### Structural Complexity\n![](../figures/grad_structural_complexity.png)\n\n")
        f.write("### Class-wise Explanation Strength\n![](../figures/grad_class_explanation_strength.png)\n\n")
        f.write("### Entity Coverage\n![](../figures/grad_entity_coverage.png)\n\n")

        f.write("---\n\n")

        sample = df.iloc[0]
        f.write("## Sample Explanation\n\n")
        f.write(f"{sample['entity']} is classified as {sample['predicted_label']}.\n\n")
        f.write("Key evidence:\n")

        for e in parse_edges(sample.get("top_edges", ""))[:5]:
            f.write(f"- {e}\n")

        f.write("\nThis explanation is purely graph-structural.\n")

    print("\n========== STEP 7 COMPLETE ==========")
    print("[INFO] Tables + Figures + Report generated")


if __name__ == "__main__":
    main()