import os
import pandas as pd

import config


# =========================================================
# LOAD STEP 6 OUTPUT
# =========================================================
def load_data():
    path = os.path.join(config.RESULTS_TABLES_DIR, "grad_explanations.csv")
    return pd.read_csv(path)


def parse_edges(edge_str):
    if not isinstance(edge_str, str):
        return []
    return [e for e in edge_str.split(" | ") if e.strip()]


# =========================================================
# MAIN
# =========================================================
def main():

    df = load_data()

    os.makedirs(config.RESULTS_TABLES_DIR, exist_ok=True)

    num_nodes = len(df)

    avg_size = 0
    if "top_edges" in df.columns:
        avg_size = df["top_edges"].apply(lambda x: len(parse_edges(x))).mean()

    report_path = os.path.join(config.RESULTS_TABLES_DIR, "grad_report.md")

    with open(report_path, "w", encoding="utf-8") as f:

        # ========================
        # HEADER
        # ========================
        f.write("# Grad Explanation Report\n\n")

        f.write("## Overview\n")
        f.write(f"- Nodes: {num_nodes}\n")
        f.write(f"- Avg explanation size: {avg_size:.2f}\n\n")

        f.write("---\n\n")

        # ========================
        # FIGURES
        # ========================
        f.write("## Figures\n\n")

        f.write("### Prediction Distribution\n")
        f.write("![](../figures/grad_prediction_distribution.png)\n\n")

        f.write("### Explanation Size\n")
        f.write("![](../figures/grad_explanation_size.png)\n\n")

        f.write("### Relation Importance\n")
        f.write("![](../figures/grad_relation_importance.png)\n\n")

        f.write("### Structural Complexity\n")
        f.write("![](../figures/grad_structural_complexity.png)\n\n")

        f.write("### Class-wise Explanation Strength\n")
        f.write("![](../figures/grad_class_explanation_strength.png)\n\n")

        f.write("### Entity Coverage\n")
        f.write("![](../figures/grad_entity_coverage.png)\n\n")

        f.write("---\n\n")

        # ========================
        # SAMPLE EXPLANATION
        # ========================
        f.write("## Sample Explanation\n\n")

        sample = df.iloc[0]

        entity = sample["entity"]
        label = sample["predicted_label"]

        edges = parse_edges(sample.get("top_edges", ""))

        f.write(f"{entity} is classified as {label}.\n\n")
        f.write("The model uses relationships in the knowledge graph.\n\n")

        f.write("Key evidence:\n")
        for e in edges[:5]:
            f.write(f"- {e}\n")

        f.write("\nThis explanation is purely graph-structural.\n\n")

        f.write("---\n\n")
        f.write("Generated automatically by Step 7 pipeline.\n")

    print("\n========== REPORT GENERATED ==========")
    print(f"[INFO] Saved at: {report_path}")


if __name__ == "__main__":
    main()