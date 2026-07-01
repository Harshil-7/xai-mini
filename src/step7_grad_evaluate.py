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
# EDGE PARSING
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
# NATURAL LANGUAGE (HUMAN FRIENDLY)
# =========================================================
def make_nl(row):

    entity = row["entity"]
    label = row["predicted_label"]
    edges = parse_edges(row.get("top_edges", ""))

    edge_text = "\n".join([f"- {e}" for e in edges]) if edges else "- No strong graph evidence"

    return (
        f"{entity} is classified as {label}.\n\n"
        f"The model uses relationships in the knowledge graph.\n\n"
        f"Key evidence:\n"
        f"{edge_text}\n\n"
        f"This explanation is purely graph-structural."
    )


# =========================================================
# MAIN PIPELINE
# =========================================================
def main():

    print("\n========== STEP 7 STARTED ==========\n")

    os.makedirs(config.RESULTS_TABLES_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_FIGURES_DIR, exist_ok=True)

    df = load_step6()

    eval_rows = []
    nl_rows = []
    edge_rows = []
    relation_counter = Counter()

    # =====================================================
    # PROCESS EACH NODE
    # =====================================================
    for i, row in df.iterrows():

        entity = row["entity"]
        label = row["predicted_label"]
        edges = parse_edges(row.get("top_edges", ""))

        # -------------------------
        # TABLE 1: evaluation
        # -------------------------
        eval_rows.append({
            "entity": entity,
            "predicted_label": label,
            "num_edges": len(edges)
        })

        # -------------------------
        # TABLE 2: NL explanations
        # -------------------------
        nl_rows.append({
            "entity": entity,
            "predicted_label": label,
            "explanation": make_nl(row)
        })

        # -------------------------
        # TABLE 3: edge-level
        # -------------------------
        for e in edges:
            edge_rows.append({
                "entity": entity,
                "edge": e
            })

            rel = extract_relation(e)
            if rel:
                relation_counter[rel] += 1

        print(f"[{i+1}/{len(df)}] {entity}")

    # =====================================================
    # DATAFRAMES
    # =====================================================
    eval_df = pd.DataFrame(eval_rows)
    nl_df = pd.DataFrame(nl_rows)
    edge_df = pd.DataFrame(edge_rows)

    summary_df = pd.DataFrame([{
        "avg_explanation_size": eval_df["num_edges"].mean(),
        "total_entities": len(eval_df)
    }])

    relation_df = pd.DataFrame(
        relation_counter.items(),
        columns=["relation", "count"]
    ).sort_values("count", ascending=False)

    # =====================================================
    # SAVE TABLES (5 FILES)
    # =====================================================
    eval_df.to_csv(os.path.join(config.RESULTS_TABLES_DIR, "grad_eval_summary.csv"), index=False)
    nl_df.to_csv(os.path.join(config.RESULTS_TABLES_DIR, "grad_nl_explanations.csv"), index=False)
    edge_df.to_csv(os.path.join(config.RESULTS_TABLES_DIR, "grad_edge_explanations.csv"), index=False)
    summary_df.to_csv(os.path.join(config.RESULTS_TABLES_DIR, "grad_summary.csv"), index=False)
    relation_df.to_csv(os.path.join(config.RESULTS_TABLES_DIR, "grad_relation_importance.csv"), index=False)

    # =====================================================
    # FIGURE 1: Prediction distribution
    # =====================================================
    plt.figure()
    eval_df["predicted_label"].value_counts().plot(kind="bar")
    plt.title("Prediction Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_prediction_distribution.png"))
    plt.close()

    # =====================================================
    # FIGURE 2: Explanation size
    # =====================================================
    plt.figure()
    eval_df["num_edges"].hist(bins=20)
    plt.title("Explanation Size Distribution")
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_explanation_size.png"))
    plt.close()

    # =====================================================
    # FIGURE 3: Top relations
    # =====================================================
    plt.figure()
    relation_df.head(10).set_index("relation")["count"].plot(kind="bar")
    plt.title("Top Relations in Explanations")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_relation_importance.png"))
    plt.close()

    # =====================================================
    # FIGURE 4: Structural complexity
    # =====================================================
    plt.figure()
    eval_df["num_edges"].plot(kind="hist", bins=20)
    plt.title("Structural Complexity")
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_structural_complexity.png"))
    plt.close()

    # =====================================================
    # FIGURE 5: Class-wise explanation size
    # =====================================================
    plt.figure()
    eval_df.groupby("predicted_label")["num_edges"].mean().plot(kind="bar")
    plt.title("Avg Explanation Size per Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_class_explanation_strength.png"))
    plt.close()

    # =====================================================
    # FIGURE 6: Entity coverage
    # =====================================================
    plt.figure()
    eval_df["entity"].value_counts().head(15).plot(kind="bar")
    plt.title("Most Explained Entities")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_entity_coverage.png"))
    plt.close()

    # =====================================================
    # MARKDOWN REPORT (.md)
    # =====================================================
    md_path = os.path.join(config.RESULTS_TABLES_DIR, "report.md")

    md_content = f"""
# R-GCN Explanation Report

## Overview
- Nodes: {len(eval_df)}
- Avg explanation size: {eval_df['num_edges'].mean():.2f}

---

## Figures

### Prediction Distribution
![](../figures/grad_prediction_distribution.png)

### Explanation Size
![](../figures/grad_explanation_size.png)

### Relation Importance
![](../figures/grad_relation_importance.png)

### Structural Complexity
![](../figures/grad_structural_complexity.png)

### Class-wise Explanation Strength
![](../figures/grad_class_explanation_strength.png)

### Entity Coverage
![](../figures/grad_entity_coverage.png)

---

## Sample Explanation

{nl_df.iloc[0]['explanation'] if len(nl_df) > 0 else 'No explanation available'}

---

Generated automatically by Step 7 pipeline.
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n========== STEP 7 COMPLETE ==========")
    print("Tables: 5 saved")
    print("Figures: 6 saved")
    print("Markdown report generated ✔")


# =========================================================
if __name__ == "__main__":
    main()