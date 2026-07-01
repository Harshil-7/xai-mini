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
# CLEAN EDGE PARSING
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
# NATURAL LANGUAGE EXPLANATION (HUMAN + SHAP STYLE)
# =========================================================
def make_nl(row):

    entity = row["entity"]
    label = row["predicted_label"]
    edges = parse_edges(row.get("top_edges", ""))

    edge_text = "\n".join([f"- {e}" for e in edges]) if edges else "- No strong graph evidence found"

    return (
        f"{entity} is classified as {label}.\n\n"
        f"The model predicts this based on graph structure in DBpedia.\n\n"
        f"Key evidence:\n"
        f"{edge_text}\n\n"
        f"This explanation reflects relational structure, not textual features."
    )


# =========================================================
# MAIN
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
        # TABLE 1: evaluation summary
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
        # TABLE 3: edge-level table
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

    # =====================================================
    # TABLE 4: SUMMARY STATS
    # =====================================================
    summary_df = pd.DataFrame([{
        "avg_explanation_size": eval_df["num_edges"].mean(),
        "total_entities": len(eval_df),
        "avg_edges": eval_df["num_edges"].mean()
    }])

    # =====================================================
    # TABLE 5: RELATION IMPORTANCE
    # =====================================================
    relation_df = pd.DataFrame(
        relation_counter.items(),
        columns=["relation", "count"]
    ).sort_values("count", ascending=False)

    # =====================================================
    # SAVE TABLES
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
    # FIGURE 2: Explanation size distribution
    # =====================================================
    plt.figure()
    eval_df["num_edges"].hist(bins=20)
    plt.title("Explanation Size Distribution")
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_explanation_size.png"))
    plt.close()

    # =====================================================
    # FIGURE 3: Relation importance
    # =====================================================
    plt.figure()
    relation_df.head(10).set_index("relation")["count"].plot(kind="bar")
    plt.title("Top Relations in Explanations")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_relation_importance.png"))
    plt.close()

    # =====================================================
    # FIGURE 4: Fidelity proxy (structure-based)
    # =====================================================
    plt.figure()
    eval_df["num_edges"].plot(kind="hist", bins=20)
    plt.title("Structural Explanation Complexity")
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_structural_complexity.png"))
    plt.close()

    # =====================================================
    # FIGURE 5: Class-wise explanation strength
    # =====================================================
    plt.figure()
    eval_df.groupby("predicted_label")["num_edges"].mean().plot(kind="bar")
    plt.title("Average Explanation Size per Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_class_explanation_strength.png"))
    plt.close()

    # =====================================================
    # FIGURE 6: Entity coverage
    # =====================================================
    plt.figure()
    eval_df["entity"].value_counts().head(15).plot(kind="bar")
    plt.title("Most Frequently Explained Entities")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_FIGURES_DIR, "grad_entity_coverage.png"))
    plt.close()

    # =====================================================
    # DONE
    # =====================================================
    print("\n========== STEP 7 COMPLETE ==========")
    print("Tables saved: 5")
    print("Figures saved: 6")


if __name__ == "__main__":
    main()