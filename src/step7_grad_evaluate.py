import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 0. SAFE FOLDER CREATION
# =========================================================
os.makedirs("results/figures", exist_ok=True)

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv("results/tables/grad_explanations.csv")
eval_df = pd.read_csv("results/tables/grad_evaluation.csv")

print("[INFO] Data loaded")
print(df.head())

# =========================================================
# 2. CLASS-WISE EXPLANATION STRENGTH
# =========================================================
if "pred" in df.columns and "avg_edge_score" in df.columns:
    class_scores = df.groupby("pred")["avg_edge_score"].mean().sort_values()

    plt.figure()
    class_scores.plot(kind="bar")
    plt.title("Average Explanation Strength per Class (Grad-RGCN)")
    plt.xlabel("Class")
    plt.ylabel("Avg Edge Score")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig("results/figures/grad_class_strength.png")
    plt.show()

# =========================================================
# 3. TOP INFLUENTIAL NODES
# =========================================================
top_nodes = df.groupby("node")["avg_edge_score"].mean().sort_values(ascending=False).head(10)

plt.figure()
top_nodes.plot(kind="bar")
plt.title("Top 10 Most Influential Nodes")
plt.xlabel("Node ID")
plt.ylabel("Avg Edge Score")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("results/figures/grad_top_nodes.png")
plt.show()

# =========================================================
# 4. FIDELITY vs SPARSITY
# =========================================================
if "fidelity" in eval_df.columns and "sparsity" in eval_df.columns:
    plt.figure()
    plt.scatter(eval_df["sparsity"], eval_df["fidelity"])
    plt.title("Fidelity vs Sparsity Trade-off")
    plt.xlabel("Sparsity (higher = simpler explanation)")
    plt.ylabel("Fidelity")
    plt.tight_layout()

    plt.savefig("results/figures/grad_fidelity_sparsity.png")
    plt.show()

# =========================================================
# 5. DISTRIBUTION
# =========================================================
plt.figure()
plt.hist(df["avg_edge_score"], bins=20)
plt.title("Distribution of Explanation Strength")
plt.xlabel("Avg Edge Score")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig("results/figures/grad_importance_distribution.png")
plt.show()

# =========================================================
# 6. NATURAL LANGUAGE EXPLANATIONS
# =========================================================
def explain(row):
    node = row["node"]
    pred = str(row["pred"])
    score = row["avg_edge_score"]

    if pred == "3":
        if score > 0.7:
            return f"Node {node} is strongly classified as ATHLETE due to strong sports-related connections."
        else:
            return f"Node {node} is weakly classified as ATHLETE."

    elif pred == "1":
        if score > 0.7:
            return f"Node {node} is strongly classified as SCIENTIST due to academic links."
        else:
            return f"Node {node} is weakly classified as SCIENTIST."

    else:
        return f"Node {node} belongs to class {pred} with importance {score:.4f}."

df["explanation"] = df.apply(explain, axis=1)

df.to_csv("results/tables/grad_natural_language_explanations.csv", index=False)

print(df["explanation"].head(10))

# =========================================================
# 7. FINAL SUMMARY
# =========================================================
print("\n===== FINAL RESULTS =====")
print("Avg Fidelity:", eval_df["fidelity"].mean())
print("Avg Sparsity:", eval_df["sparsity"].mean())