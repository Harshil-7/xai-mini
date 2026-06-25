import pandas as pd
import matplotlib.pyplot as plt

# --------------------- LOAD DATA ---------------------

df = pd.read_csv("results/tables/grad_explanations.csv")
eval_df = pd.read_csv("results/tables/explanation_evaluation.csv")


# --------------------- 1. Class-wise explanation strength ---------------------

class_scores = df.groupby("pred")["avg_edge_score"].mean().sort_values()

plt.figure()
class_scores.plot(kind="bar")
plt.title("Average Explanation Strength per Class (R-GCN + Gradients)")
plt.xlabel("Class")
plt.ylabel("Avg Edge Score")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("results/figures/class_strength.png")
plt.show()


# --------------------- 2. Top influential nodes ---------------------

top_nodes = df.groupby("node")["avg_edge_score"].mean().sort_values(ascending=False).head(10)

plt.figure()
top_nodes.plot(kind="bar")
plt.title("Top 10 Most Influential Nodes")
plt.xlabel("Node ID")
plt.ylabel("Avg Edge Score")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("results/figures/top_nodes.png")
plt.show()


# --------------------- 3. Fidelity vs Sparsity ---------------------

# FIXED: compute fidelity once only
eval_df["fidelity"] = (
    eval_df["fidelity_plus"] - eval_df["fidelity_minus"]
).abs()

plt.figure()
plt.scatter(eval_df["sparsity"], eval_df["fidelity"])
plt.title("Fidelity vs Sparsity Trade-off")
plt.xlabel("Sparsity (higher = simpler explanation)")
plt.ylabel("Fidelity (difference-based score)")
plt.tight_layout()

plt.savefig("results/figures/fidelity_sparsity.png")
plt.show()


# --------------------- 4. Distribution of explanation strength ---------------------

plt.figure()
plt.hist(df["avg_edge_score"], bins=20)
plt.title("Distribution of Explanation Strength")
plt.xlabel("Avg Edge Score")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig("results/figures/importance_distribution.png")
plt.show()