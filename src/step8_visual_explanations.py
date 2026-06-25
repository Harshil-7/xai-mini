
#--------------------- 1. Class-wise explanation strength---------------------

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/tables/grad_explanations.csv")

class_scores = df.groupby("pred")["score_mean"].mean().sort_values()

plt.figure()
class_scores.plot(kind="bar")
plt.title("Average Explanation Strength per Class (R-GCN + Gradients)")
plt.xlabel("Class")
plt.ylabel("Mean Importance Score")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("results/figures/class_strength.png")
plt.show()


# -----------------------------------2. Top influential nodes ---------------------------------

top_nodes = df.groupby("node")["score_mean"].mean().sort_values(ascending=False).head(10)

plt.figure()
top_nodes.plot(kind="bar")
plt.title("Top 10 Most Influential Nodes in Graph")
plt.xlabel("Node ID")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("results/figures/top_nodes.png")
plt.show()


# -----------------------------------3. Explanation quality: Fidelity vs Sparsity ---------------------------------

eval_df = pd.read_csv("results/tables/explanation_evaluation.csv")

plt.figure()
plt.scatter(eval_df["sparsity"], eval_df["fidelity"])
plt.title("Fidelity vs Sparsity Trade-off")
plt.xlabel("Sparsity (simpler explanation)")
plt.ylabel("Fidelity (faithfulness)")
plt.tight_layout()

plt.savefig("results/figures/fidelity_sparsity.png")
plt.show()


# -----------------------------------4. Distribution of explanation strength ---------------------------------

plt.figure()
plt.hist(df["score_mean"], bins=20)
plt.title("Distribution of Explanation Strength (Gradient Scores)")
plt.xlabel("Importance Score")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig("results/figures/importance_distribution.png")
plt.show()



