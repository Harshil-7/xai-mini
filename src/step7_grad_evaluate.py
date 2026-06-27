import torch
import pandas as pd
import matplotlib.pyplot as plt
import config
from step3_train_rgcn import FastRGCN
import pickle

# =========================================================
# 1. LOAD MODEL (for safety / consistency - optional use)
# =========================================================
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(config.MODEL_FILE, map_location=device, weights_only=False)

    model = FastRGCN(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data = ckpt["data"].to(device)

    with open(config.MODEL_FILE + ".mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    return model, data, device, mappings


# =========================================================
# 2. LOAD DATA
# =========================================================
df = pd.read_csv("results/tables/grad_explanations.csv")
eval_df = pd.read_csv("results/tables/grad_evaluation.csv")

print("[INFO] Data loaded:")
print(df.head())


# =========================================================
# 3. VISUALIZATION 1 - Class-wise explanation strength
# =========================================================
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
# 4. VISUALIZATION 2 - Top influential nodes
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
# 5. VISUALIZATION 3 - Fidelity vs Sparsity
# =========================================================
plt.figure()
plt.scatter(eval_df["sparsity"], eval_df["fidelity"])
plt.title("Fidelity vs Sparsity Trade-off")
plt.xlabel("Sparsity (higher = simpler explanation)")
plt.ylabel("Fidelity")
plt.tight_layout()

plt.savefig("results/figures/grad_fidelity_sparsity.png")
plt.show()


# =========================================================
# 6. VISUALIZATION 4 - Distribution of explanation scores
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
# 7. NATURAL LANGUAGE EXPLANATIONS
# =========================================================
def explain(row):
    node = row["node"]
    pred = str(row["pred"])
    score = row["avg_edge_score"]

    pred_lower = pred.lower()

    if "athlete" in pred_lower or pred == "3":
        if score > 0.7:
            return f"Node {node} is classified as ATHLETE because it has strong connections to sports-related entities and communities."
        else:
            return f"Node {node} is classified as ATHLETE due to moderate sports-related connections."

    elif "scientist" in pred_lower or pred == "1":
        if score > 0.7:
            return f"Node {node} is classified as SCIENTIST due to strong academic and research connections."
        else:
            return f"Node {node} is classified as SCIENTIST based on partial academic associations."

    else:
        return f"Node {node} is classified as class {pred} with importance score {score:.4f}."


df["explanation"] = df.apply(explain, axis=1)

print("\nSample Explanations:")
print(df["explanation"].head(10))

df.to_csv("results/tables/grad_natural_language_explanations.csv", index=False)
print("\nSaved: grad_natural_language_explanations.csv")


# =========================================================
# 8. SUMMARY PRINT
# =========================================================
print("\n===== FINAL RESULTS =====")
print("Avg Fidelity:", eval_df["fidelity"].mean())
print("Avg Sparsity:", eval_df["sparsity"].mean())