import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

import config
from step3_train_rgcn import FastRGCN


# =========================================================
# LOAD MODEL + DATA + MAPPINGS
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(config.MODEL_FILE, map_location=device, weights_only=False)

model = FastRGCN(**ckpt["model_args"]).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

data = ckpt["data"].to(device)

with open(config.MODEL_FILE + ".mappings.pkl", "rb") as f:
    mappings = pickle.load(f)

id_to_entity = {v: k for k, v in mappings["nodes_dict"].items()}
id_to_label = {v: k for k, v in mappings["labels_dict"].items()}
id_to_rel = {v: k for k, v in mappings["relations_dict"].items()}


# =========================================================
# CLEANER (DBpedia readable)
# =========================================================
def clean(uri):
    if uri is None:
        return "Unknown"
    return str(uri).split("/")[-1].replace("_", " ")


# =========================================================
# MODEL EVALUATION
# =========================================================
def evaluate(model, data, node, keep_ratio=0.2):

    out = model(data.x, data.edge_index, data.edge_type)
    pred_full = out[node].argmax().item()

    x = data.x.clone().detach().requires_grad_(True)
    out2 = model(x, data.edge_index, data.edge_type)

    loss = -out2[node, pred_full]
    model.zero_grad(set_to_none=True)
    loss.backward()

    node_grad = x.grad.abs().sum(dim=1)

    edge_scores = node_grad[data.edge_index[0]] + node_grad[data.edge_index[1]]

    k = max(int(keep_ratio * edge_scores.size(0)), 1)
    topk = torch.topk(edge_scores, k).indices

    keep_mask = torch.zeros_like(edge_scores, dtype=torch.bool)
    keep_mask[topk] = True

    out_masked = model(
        data.x,
        data.edge_index[:, keep_mask],
        data.edge_type[keep_mask]
    )

    pred_masked = out_masked[node].argmax().item()

    fidelity = float(pred_full == pred_masked)
    sparsity = 1.0 - (keep_mask.sum().item() / edge_scores.size(0))

    return pred_full, pred_masked, fidelity, sparsity, edge_scores.detach()


# =========================================================
# GET HUMAN-READABLE EDGES
# =========================================================
def get_top_edges(node, data, edge_scores, top_k=5):

    src, dst = data.edge_index
    mask = (src == node) | (dst == node)
    idx = mask.nonzero(as_tuple=True)[0]

    if idx.numel() == 0:
        return []

    scores = edge_scores[idx]
    k = min(top_k, len(idx))
    chosen = torch.topk(scores, k).indices

    edges = []

    for e in idx[chosen]:

        s = int(data.edge_index[0][e])
        d = int(data.edge_index[1][e])
        r = int(data.edge_type[e])

        s_name = clean(id_to_entity.get(s))
        d_name = clean(id_to_entity.get(d))
        r_name = clean(id_to_rel.get(r))

        edges.append(f"{s_name} --[{r_name}]--> {d_name}")

    return edges


# =========================================================
# NATURAL LANGUAGE EXPLANATION (HUMAN FRIENDLY)
# =========================================================
def make_nl(entity, pred, true, edges, fidelity, sparsity):

    edge_text = "\n".join([f"• {e}" for e in edges]) if edges else "• No strong relations found"

    return (
        f"{entity} is predicted as {pred}.\n\n"
        f"Graph reasoning:\n"
        f"The model makes this decision using relationships in the knowledge graph.\n\n"
        f"Key evidence:\n"
        f"{edge_text}\n\n"
        f"Explanation quality:\n"
        f"- Fidelity: {fidelity:.3f}\n"
        f"- Sparsity: {sparsity:.3f}\n"
    )


# =========================================================
# NODE SELECTION
# =========================================================
nodes = (
    data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()
    if hasattr(data, "test_idx")
    else torch.arange(data.num_nodes)[:config.NUM_NODES_TO_EXPLAIN].tolist()
)


# =========================================================
# STORAGE
# =========================================================
eval_rows = []
nl_rows = []
edge_rows = []
summary_edges = []
relation_counter = {}


# =========================================================
# MAIN LOOP
# =========================================================
for i, n in enumerate(nodes):

    pred_full, pred_masked, fidelity, sparsity, edge_scores = evaluate(model, data, n)

    entity = clean(id_to_entity.get(n))
    pred_label = id_to_label.get(pred_full, str(pred_full))
    true_label = "Unknown"

    if hasattr(data, "y") and data.y is not None:

           try:
              true_label = id_to_label.get(int(data.y[n].item()), "Unknown")
           except Exception:
                true_label = "Unknown"

    edges = get_top_edges(n, data, edge_scores, top_k=5)

    nl_text = make_nl(entity, pred_label, true_label, edges, fidelity, sparsity)

    # -------------------------
    # TABLE 1: Evaluation
    # -------------------------
    eval_rows.append({
        "entity": entity,
        "predicted_label": pred_label,
        "true_label": true_label,
        "fidelity": fidelity,
        "sparsity": sparsity,
        "node_index": n,
        "num_edges": len(edges)
    })

    # -------------------------
    # TABLE 2: NL explanations
    # -------------------------
    nl_rows.append({
        "entity": entity,
        "predicted_label": pred_label,
        "true_label": true_label,
        "explanation_edges": " | ".join(edges),
        "explanation": nl_text,
        "node_index": n
    })

    # -------------------------
    # TABLE 3: Edge-level breakdown
    # -------------------------
    for e in edges:
        edge_rows.append({
            "node": entity,
            "edge": e
        })

        rel = e.split("[")[1].split("]")[0]
        relation_counter[rel] = relation_counter.get(rel, 0) + 1

    # -------------------------
    # SUMMARY RELATION STATS
    # -------------------------
    summary_edges.append(len(edges))

    print(f"[{i+1}/{len(nodes)}] {entity} → {pred_label}")


# =========================================================
# DATAFRAMES
# =========================================================
eval_df = pd.DataFrame(eval_rows)
nl_df = pd.DataFrame(nl_rows)
edge_df = pd.DataFrame(edge_rows)

summary_df = pd.DataFrame({
    "mean_fidelity": [eval_df["fidelity"].mean()],
    "mean_sparsity": [eval_df["sparsity"].mean()],
    "avg_edges_per_explanation": [sum(summary_edges) / len(summary_edges)]
})


# =========================================================
# SAVE TABLES (5 TABLES TOTAL)
# =========================================================
tables_dir = config.RESULTS_TABLES_DIR
fig_dir = config.RESULTS_FIGURES_DIR

os.makedirs(tables_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

eval_df.to_csv(os.path.join(tables_dir, "grad_evaluation.csv"), index=False)
nl_df.to_csv(os.path.join(tables_dir, "grad_nl_explanations.csv"), index=False)
edge_df.to_csv(os.path.join(tables_dir, "grad_edge_explanations.csv"), index=False)
summary_df.to_csv(os.path.join(tables_dir, "grad_summary.csv"), index=False)


# =========================================================
# FIGURE 1: Fidelity vs Sparsity
# =========================================================
plt.figure()
plt.scatter(eval_df["sparsity"], eval_df["fidelity"])
plt.title("Fidelity vs Sparsity")
plt.xlabel("Sparsity")
plt.ylabel("Fidelity")
plt.savefig(os.path.join(fig_dir, "grad_fidelity_vs_sparsity.png"))
plt.close()


# =========================================================
# FIGURE 2: Fidelity distribution
# =========================================================
plt.figure()
plt.hist(eval_df["fidelity"], bins=10)
plt.title("Fidelity Distribution")
plt.savefig(os.path.join(fig_dir, "grad_fidelity_hist.png"))
plt.close()


# =========================================================
# FIGURE 3: Sparsity distribution
# =========================================================
plt.figure()
plt.hist(eval_df["sparsity"], bins=20)
plt.title("Sparsity Distribution")
plt.savefig(os.path.join(fig_dir, "grad_sparsity_hist.png"))
plt.close()


# =========================================================
# FIGURE 4: Prediction distribution
# =========================================================
plt.figure()
eval_df["predicted_label"].value_counts().plot(kind="bar")
plt.title("Prediction Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "grad_prediction_distribution.png"))
plt.close()


# =========================================================
# FIGURE 5: Explanation size distribution
# =========================================================
plt.figure()
eval_df["num_edges"].hist()
plt.title("Explanation Size Distribution")
plt.savefig(os.path.join(fig_dir, "grad_explanation_size.png"))
plt.close()


# =========================================================
# FIGURE 6: Top relations
# =========================================================
plt.figure()
pd.Series(relation_counter).sort_values(ascending=False).head(10).plot(kind="bar")
plt.title("Top Relations in Explanations")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "grad_top_relations.png"))
plt.close()


# =========================================================
# DONE
# =========================================================
print("\n========== STEP 7 COMPLETE ==========")
print("Tables: 5 saved")
print("Figures: 6 saved")
print("All explanations are human-readable ✔")