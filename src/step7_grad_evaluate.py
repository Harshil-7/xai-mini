import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

import config
from step3_train_rgcn import FastRGCN


# ----------------------------
# LOAD MODEL + DATA + MAPPINGS
# ----------------------------
def load_model():

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

    return model, data, device, id_to_entity, id_to_label


# ----------------------------
# REALISTIC EVALUATION
# ----------------------------
@torch.no_grad()
def evaluate(model, data, node):

    out = model(data.x, data.edge_index, data.edge_type)

    pred_full = out[node].argmax().item()

    # explanation proxy (gradient-based, not random)
    x = data.x.clone().detach().requires_grad_(True)

    out2 = model(x, data.edge_index, data.edge_type)
    loss = -out2[node, pred_full]
    loss.backward()

    node_grad = x.grad.abs().sum(dim=1)

    edge_scores = node_grad[data.edge_index[0]] + node_grad[data.edge_index[1]]

    k = int(0.2 * edge_scores.size(0))

    topk = torch.topk(edge_scores, k).indices

    mask = torch.zeros_like(edge_scores)
    mask[topk] = 1.0

    keep = mask.bool()

    out_masked = model(
        data.x,
        data.edge_index[:, keep],
        data.edge_type[keep]
    )

    pred_masked = out_masked[node].argmax().item()

    fidelity = float(pred_full == pred_masked)
    sparsity = 1.0 - (keep.sum().item() / edge_scores.size(0))

    return pred_full, pred_masked, fidelity, sparsity


# ----------------------------
# CLEAN ENTITY
# ----------------------------
def clean(uri):
    return uri.split("/")[-1].replace("_", " ")


# ----------------------------
# NATURAL LANGUAGE EXPLANATION
# ----------------------------
def explain_text(entity, label, fidelity, sparsity):

    strength = (
        "strong structural evidence"
        if sparsity > 0.7 else
        "moderate structural evidence"
        if sparsity > 0.4 else
        "weak structural evidence"
    )

    stability = (
        "stable prediction under explanation masking"
        if fidelity == 1.0 else
        "sensitive prediction under masking"
    )

    return f"{entity} is classified as {label}. It shows {strength} and {stability}."


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():

    print("\n========== STEP 7 FULL PIPELINE STARTED ==========\n")

    model, data, device, id_to_entity, id_to_label = load_model()

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    nodes = data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()

    results = []
    explanations = []

    for n in nodes:

        pred_full, pred_masked, fidelity, sparsity = evaluate(model, data, n)

        entity = clean(id_to_entity.get(n, f"Unknown_{n}"))

        label = id_to_label.get(pred_full, str(pred_full))

        nl = explain_text(entity, label, fidelity, sparsity)

        results.append({
            "entity": entity,
            "pred_full": label,
            "pred_masked": id_to_label.get(pred_masked, str(pred_masked)),
            "fidelity": fidelity,
            "sparsity": sparsity,
            "node_index": n
        })

        explanations.append(nl)

        print("[OK]", nl)

    df = pd.DataFrame(results)

    # ----------------------------
    # SAVE CSV
    # ----------------------------
    csv_path = "results/tables/grad_evaluation.csv"
    df.to_csv(csv_path, index=False)

    print("\nSaved CSV:", csv_path)

    # ----------------------------
    # FIGURE 1: Fidelity vs Sparsity
    # ----------------------------
    plt.figure()
    plt.scatter(df["sparsity"], df["fidelity"])
    plt.title("Fidelity vs Sparsity")
    plt.xlabel("Sparsity")
    plt.ylabel("Fidelity")
    plt.savefig("results/figures/grad_fidelity_vs_sparsity.png")
    plt.close()

    # ----------------------------
    # FIGURE 2: Class distribution
    # ----------------------------
    plt.figure()
    df["pred_full"].value_counts().plot(kind="bar")
    plt.title("Prediction Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("results/figures/grad_class_distribution.png")
    plt.close()

    # ----------------------------
    # SAVE NL EXPLANATIONS
    # ----------------------------
    nl_df = df.copy()
    nl_df["explanation"] = explanations

    nl_df.to_csv("results/tables/grad_natural_language_explanations.csv", index=False)

    print("\n========== DONE ==========")
    print("Figures saved in results/figures/")
    print("NL explanations saved in results/tables/")


if __name__ == "__main__":
    main()