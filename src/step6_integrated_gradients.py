import torch
import torch.nn.functional as F
import pandas as pd
import pickle

import config
from step3_train_rgcn import FastRGCN


def load():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(config.MODEL_FILE, map_location=device, weights_only=False)

    model = FastRGCN(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data = ckpt["data"].to(device)

    with open(config.MODEL_FILE + ".mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    return model, data, mappings, device


# -----------------------------
# Integrated Gradients
# -----------------------------
def integrated_gradients(model, x, edge_index, edge_type, node_idx, target_class, steps=20):
    x = x.clone().detach().requires_grad_(True)

    baseline = torch.zeros_like(x)

    total_grad = torch.zeros_like(x)

    for alpha in torch.linspace(0, 1, steps):

        x_interp = baseline + alpha * (x - baseline)
        x_interp.requires_grad_(True)

        out = model(x_interp, edge_index, edge_type)
        score = out[node_idx, target_class]

        score.backward()

        total_grad += x_interp.grad.detach()

    avg_grad = total_grad / steps
    return (x - baseline) * avg_grad


def explain_node(model, data, node_idx, device):
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_type)
        pred = out[node_idx].argmax().item()

    ig = integrated_gradients(
        model,
        data.x,
        data.edge_index,
        data.edge_type,
        node_idx,
        pred,
    )

    node_score = ig.abs().mean(dim=1)

    return node_score, pred


def main():
    model, data, mappings, device = load()

    inv_labels = {v: k for k, v in mappings["labels_dict"].items()}

    nodes = data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()

    results = []

    print("[IG] Running Integrated Gradients...")

    for n in nodes:
        scores, pred = explain_node(model, data, n, device)

        true = inv_labels[int(
            data.test_y[(data.test_idx == n).nonzero()[0, 0]]
        )]

        print(f"Node {n} | pred={pred} | true={true}")

        results.append({
            "node": int(n),
            "pred": pred,
            "true": true,
            "importance_mean": float(scores.mean().item())
        })

    pd.DataFrame(results).to_csv(
        f"{config.RESULTS_TABLES_DIR}/ig_explanations.csv",
        index=False
    )

    print("DONE")


if __name__ == "__main__":
    main()