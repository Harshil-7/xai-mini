import torch
import pandas as pd
import config
from step3_train_rgcn import FastRGCN
import pickle


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


@torch.no_grad()
def evaluate_one_node(model, data, node, mask_ratio=0.2):
    """
    mask_ratio: fraction of weakest edges to remove (simulate explanation)
    """

    out = model(data.x, data.edge_index, data.edge_type)
    pred_full = out[node].argmax().item()

    # ---- simple "fake explanation" placeholder (replace with your mask) ----
    edge_scores = torch.rand(data.edge_index.size(1), device=data.x.device)

    k = int(mask_ratio * edge_scores.size(0))
    topk_idx = torch.topk(edge_scores, k).indices

    mask = torch.zeros_like(edge_scores)
    mask[topk_idx] = 1.0

    # remove edges NOT in explanation
    keep_edges = mask.bool()

    new_edge_index = data.edge_index[:, keep_edges]
    new_edge_type = data.edge_type[keep_edges]

    out2 = model(data.x, new_edge_index, new_edge_type)
    pred_masked = out2[node].argmax().item()

    # ---- metrics ----
    fidelity = (pred_full == pred_masked)

    sparsity = 1.0 - (keep_edges.sum().item() / edge_scores.size(0))

    return {
        "node": node,
        "pred_full": pred_full,
        "pred_masked": pred_masked,
        "fidelity": float(fidelity),
        "sparsity": float(sparsity),
    }


def main():
    model, data, device, mappings = load_model()

    nodes = data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()

    results = []

    print("[EVAL] Running explanation evaluation...")

    for n in nodes:
        res = evaluate_one_node(model, data, n)
        results.append(res)
        print("done node:", n)

    df = pd.DataFrame(results)

    # ---- summary ----
    print("\n===== FINAL RESULTS =====")
    print("Fidelity (avg):", df["fidelity"].mean())
    print("Sparsity (avg):", df["sparsity"].mean())

    df.to_csv(f"{config.RESULTS_TABLES_DIR}/evaluation.csv", index=False)
    print("Saved to evaluation.csv")


if __name__ == "__main__":
    main()