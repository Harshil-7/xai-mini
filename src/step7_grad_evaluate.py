import torch
import pandas as pd
import config
from step3_train_rgcn import FastRGCN
import pickle

# =========================================================
# 1. LOAD MODEL
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
# 2. FIXED GRADIENT-BASED EXPLANATION
# =========================================================
def evaluate_one_node(model, data, node, mask_ratio=0.4):
    """
    Improved gradient-based explanation (FIXED VERSION)
    """

    # IMPORTANT: enable gradients
    x = data.x.clone().detach().requires_grad_(True)

    out = model(x, data.edge_index, data.edge_type)
    pred_full = out[node].argmax().item()

    # =====================================================
    # 1. Compute gradient for predicted class
    # =====================================================
    loss = -out[node, pred_full]
    loss.backward()

    node_grad = x.grad.abs()

    # =====================================================
    # 2. Convert node gradients → edge importance
    # =====================================================
    src, dst = data.edge_index

    edge_scores = node_grad[src].sum(dim=1) + node_grad[dst].sum(dim=1)

    # =====================================================
    # 3. NORMALIZE edge scores (VERY IMPORTANT FIX)
    # =====================================================
    edge_scores = (edge_scores - edge_scores.min()) / (
        edge_scores.max() - edge_scores.min() + 1e-8
    )

    # =====================================================
    # 4. MORE BALANCED MASK (FIXED SPARSITY ISSUE)
    # =====================================================
    threshold = torch.quantile(edge_scores, 1.0 - mask_ratio)
    mask = (edge_scores >= threshold).float()

    keep_edges = mask.bool()

    # =====================================================
    # 5. Create masked graph
    # =====================================================
    new_edge_index = data.edge_index[:, keep_edges]
    new_edge_type = data.edge_type[keep_edges]

    # =====================================================
    # 6. Re-evaluate
    # =====================================================
    out2 = model(x, new_edge_index, new_edge_type)
    pred_masked = out2[node].argmax().item()

    # =====================================================
    # 7. METRICS
    # =====================================================
    fidelity = float(pred_full == pred_masked)
    sparsity = 1.0 - (keep_edges.sum().item() / edge_scores.size(0))

    return {
        "node": node,
        "pred_full": pred_full,
        "pred_masked": pred_masked,
        "fidelity": fidelity,
        "sparsity": sparsity,
    }


# =========================================================
# 3. MAIN
# =========================================================
def main():
    model, data, device, mappings = load_model()

    nodes = data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()

    results = []

    print("[EVAL] Running FIXED gradient explanation evaluation...")

    for n in nodes:
        res = evaluate_one_node(model, data, n)
        results.append(res)
        print(f"done node: {n}")

    df = pd.DataFrame(results)

    # =====================================================
    # SUMMARY
    # =====================================================
    print("\n===== FINAL RESULTS =====")
    print("Avg Fidelity:", df["fidelity"].mean())
    print("Avg Sparsity:", df["sparsity"].mean())

    # =====================================================
    # SAVE
    # =====================================================
    out_path = "results/tables/grad_explanation_evaluation.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()