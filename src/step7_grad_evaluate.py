import torch
import pandas as pd
import config
from step3_train_rgcn import FastRGCN
import pickle


# ----------------------------
# Load model + data + mappings
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

    # invert mappings
    id_to_entity = {v: k for k, v in mappings["nodes_dict"].items()}
    id_to_label = {v: k for k, v in mappings["labels_dict"].items()}

    return model, data, device, id_to_entity, id_to_label


# ----------------------------
# Evaluation per node
# ----------------------------
@torch.no_grad()
def evaluate_one_node(model, data, node, mask_ratio=0.2):

    out = model(data.x, data.edge_index, data.edge_type)
    pred_full = out[node].argmax().item()

    # ---- fake explanation scores (your current logic) ----
    edge_scores = torch.rand(data.edge_index.size(1), device=data.x.device)

    k = int(mask_ratio * edge_scores.size(0))
    topk_idx = torch.topk(edge_scores, k).indices

    mask = torch.zeros_like(edge_scores)
    mask[topk_idx] = 1.0

    keep_edges = mask.bool()

    new_edge_index = data.edge_index[:, keep_edges]
    new_edge_type = data.edge_type[keep_edges]

    out2 = model(data.x, new_edge_index, new_edge_type)
    pred_masked = out2[node].argmax().item()

    fidelity = (pred_full == pred_masked)
    sparsity = 1.0 - (keep_edges.sum().item() / edge_scores.size(0))

    return {
        "node": node,
        "pred_full": pred_full,
        "pred_masked": pred_masked,
        "fidelity": float(fidelity),
        "sparsity": float(sparsity),
    }


# ----------------------------
# Helper
# ----------------------------
def clean_entity(uri: str):
    if uri is None:
        return "Unknown"
    return uri.split("/")[-1].replace("_", " ")


# ----------------------------
# MAIN
# ----------------------------
def main():

    model, data, device, id_to_entity, id_to_label = load_model()

    nodes = data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()

    results = []

    print("\n[EVAL] Running explanation evaluation...\n")

    for n in nodes:

        res = evaluate_one_node(model, data, n)

        # ----------------------------
        # Convert node → entity name
        # ----------------------------
        raw_entity = id_to_entity.get(n, f"Unknown_{n}")
        entity_name = clean_entity(raw_entity)

        # ----------------------------
        # Convert predictions → labels
        # ----------------------------
        pred_full_label = id_to_label.get(res["pred_full"], str(res["pred_full"]))
        pred_masked_label = id_to_label.get(res["pred_masked"], str(res["pred_masked"]))

        results.append({
            "entity": entity_name,
            "pred_full": pred_full_label,
            "pred_masked": pred_masked_label,
            "fidelity": res["fidelity"],
            "sparsity": res["sparsity"],
            "node_index": n
        })

        print(f"done node: {entity_name} → {pred_full_label}")

    df = pd.DataFrame(results)

    # ----------------------------
    # Summary
    # ----------------------------
    print("\n===== FINAL RESULTS =====")
    print("Fidelity (avg):", df["fidelity"].mean())
    print("Sparsity (avg):", df["sparsity"].mean())

    out_path = f"{config.RESULTS_TABLES_DIR}/grad_evaluation.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()