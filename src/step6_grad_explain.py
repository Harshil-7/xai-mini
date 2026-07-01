import torch
import torch.nn.functional as F
import pandas as pd
import pickle
import os

import config
from step3_train_rgcn import FastRGCN


# ----------------------------
# Load model + data + mappings
# ----------------------------
def load():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading checkpoint...")
    ckpt = torch.load(config.MODEL_FILE, map_location=device, weights_only=False)

    print("[INFO] Building model...")
    model = FastRGCN(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("[INFO] Loading data...")
    data = ckpt["data"].to(device)

    print("[INFO] Loading mappings...")
    with open(config.MODEL_FILE + ".mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    id_to_entity = {v: k for k, v in mappings["nodes_dict"].items()}
    id_to_label = {v: k for k, v in mappings["labels_dict"].items()}
    id_to_rel = {v: k for k, v in mappings["relations_dict"].items()}

    return model, data, id_to_entity, id_to_label, id_to_rel, device


# ----------------------------
# Explanation function
# ----------------------------
def explain_node(model, data, node, device):

    x = data.x.clone().detach().requires_grad_(True)
    edge_index = data.edge_index
    edge_type = data.edge_type

    out = model(x, edge_index, edge_type)

    pred = out[node].argmax().item()

    loss = -out[node, pred]
    loss.backward()

    src, dst = edge_index
    node_grad = x.grad

    edge_scores = (
        node_grad[src].abs().sum(dim=1) +
        node_grad[dst].abs().sum(dim=1)
    )

    return edge_scores.detach().cpu(), pred


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

    print("\n========== STEP 6 STARTED ==========\n")

    model, data, id_to_entity, id_to_label, id_to_rel, device = load()

    # ----------------------------
    # SAFE NODE SELECTION
    # ----------------------------
    if hasattr(data, "test_idx") and data.test_idx is not None:
        nodes = data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()
        print("[INFO] Using test_idx nodes")
    else:
        nodes = torch.arange(data.num_nodes)[:config.NUM_NODES_TO_EXPLAIN].tolist()
        print("[INFO] Using fallback full node list")

    print(f"[INFO] Explaining {len(nodes)} nodes")

    results = []

    # ----------------------------
    # LOOP
    # ----------------------------
    for i, n in enumerate(nodes):

        scores, pred = explain_node(model, data, n, device)

        raw_entity = id_to_entity.get(n, f"Unknown_{n}")
        entity_name = clean_entity(raw_entity)

        pred_label = id_to_label.get(pred, str(pred))

        explanation = f"{entity_name} is predicted as {pred_label}"

        results.append({
            "entity": entity_name,
            "predicted_label": pred_label,
            "avg_edge_score": float(scores.mean()),
            "explanation": explanation,
            "node_index": int(n)
        })

        print(f"[{i+1}/{len(nodes)}] OK → {entity_name} → {pred_label}")

    # ----------------------------
    # SAVE OUTPUT
    # ----------------------------
    os.makedirs(config.RESULTS_TABLES_DIR, exist_ok=True)

    out_path = f"{config.RESULTS_TABLES_DIR}/grad_explanations.csv"

    pd.DataFrame(results).to_csv(out_path, index=False)

    print("\n========== DONE ==========")
    print(f"[INFO] Saved to: {out_path}")


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    main()