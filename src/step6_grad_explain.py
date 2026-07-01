import torch
import torch.nn.functional as F
import pandas as pd
import pickle

import config
from step3_train_rgcn import FastRGCN


# ----------------------------
# Load trained model + data + mappings
# ----------------------------
def load():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(config.MODEL_FILE, map_location=device, weights_only=False)

    model = FastRGCN(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data = ckpt["data"].to(device)

    # load mappings
    with open(config.MODEL_FILE + ".mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    # invert mappings
    id_to_entity = {v: k for k, v in mappings["nodes_dict"].items()}
    id_to_label = {v: k for k, v in mappings["labels_dict"].items()}
    id_to_rel = {v: k for k, v in mappings["relations_dict"].items()}

    return model, data, mappings, id_to_entity, id_to_label, id_to_rel, device


# ----------------------------
# Edge attribution via gradients
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
# Helper: convert node → readable entity
# ----------------------------
def clean_entity(uri):
    return uri.split("/")[-1].replace("_", " ")


# ----------------------------
# Main
# ----------------------------
def main():

    model, data, mappings, id_to_entity, id_to_label, id_to_rel, device = load()

    nodes = data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()

    results = []

    for n in nodes:

        scores, pred = explain_node(model, data, n, device)

        # ----------------------------
        # Convert node → entity name
        # ----------------------------
        raw_entity = id_to_entity.get(n, f"Unknown_{n}")
        entity_name = clean_entity(raw_entity)

        # predicted label
        pred_label = id_to_label.get(pred, str(pred))

        # explanation (simple readable version)
        explanation = f"{entity_name} is predicted as {pred_label}"

        results.append({
            "entity": entity_name,
            "predicted_label": pred_label,
            "avg_edge_score": float(scores.mean()),
            "explanation": explanation,
            "node_index": int(n)
        })

        print(f"[OK] {entity_name} → {pred_label}")

    # save CSV
    pd.DataFrame(results).to_csv(
        f"{config.RESULTS_TABLES_DIR}/grad_explanations.csv",
        index=False
    )

    print("DONE")