import torch
import pandas as pd
import pickle
import os

import config
from step3_train_rgcn import FastRGCN


# ----------------------------
# LOAD MODEL + DATA
# ----------------------------
def load():

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

    return model, data, id_to_entity, id_to_label, id_to_rel, device


# ----------------------------
# CLEAN ENTITY
# ----------------------------
def clean(uri):
    if uri is None:
        return "Unknown"
    return str(uri).split("/")[-1].replace("_", " ")


# ----------------------------
# EXPLANATION CORE (FIXED)
# ----------------------------
def explain_node(model, data, node):

    x = data.x.clone().detach().requires_grad_(True)

    out = model(x, data.edge_index, data.edge_type)
    pred = out[node].argmax().item()

    loss = -out[node, pred]

    model.zero_grad(set_to_none=True)
    loss.backward()

    node_grad = x.grad.abs().sum(dim=1)

    edge_scores = (
        node_grad[data.edge_index[0]] +
        node_grad[data.edge_index[1]]
    )

    return pred, edge_scores.detach()


# ----------------------------
# GET TOP EDGES (IMPORTANT FOR STEP 7)
# ----------------------------
def get_top_edges(data, node, edge_scores, top_k=5):

    src, dst = data.edge_index

    mask = (src == node) | (dst == node)
    idx = mask.nonzero(as_tuple=True)[0]

    if idx.numel() == 0:
        return []

    scores = edge_scores[idx]

    k = min(top_k, len(idx))
    top_idx = torch.topk(scores, k).indices

    edges = []

    for i in top_idx:

        e = idx[i]

        s = int(src[e])
        d = int(dst[e])
        r = int(data.edge_type[e])

        edges.append((s, d, r))

    return edges


# ----------------------------
# MAIN
# ----------------------------
def main():

    print("\n========== STEP 6 STARTED ==========\n")

    model, data, id_to_entity, id_to_label, id_to_rel, device = load()

    if hasattr(data, "test_idx") and data.test_idx is not None:
        nodes = data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()
    else:
        nodes = torch.arange(data.num_nodes)[:config.NUM_NODES_TO_EXPLAIN].tolist()

    results = []

    for i, n in enumerate(nodes):

        pred, edge_scores = explain_node(model, data, n)

        entity = clean(id_to_entity.get(n))
        pred_label = id_to_label.get(pred, str(pred))

        top_edges = get_top_edges(data, n, edge_scores, top_k=5)

        edge_str = []
        for s, d, r in top_edges:
            s_name = clean(id_to_entity.get(s))
            d_name = clean(id_to_entity.get(d))
            r_name = str(id_to_rel.get(r, r))

            edge_str.append(f"{s_name} --[{r_name}]--> {d_name}")

        results.append({
            "entity": entity,
            "predicted_label": pred_label,
            "top_edges": " | ".join(edge_str),
            "node_index": int(n)
        })

        print(f"[{i+1}] {entity} → {pred_label}")

    os.makedirs(config.RESULTS_TABLES_DIR, exist_ok=True)

    out_path = os.path.join(config.RESULTS_TABLES_DIR, "grad_explanations.csv")

    pd.DataFrame(results).to_csv(out_path, index=False)

    print("\n========== DONE ==========")
    print(f"[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()