import torch
import pandas as pd
import pickle
import os
import config


# ----------------------------
# LOAD MODEL
# ----------------------------
from step3_train_rgcn import FastRGCN
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
    id_to_rel = {v: k for k, v in mappings["relations_dict"].items()}

    return model, data, device, id_to_entity, id_to_label, id_to_rel


# ----------------------------
# CLEAN RELATIONS
# ----------------------------
BAD_RELATIONS = {
    "alias", "weight", "candidate", "thesis",
    "wikiPageWikiLink", "rdf", "unknown",
    "wickets", "preceded"
}

def is_valid_relation(r):
    r = str(r).lower()
    return not any(b in r for b in BAD_RELATIONS)

def clean(x):
    if x is None:
        return "Unknown"
    return str(x).split("/")[-1].replace("_", " ")


# ----------------------------
# EXPLAIN
# ----------------------------
def explain(model, data, node):
    x = data.x.clone().detach().requires_grad_(True)

    out = model(x, data.edge_index, data.edge_type)
    pred = out[node].argmax().item()

    loss = -out[node, pred]

    model.zero_grad(set_to_none=True)
    loss.backward()

    grad = x.grad.abs().sum(dim=1)
    edge_scores = grad[data.edge_index[0]] + grad[data.edge_index[1]]

    return pred, edge_scores.detach()


# ----------------------------
# TOP EDGES
# ----------------------------
def top_edges(data, node, scores, k, id_to_entity, id_to_rel):

    src, dst = data.edge_index
    mask = (src == node) | (dst == node)
    idx = mask.nonzero(as_tuple=True)[0]

    edges = []

    if len(idx) == 0:
        return edges

    local_scores = scores[idx]
    k = min(k, len(idx))
    topk = torch.topk(local_scores, k).indices

    for i in topk:
        e = idx[i].item()

        r = clean(id_to_rel.get(int(data.edge_type[e])))
        if not is_valid_relation(r):
            continue

        s = clean(id_to_entity.get(int(src[e])))
        d = clean(id_to_entity.get(int(dst[e])))

        edges.append(f"{s} --[{r}]--> {d}")

    return edges


# ----------------------------
# RUN STEP 6
# ----------------------------
def run_step6():

    model, data, device, id_to_entity, id_to_label, id_to_rel = load_model()

    if hasattr(data, "test_idx"):
        nodes = data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()
    else:
        nodes = list(range(min(config.NUM_NODES_TO_EXPLAIN, data.num_nodes)))

    rows = []

    for n in nodes:

        pred, scores = explain(model, data, n)

        entity = clean(id_to_entity.get(n))
        pred_label = id_to_label.get(pred, str(pred))

        edges = top_edges(data, n, scores, 5, id_to_entity, id_to_rel)

        rows.append({
            "entity": entity,
            "predicted_label": pred_label,
            "explanation_edges": " | ".join(edges),
            "node_index": n
        })

    os.makedirs(config.RESULTS_TABLES_DIR, exist_ok=True)

    pd.DataFrame(rows).to_csv(
        os.path.join(config.RESULTS_TABLES_DIR, "grad_explanations.csv"),
        index=False
    )

    print("Step 6 done: clean explanations generated")


# RUN
run_step6()