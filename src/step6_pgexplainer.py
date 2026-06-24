import torch
import torch.nn.functional as F
import pandas as pd
import pickle

import config
from step3_train_rgcn import FastRGCN
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain import Explainer, PGExplainer


# -------------------------
# load model
# -------------------------
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


# -------------------------
# k-hop subgraph
# -------------------------
def subgraph(node, data):
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node,
        num_hops=2,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.x.size(0),
    )

    return (
        data.x[subset],
        edge_index,
        data.edge_type[edge_mask],
        mapping.item(),
        subset,
    )


# -------------------------
# PGExplainer wrapper training
# -------------------------
def train_pg(pg, model, data, nodes, device):

    optimizer = torch.optim.Adam(pg.parameters(), lr=0.003)

    pg.train()

    for epoch in range(10):

        total_loss = 0

        for n in nodes:

            x, ei, et, idx, _ = subgraph(n, data)

            x, ei, et = x.to(device), ei.to(device), et.to(device)

            with torch.no_grad():
                pred = model(x, ei, et)[idx].argmax()

            exp = pg(
                model=model,
                x=x,
                edge_index=ei,
                target=pred,
                edge_type=et,
                index=idx,
            )

            loss = exp.loss

            loss.backward()
            total_loss += float(loss.item())

        optimizer.step()
        optimizer.zero_grad()

        print(f"[PG] epoch {epoch} loss={total_loss:.4f}")


# -------------------------
# explain node
# -------------------------
def explain(pg, model, data, node, device):

    x, ei, et, idx, subset = subgraph(node, data)

    x, ei, et = x.to(device), ei.to(device), et.to(device)

    with torch.no_grad():
        pred = model(x, ei, et)[idx].argmax().item()

    exp = pg(
        model=model,
        x=x,
        edge_index=ei,
        edge_type=et,
        index=idx,
    )

    return exp.edge_mask.detach().cpu(), pred


# -------------------------
# main
# -------------------------
def main():

    model, data, mappings, device = load()

    nodes = data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()

    pg = PGExplainer(epochs=30, lr=0.003).to(device)

    print("[PG] Training PGExplainer...")

    train_pg(pg, model, data, nodes, device)

    print("[PG] Explaining...")

    results = []

    for n in nodes:

        mask, pred = explain(pg, model, data, n, device)

        results.append({
            "node": int(n),
            "mask_mean": float(mask.mean()),
            "pred": int(pred),
        })

        print("done:", n)

    pd.DataFrame(results).to_csv(
        f"{config.RESULTS_TABLES_DIR}/pg_explanations.csv",
        index=False
    )

    print("DONE")


if __name__ == "__main__":
    main()