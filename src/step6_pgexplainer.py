import torch
import torch.nn.functional as F
import pandas as pd
import pickle

import config
from step3_train_rgcn import FastRGCN
from torch_geometric.utils import k_hop_subgraph


# -----------------------------
# LOAD
# -----------------------------
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
# SUBGRAPH
# -----------------------------
def subgraph(node, data):
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node,
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


# -----------------------------
# EDGE MLP (PGExplainer style)
# -----------------------------
class EdgeMLP(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, h_src, h_dst):
        return torch.sigmoid(self.net(torch.cat([h_src, h_dst], dim=-1))).squeeze()


# -----------------------------
# EXPLAIN ONE NODE
# -----------------------------
def explain(model, data, node, device):
    x, ei, et, idx, subset = subgraph(node, data)

    x, ei, et = x.to(device), ei.to(device), et.to(device)

    src, dst = ei

    with torch.no_grad():
        pred = model(x, ei, et)[idx].argmax().item()

    mlp = EdgeMLP(x.size(1)).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=0.01)

    for _ in range(100):
        opt.zero_grad()

        h = model.conv1(x, ei, et).relu()

        mask = mlp(h[src], h[dst])

        out = model.conv2(h, ei, et)
        logp = F.log_softmax(out, dim=1)

        loss = -logp[idx, pred] + 0.01 * mask.mean()

        loss.backward()
        opt.step()

    return mask.detach().cpu(), pred


# -----------------------------
# MAIN
# -----------------------------
def main():
    model, data, mappings, device = load()

    inv_nodes = {v: k for k, v in mappings["nodes_dict"].items()}
    inv_labels = {v: k for k, v in mappings["labels_dict"].items()}

    nodes = data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()

    results = []

    print("[PG-MLP] Running explanations...")

    for n in nodes:
        mask, pred = explain(model, data, n, device)

        true = inv_labels[int(
            data.test_y[(data.test_idx == n).nonzero()[0, 0]]
        )]

        print(f"{n} | pred={pred} | true={true} | mask_mean={mask.mean().item():.4f}")

        results.append({
            "node": int(n),
            "pred": int(pred),
            "true": true,
            "mask_mean": float(mask.mean())
        })

    pd.DataFrame(results).to_csv(
        f"{config.RESULTS_TABLES_DIR}/pg_explanations.csv",
        index=False
    )

    print("DONE")


if __name__ == "__main__":
    main()