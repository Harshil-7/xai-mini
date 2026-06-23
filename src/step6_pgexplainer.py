import torch
import torch.nn.functional as F
import pandas as pd
import pickle

import config
from step3_train_rgcn import FastRGCN
from torch_geometric.utils import k_hop_subgraph


# ---------------------------
# load model
# ---------------------------
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


# ---------------------------
# k-hop subgraph
# ---------------------------
def subgraph(node, data):
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node,
        2,
        data.edge_index,
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


# ---------------------------
# REAL PG-style explainer
# ---------------------------
class EdgeMask(torch.nn.Module):
    def __init__(self, num_edges):
        super().__init__()
        self.mask = torch.nn.Parameter(torch.randn(num_edges))

    def forward(self):
        return torch.sigmoid(self.mask)


def explain(model, data, node, device):

    x, ei, et, idx, subset = subgraph(node, data)

    x, ei, et = x.to(device), ei.to(device), et.to(device)

    with torch.no_grad():
        pred = model(x, ei, et)[idx].argmax().item()

    num_edges = ei.size(1)
    edge_mask_model = EdgeMask(num_edges).to(device)
    opt = torch.optim.Adam(edge_mask_model.parameters(), lr=0.05)

    src, dst = ei

    for _ in range(200):

        opt.zero_grad()

        mask = edge_mask_model()

        # apply mask (simple gating)
        x1 = model.conv1(x, ei, et)
        x1 = x1.relu()

        x1 = x1 * mask[src].unsqueeze(-1)

        out = model.conv2(x1, ei, et)

        logp = F.log_softmax(out, dim=1)

        fidelity_loss = -logp[idx, pred]

        sparsity = 0.01 * mask.sum()

        loss = fidelity_loss + sparsity

        loss.backward()
        opt.step()

    return edge_mask_model().detach().cpu(), pred


# ---------------------------
# main
# ---------------------------
def main():

    model, data, mappings, device = load()

    inv_labels = {v: k for k, v in mappings["labels_dict"].items()}

    nodes = data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()

    results = []

    for n in nodes:

        mask, pred = explain(model, data, n, device)

        true = inv_labels[int(
            data.test_y[(data.test_idx == n).nonzero()[0, 0]]
        )]

        print(n, "pred:", pred, "true:", true)

        results.append({
            "node": int(n),
            "pred": pred,
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