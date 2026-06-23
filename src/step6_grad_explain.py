import torch
import torch.nn.functional as F
import pandas as pd
import pickle

import config
from step3_train_rgcn import FastRGCN


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


def edge_attribution(model, x, edge_index, edge_type, target_node, target_class):
    x = x.clone().detach().requires_grad_(True)

    out = model(x, edge_index, edge_type)
    loss = -out[target_node, target_class]

    loss.backward()

    edge_index = edge_index.clone()
    src, dst = edge_index

    node_grad = x.grad.detach()

    scores = (node_grad[src].abs().sum(dim=1) +
              node_grad[dst].abs().sum(dim=1))

    return scores


def main():
    model, data, mappings, device = load()

    nodes = data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()

    results = []

    for n in nodes:
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.edge_type)[n].argmax().item()

        scores = edge_attribution(
            model,
            data.x,
            data.edge_index,
            data.edge_type,
            n,
            pred
        )

        results.append({
            "node": int(n),
            "pred": int(pred),
            "avg_edge_score": float(scores.mean().item())
        })

        print("done node:", n)

    pd.DataFrame(results).to_csv(
        f"{config.RESULTS_TABLES_DIR}/grad_explanations.csv",
        index=False
    )

    print("DONE")


if __name__ == "__main__":
    main()