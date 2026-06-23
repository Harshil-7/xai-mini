import torch
import torch.nn.functional as F
import pandas as pd
import pickle

import config
from step3_train_rgcn import FastRGCN
from torch_geometric.explain import Explainer, PGExplainer


# -----------------------------
# load model + data
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
# main
# -----------------------------
def main():
    model, data, mappings, device = load()

    inv_nodes = {v: k for k, v in mappings["nodes_dict"].items()}
    inv_labels = {v: k for k, v in mappings["labels_dict"].items()}

    # -----------------------------
    # PGExplainer setup (IMPORTANT FIX)
    # -----------------------------
    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(
            epochs=30,
            lr=0.003,
        ),
        explanation_type="phenomenon",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="log_probs",
        ),
    )

    nodes = data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()

    print("[PG] Training PGExplainer...")

    # -----------------------------
    # TRAIN (FIXED API)
    # -----------------------------
    for epoch in range(30):
        for n in nodes:
            with torch.no_grad():
                out = model(data.x, data.edge_index, data.edge_type)
                target = out[n].argmax()

            explainer.algorithm.train(
                epoch,
                model,
                data.x,
                data.edge_index,
                target=target,
            )

    print("[PG] Generating explanations...")

    results = []

    # -----------------------------
    # EXPLAIN
    # -----------------------------
    for n in nodes:
        exp = explainer(
            x=data.x,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            index=n,
        )

        edge_mask = exp.edge_mask

        true = inv_labels[int(
            data.test_y[(data.test_idx == n).nonzero()[0, 0]]
        )]

        pred = model(data.x, data.edge_index, data.edge_type)[n].argmax().item()

        print(f"{n} | pred={pred} | true={true} | mask_mean={edge_mask.mean().item():.4f}")

        results.append({
            "node": int(n),
            "pred": int(pred),
            "true": true,
            "mask_mean": float(edge_mask.mean()),
        })

    pd.DataFrame(results).to_csv(
        f"{config.RESULTS_TABLES_DIR}/pg_explanations.csv",
        index=False
    )

    print("DONE")


if __name__ == "__main__":
    main()