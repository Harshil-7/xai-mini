import torch
import torch.nn.functional as F
import pandas as pd
import pickle

import config
from step3_train_rgcn import FastRGCN
from torch_geometric.explain import Explainer, PGExplainer


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


def main():
    model, data, mappings, device = load()

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=30, lr=0.003),
        explanation_type="phenomenon",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="log_probs",
        ),
    ).to(device)

    nodes = data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()

    records = []

    print("[PG] Training PGExplainer...")
    
    # IMPORTANT: training happens on FULL GRAPH
    for node in nodes:
        label = model(data.x, data.edge_index, data.edge_type)[node].argmax()

        explainer.algorithm.train(
            model=model,
            x=data.x,
            edge_index=data.edge_index,
            target=label
        )

    print("[PG] Generating explanations...")

    for node in nodes:
        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            index=node,
        )

        edge_mask = explanation.edge_mask

        records.append({
            "node": int(node),
            "mask_mean": float(edge_mask.mean()),
        })

        print(f"Node {node} done")

    pd.DataFrame(records).to_csv(
        f"{config.RESULTS_TABLES_DIR}/pg_explanations.csv",
        index=False
    )

    print("DONE")


if __name__ == "__main__":
    main()