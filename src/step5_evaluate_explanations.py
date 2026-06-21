"""
Step 5 -- Quantitative evaluation of explanations (fidelity, sparsity).

Output:
    results/tables/explanation_evaluation.csv
"""

import pickle

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer

import config
from step3_train_rgcn import FastRGCN


def load_model_and_data():
    checkpoint = torch.load(config.MODEL_FILE, weights_only=False)
    with open(config.MODEL_FILE + ".mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    model = FastRGCN(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["data"], mappings


def class_probability(model, x, edge_index, edge_type, node_index, class_id):
    with torch.no_grad():
        log_probs = model(x, edge_index, edge_type)
        return float(F.softmax(log_probs[node_index], dim=0)[class_id])


def evaluate_node(model, data, explainer, node_index):
    explanation = explainer(x=data.x, edge_index=data.edge_index, edge_type=data.edge_type, index=node_index)
    keep_mask = explanation.edge_mask > 0

    with torch.no_grad():
        pred_class = int(model(data.x, data.edge_index, data.edge_type)[node_index].argmax())

    full_prob = class_probability(model, data.x, data.edge_index, data.edge_type, node_index, pred_class)

    edge_index_minus = data.edge_index[:, ~keep_mask]
    edge_type_minus = data.edge_type[~keep_mask]
    prob_minus = class_probability(model, data.x, edge_index_minus, edge_type_minus, node_index, pred_class)
    fidelity_plus = full_prob - prob_minus

    edge_index_only = data.edge_index[:, keep_mask]
    edge_type_only = data.edge_type[keep_mask]
    prob_only = class_probability(model, data.x, edge_index_only, edge_type_only, node_index, pred_class)
    fidelity_minus = full_prob - prob_only

    total_edges = keep_mask.numel()
    sparsity = 1.0 - (int(keep_mask.sum()) / total_edges) if total_edges > 0 else float("nan")

    return {
        "node_index": node_index,
        "predicted_class": pred_class,
        "full_prob": full_prob,
        "fidelity_plus": fidelity_plus,
        "fidelity_minus": fidelity_minus,
        "sparsity": sparsity,
    }


def main():
    print("[step5] Loading model ...")
    model, data, mappings = load_model_and_data()

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=config.GNN_EXPLAINER_EPOCHS),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="log_probs"),
        threshold_config=dict(threshold_type="topk", value=config.TOP_K_EDGES),
    )

    nodes_to_evaluate = data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()

    records = []
    for node_index in nodes_to_evaluate:
        print(f"[step5] Evaluating explanation for node {node_index} ...")
        records.append(evaluate_node(model, data, explainer, node_index))

    df = pd.DataFrame(records)
    df.to_csv(f"{config.RESULTS_TABLES_DIR}/explanation_evaluation.csv", index=False)

    print("\n[step5] Summary (mean over evaluated nodes):")
    print(df[["fidelity_plus", "fidelity_minus", "sparsity"]].mean())
    print("[step5] Saved results/tables/explanation_evaluation.csv")


if __name__ == "__main__":
    main()