"""
Step 4 -- Generate local explanations with GNNExplainer.

Output:
    results/figures/explanation_<entity>.png
    results/tables/explanations.csv
"""

import pickle

import pandas as pd
import torch
from torch_geometric.explain import Explainer, GNNExplainer

import config
from step3_train_rgcn import FastRGCN


def short_name(uri):
    name = uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    return name.replace("_", " ")


def load_model_and_data():
    checkpoint = torch.load(config.MODEL_FILE, weights_only=False)
    with open(config.MODEL_FILE + ".mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    model = FastRGCN(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, checkpoint["data"], mappings


def explain_node(explainer, data, node_index):
    explanation = explainer(
        x=data.x, edge_index=data.edge_index, edge_type=data.edge_type, index=node_index,
    )
    return explanation.get_explanation_subgraph()


def describe_explanation(sub, node_index, inv_nodes, inv_relations, top_k):
    d = sub.to_dict()
    edge_mask = d["edge_mask"]
    sorted_idx = edge_mask.argsort(descending=True)[:top_k]

    entity_name = short_name(inv_nodes[node_index])
    sentences = []
    for i in sorted_idx:
        src = int(d["edge_index"][0][i])
        dst = int(d["edge_index"][1][i])
        rel = int(d["edge_type"][i])

        base_rel = rel // 2
        predicate_uri = inv_relations[base_rel]
        predicate_name = short_name(predicate_uri)
        is_inverse = (rel % 2 == 1)

        src_name = short_name(inv_nodes.get(src, str(src)))
        dst_name = short_name(inv_nodes.get(dst, str(dst)))

        if is_inverse:
            sentence = f"{dst_name} --[{predicate_name}]--> {src_name}"
        else:
            sentence = f"{src_name} --[{predicate_name}]--> {dst_name}"
        sentences.append(sentence)

    return entity_name, sentences


def main():
    print("[step4] Loading trained model ...")
    model, data, mappings = load_model_and_data()

    inv_nodes = {v: k for k, v in mappings["nodes_dict"].items()}
    inv_relations = {v: k for k, v in mappings["relations_dict"].items()}
    inv_labels = {v: k for k, v in mappings["labels_dict"].items()}

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=config.GNN_EXPLAINER_EPOCHS),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="log_probs"),
        threshold_config=dict(threshold_type="topk", value=config.TOP_K_EDGES),
    )

    nodes_to_explain = data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()

    records = []
    for node_index in nodes_to_explain:
        print(f"[step4] Explaining node {node_index} ({short_name(inv_nodes[node_index])}) ...")
        sub = explain_node(explainer, data, node_index)
        entity_name, sentences = describe_explanation(sub, node_index, inv_nodes, inv_relations, config.TOP_K_EDGES)

        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.edge_type)[node_index].argmax().item()
        pred_label = inv_labels[pred]
        true_label = inv_labels[int(data.test_y[(data.test_idx == node_index).nonzero()[0, 0]])]

        print(f"  Predicted: {pred_label} | True: {true_label}")
        for s in sentences:
            print(f"    {s}")

        fig_path = f"{config.RESULTS_FIGURES_DIR}/explanation_{entity_name.replace(' ', '_')}.png"
        sub.visualize_graph(fig_path)

        records.append({
            "entity": entity_name,
            "predicted_label": pred_label,
            "true_label": true_label,
            "explanation_edges": " | ".join(sentences),
            "node_index": node_index,
        })

    pd.DataFrame(records).to_csv(f"{config.RESULTS_TABLES_DIR}/explanations.csv", index=False)
    print(f"[step4] Saved {len(records)} explanations to results/tables/explanations.csv")


if __name__ == "__main__":
    main()