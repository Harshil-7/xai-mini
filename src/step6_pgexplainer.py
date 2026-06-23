"""
Step 6 -- Generate local explanations with PGExplainer.

Fixes applied vs. the previous version:
  1. OOM fix: every explain step now runs on a k-hop LOCAL SUBGRAPH around
     the target node instead of the full graph. Message passing is local,
     so this gives identical results to running on the full graph as long
     as NUM_HOPS == number of RGCNConv layers in FastRGCN -- check that.
  2. PGExplainer is actually trained now (it's a *parameterized* explainer;
     without a .train() pass its mask-predicting MLP is just random).
  3. explanation_type="phenomenon" -- PyG's PGExplainer ONLY supports this
     mode (explanation_type="model" raises a ValueError, as you saw). To
     still explain "why did the model predict this" rather than "why is
     the true label this", we feed the model's OWN prediction in as the
     `target` for both training and explaining, computed per-subgraph
     (never on the full graph, to avoid the earlier OOM). If you'd rather
     explain against true labels instead, swap target_sub below for
     data.train_y / data.test_y indexed the same way.

Output:
    results/figures/explanation_<entity>.png
    results/tables/explanations.csv
"""

import pickle
import random

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, PGExplainer
from torch_geometric.utils import k_hop_subgraph

import config
from torch_geometric.nn import RGCNConv


class RGCNForExplain(torch.nn.Module):
    """
    Same architecture as FastRGCN in step3_train_rgcn.py, but using
    RGCNConv instead of FastRGCNConv for inference/explanation here.

    FastRGCNConv is a subclass of RGCNConv with identical learned
    parameters (weight, comp, root, bias) -- only the internal
    message-passing computation differs. FastRGCNConv builds a dense
    [num_edges, num_relations] one-hot matrix for its degree
    normalization, which is what was running out of memory (your
    relation count x edge count made that matrix ~2.6GB in one shot).
    RGCNConv loops over relations instead, so it doesn't need that
    allocation. The trained weights load here directly -- no retraining.
    """
    def __init__(self, in_channels, hidden_channels, num_classes, num_relations, num_bases):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_channels, num_classes, num_relations, num_bases=num_bases)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


# Must match the number of RGCNConv layers in FastRGCN -- verify in
# step3_train_rgcn.py and update if it's not 2.
NUM_HOPS = 2

# Cap how many train nodes PGExplainer trains its MLP on per epoch.
# Training on all 1920 nodes x 30 epochs, even with subgraphs, can be slow;
# a few hundred sampled nodes per epoch is usually enough for the MLP to
# generalize.
MAX_EXPLAINER_TRAIN_NODES = 300


def short_name(uri):
    name = uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    return name.replace("_", " ")


def load_model_and_data():
    checkpoint = torch.load(config.MODEL_FILE, weights_only=False)
    with open(config.MODEL_FILE + ".mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    model = RGCNForExplain(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, checkpoint["data"], mappings


def get_local_subgraph(node_index, data, num_hops=NUM_HOPS):
    """
    Restrict (x, edge_index, edge_type) to the num_hops-hop neighborhood of
    node_index. Returns relabeled tensors plus:
      - local_idx: node_index's new position inside the subgraph
      - subset:    original (global) node ids kept, in subgraph order --
                    use subset[local_id] to map back to global ids.
    """
    subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_index,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.x.size(0),
    )
    x_sub = data.x[subset]
    edge_type_sub = data.edge_type[edge_mask]
    local_idx = mapping.item()
    return x_sub, edge_index_sub, edge_type_sub, local_idx, subset


def explain_node(explainer, model, data, node_index):
    x_sub, ei_sub, et_sub, local_idx, subset = get_local_subgraph(node_index, data)

    # target_sub: the model's own predicted class for every node in this
    # subgraph (computed once, from the subgraph only -- never the full
    # graph). PGExplainer requires explanation_type="phenomenon", which
    # needs a `target`; using the model's own prediction here keeps the
    # original intent of "explain what the model predicted".
    with torch.no_grad():
        target_sub = model(x_sub, ei_sub, et_sub).argmax(dim=1)
    pred = int(target_sub[local_idx])

    explanation = explainer(
        x=x_sub,
        edge_index=ei_sub,
        edge_type=et_sub,
        index=local_idx,
        target=target_sub,
    )

    return explanation.get_explanation_subgraph(), subset, pred


def describe_explanation(sub, subset, inv_nodes, inv_relations, top_k):
    d = sub.to_dict()
    edge_mask = d["edge_mask"]
    sorted_idx = edge_mask.argsort(descending=True)[:top_k]

    sentences = []
    for i in sorted_idx:
        src_local = int(d["edge_index"][0][i])
        dst_local = int(d["edge_index"][1][i])
        rel = int(d["edge_type"][i])

        # Map subgraph-local node ids back to global ids before lookup
        src = int(subset[src_local])
        dst = int(subset[dst_local])

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

    return sentences


def train_explainer(explainer, model, data):
    train_nodes = data.train_idx.tolist()
    if len(train_nodes) > MAX_EXPLAINER_TRAIN_NODES:
        train_nodes = random.sample(train_nodes, MAX_EXPLAINER_TRAIN_NODES)

    print(f"[PG] Training PGExplainer on {len(train_nodes)} sampled nodes ...")
    for epoch in range(30):
        for node_index in train_nodes:
            x_sub, ei_sub, et_sub, local_idx, _ = get_local_subgraph(node_index, data)
            with torch.no_grad():
                target_sub = model(x_sub, ei_sub, et_sub).argmax(dim=1)
            explainer.algorithm.train(
                epoch,
                model,
                x_sub,
                ei_sub,
                target=target_sub,
                index=local_idx,
                edge_type=et_sub,
            )


def main():
    print("[step4] Loading trained model ...")
    model, data, mappings = load_model_and_data()
    # NOTE: predictions are now computed per-node from local subgraphs inside
    # the explain loop below, rather than with one upfront full-graph forward
    # pass -- that full-graph call is what was crashing (see explanation in
    # the module docstring / chat).

    inv_nodes = {v: k for k, v in mappings["nodes_dict"].items()}
    inv_relations = {v: k for k, v in mappings["relations_dict"].items()}
    inv_labels = {v: k for k, v in mappings["labels_dict"].items()}

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(
            epochs=30,
            lr=0.003,
        ),
        explanation_type="phenomenon",  # the only mode PGExplainer supports
        node_mask_type=None,
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="log_probs",  # model ends in F.log_softmax, not raw logits
        ),
    )

    train_explainer(explainer, model, data)

    nodes_to_explain = data.test_idx[:config.NUM_NODES_TO_EXPLAIN].tolist()

    records = []
    for node_index in nodes_to_explain:
        entity_name = short_name(inv_nodes[node_index])
        print(f"[step4] Explaining node {node_index} ({entity_name}) ...")

        sub, subset, pred = explain_node(explainer, model, data, node_index)
        sentences = describe_explanation(sub, subset, inv_nodes, inv_relations, config.TOP_K_EDGES)

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