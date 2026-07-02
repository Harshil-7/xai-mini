"""
Step 4 -- Generate local explanations with GNNExplainer.

Output:
    results/figures/explanation_<entity>.png
    results/tables/explanations.csv
"""

import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
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

CLASS_COLORS = {
    "Athlete":       "#2196F3",
    "MusicalArtist": "#E91E63",
    "Politician":    "#FF9800",
    "Scientist":     "#4CAF50",
}
 
 
def _truncate(text, max_len=28):
    """Truncate long node labels so they fit inside graph nodes."""
    return text if len(text) <= max_len else text[:max_len - 1] + "…"
 
 
def visualize_explanation_human(entity_name, sentences, pred_label, true_label, fig_path):
    """
    Build a directed graph from the human-readable `sentences` list and
    draw it with matplotlib + networkx.
 
    Each sentence has the form:
        "SRC_NAME --[PREDICATE]--> DST_NAME"
    We parse that directly — no node indices involved.
    """
    G = nx.DiGraph()
    edge_labels = {}
 
    for sentence in sentences:
        # parse "A --[rel]--> B"
        try:
            left, right = sentence.split("-->", 1)
            src_part, rel_part = left.rsplit("--[", 1)
            src  = src_part.strip()
            rel  = rel_part.rstrip("]").strip()
            dst  = right.strip()
        except ValueError:
            continue   # skip malformed sentences
 
        # truncate very long object names for readability
        src_short = _truncate(src)
        dst_short = _truncate(dst)
 
        G.add_node(src_short)
        G.add_node(dst_short)
        G.add_edge(src_short, dst_short, label=rel)
        edge_labels[(src_short, dst_short)] = rel
 
    if len(G.nodes) == 0:
        return   # nothing to draw
 
    entity_short = _truncate(entity_name)
    correct      = (pred_label == true_label)
    entity_color = CLASS_COLORS.get(pred_label, "#9C27B0")
    node_colors  = [
        entity_color if n == entity_short else "#ECEFF1"
        for n in G.nodes
    ]
 
    # layout: put the entity node at centre, spread neighbours around it
    if entity_short in G.nodes:
        fixed_positions = {entity_short: (0, 0)}
        pos = nx.spring_layout(G, seed=42, k=2.2, pos=fixed_positions, fixed=[entity_short])
    else:
        pos = nx.spring_layout(G, seed=42, k=2.2)
 
    fig, ax = plt.subplots(figsize=(10, 7))
 
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=2200,
        edgecolors="#455A64",
        linewidths=1.5,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=7.5,
        font_color="#212121",
        font_weight="bold",
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#607D8B",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=1.8,
        connectionstyle="arc3,rad=0.08",
        min_source_margin=22,
        min_target_margin=22,
    )
    nx.draw_networkx_edge_labels(
        G, pos, ax=ax,
        edge_labels=edge_labels,
        font_size=7,
        font_color="#B71C1C",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
    )
 
    verdict = "✓ correct" if correct else f"✗ predicted {pred_label}"
    ax.set_title(
        f"GNNExplainer — {entity_name}\n"
        f"True class: {true_label}   |   {verdict}",
        fontsize=11, pad=12,
    )
 
    # legend: entity colour = predicted class
    patch = mpatches.Patch(color=entity_color, label=f"Central entity ({pred_label})")
    bg    = mpatches.Patch(color="#ECEFF1",    label="Neighbour node", ec="#455A64")
    ax.legend(handles=[patch, bg], loc="lower left", fontsize=8)
 
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


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
        visualize_explanation_human(entity_name, sentences, pred_label, true_label, fig_path)

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