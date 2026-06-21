"""
Step 3 -- Build the PyG graph and train an R-GCN node classifier.

This follows the same recipe as the lecture's AIFB/PyTorch Geometric
example (FastRGCNConv + one-hot degree features), just pointed at our
DBpedia subgraph instead of AIFB.

IMPORTANT: we must remove the triples that directly state the target
class (e.g. "<X> rdf:type dbo:Scientist") before building node
features. Otherwise the model would just read the label out of its
own input instead of learning anything -- exactly the trap the AIFB
example avoids by removing label_affiliation from the feature table.

Output:
    data/processed/rgcn_model.pt        -- trained model + all mappings
    results/tables/model_performance.csv
"""

import gzip
import pickle
import time

import numpy as np
import pandas as pd
import rdflib
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import FastRGCNConv
from torch_geometric.utils import degree, index_sort

import config

torch_geometric.seed_everything(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)


def load_filtered_graph():
    """Load the cached graph and strip out the label-revealing triples."""
    graph = rdflib.Graph()
    with gzip.open(config.GRAPH_FILE, "rb") as f:
        graph.parse(file=f, format="nt")

    label_objects = set(config.TARGET_CLASSES.values())
    label_predicate = rdflib.URIRef(config.LABEL_PREDICATE)

    to_remove = [
        (s, p, o) for s, p, o in graph
        if p == label_predicate and str(o) in label_objects
    ]
    for triple in to_remove:
        graph.remove(triple)

    print(f"[step3] Removed {len(to_remove)} label-revealing triples from features.")
    return graph


def build_pyg_data(graph, labels_df):
    """Turn the rdflib graph + label table into a PyG Data object."""
    freq = {}
    for p in graph.predicates():
        freq[p] = freq.get(p, 0) + 1
    relations = sorted(set(graph.predicates()), key=lambda p: -freq.get(p, 0))

    subjects = sorted(set(graph.subjects()))
    objects = sorted(set(graph.objects()))
    nodes = sorted(set(subjects) | set(objects))

    relations_dict = {rel: i for i, rel in enumerate(relations)}
    nodes_dict = {str(node): i for i, node in enumerate(nodes)}

    edges = []
    for s, p, o in graph.triples((None, None, None)):
        src, dst = nodes_dict[str(s)], nodes_dict[str(o)]
        rel = relations_dict[p]
        edges.append([src, dst, 2 * rel])
        edges.append([dst, src, 2 * rel + 1])  # inverse relation

    edge = torch.tensor(edges, dtype=torch.long).t().contiguous()
    N, R = len(nodes), 2 * len(relations)
    _, perm = index_sort(N * R * edge[0] + R * edge[1] + edge[2])
    edge = edge[:, perm]
    edge_index, edge_type = edge[:2], edge[2]

    # Node features: clamped one-hot degree.
    #
    # Real-world DBpedia data has a few extreme hub nodes (e.g. common
    # countries, common rdf:type classes) with degree in the
    # thousands. Plain OneHotDegree assumes no node exceeds max_degree
    # and crashes if one does, so we clamp degree to
    # config.MAX_DEGREE_FEATURES first -- any node at or above that cap
    # just lands in one shared "high-degree" bucket instead of
    # crashing. This is a reasonable simplification for a baseline
    # feature; see report's "own contribution" section if you replace
    # it with something richer (e.g. RDF2Vec embeddings).
    deg = degree(edge_index[0], num_nodes=N).long()
    deg_clamped = torch.clamp(deg, max=config.MAX_DEGREE_FEATURES)
    X = F.one_hot(deg_clamped, num_classes=config.MAX_DEGREE_FEATURES + 1).float()

    n_clamped = int((deg >= config.MAX_DEGREE_FEATURES).sum())
    if n_clamped:
        print(f"[step3] {n_clamped} nodes had degree >= {config.MAX_DEGREE_FEATURES} "
              f"and were clamped into the top bucket.")

    label_names = sorted(labels_df["label"].unique())
    labels_dict = {lab: i for i, lab in enumerate(label_names)}

    def entity_indices_and_labels(df):
        idxs, labs = [], []
        for entity, label in zip(df["entity"], df["label"]):
            if entity in nodes_dict:  # entity might have 0 outgoing triples
                idxs.append(nodes_dict[entity])
                labs.append(labels_dict[label])
        return torch.tensor(idxs, dtype=torch.long), torch.tensor(labs, dtype=torch.long)

    train_df = pd.read_csv(config.TRAIN_FILE, sep="\t")
    test_df = pd.read_csv(config.TEST_FILE, sep="\t")
    train_idx, train_y = entity_indices_and_labels(train_df)
    test_idx, test_y = entity_indices_and_labels(test_df)

    data = Data(x=X, edge_index=edge_index, edge_type=edge_type,
                train_idx=train_idx, train_y=train_y,
                test_idx=test_idx, test_y=test_y, num_nodes=N)

    mappings = {
        "nodes_dict": nodes_dict,
        "relations_dict": relations_dict,
        "labels_dict": labels_dict,
        "num_relations": int(edge_type.max()) + 1,
        "num_classes": len(label_names),
        "in_channels": X.shape[1],
    }
    return data, mappings


class FastRGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_relations, num_bases):
        super().__init__()
        self.conv1 = FastRGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases)
        self.conv2 = FastRGCNConv(hidden_channels, num_classes, num_relations, num_bases=num_bases)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


def train_and_evaluate(model, data, device):
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)

    history = []
    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_type)
        loss = F.nll_loss(out[data.train_idx], data.train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.edge_type).argmax(dim=-1)
            train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
            test_acc = float((pred[data.test_idx] == data.test_y).float().mean())

        history.append({"epoch": epoch, "loss": float(loss), "train_acc": train_acc, "test_acc": test_acc})
        if epoch % 10 == 0 or epoch == config.NUM_EPOCHS:
            print(f"  epoch {epoch:02d}  loss={loss:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    return pd.DataFrame(history)


def main():
    print("[step3] Loading and filtering graph ...")
    graph = load_filtered_graph()
    labels_df = pd.read_csv(config.LABELS_FILE, sep="\t")

    print("[step3] Building PyG Data object ...")
    data, mappings = build_pyg_data(graph, labels_df)
    print(f"[step3] Graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} directed edges, "
          f"{mappings['num_relations']} relation types, {mappings['num_classes']} classes")

    model = FastRGCN(mappings["in_channels"], config.HIDDEN_CHANNELS,
                      mappings["num_classes"], mappings["num_relations"], config.NUM_BASES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[step3] Training on {device} ...")
    start = time.time()
    history = train_and_evaluate(model, data, device)
    print(f"[step3] Training took {time.time() - start:.1f}s")

    history.to_csv(f"{config.RESULTS_TABLES_DIR}/model_performance.csv", index=False)

    final = history.iloc[-1]
    print(f"[step3] Final train_acc={final['train_acc']:.4f}  test_acc={final['test_acc']:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "model_args": {
            "in_channels": mappings["in_channels"],
            "hidden_channels": config.HIDDEN_CHANNELS,
            "num_classes": mappings["num_classes"],
            "num_relations": mappings["num_relations"],
            "num_bases": config.NUM_BASES,
        },
        "data": data,
    }, config.MODEL_FILE)

    with open(config.MODEL_FILE + ".mappings.pkl", "wb") as f:
        pickle.dump(mappings, f)

    print(f"[step3] Saved model to {config.MODEL_FILE}")


if __name__ == "__main__":
    main()