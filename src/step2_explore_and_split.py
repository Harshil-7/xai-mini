"""
Step 2 -- Explore the dataset and create the train/test split.

Output:
    results/tables/dataset_stats.csv
    results/figures/predicate_frequency.png
    results/figures/class_distribution.png
    data/raw/train.tsv
    data/raw/test.tsv
"""

import gzip
import random
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import rdflib

import config


def load_graph():
    graph = rdflib.Graph()
    with gzip.open(config.GRAPH_FILE, "rb") as f:
        graph.parse(file=f, format="nt")
    return graph


def compute_stats(graph):
    subjects = set(graph.subjects())
    objects = set(graph.objects())
    nodes = subjects | objects
    predicates = Counter(graph.predicates())

    stats = {
        "num_triples": len(graph),
        "num_nodes": len(nodes),
        "num_subjects": len(subjects),
        "num_objects": len(objects),
        "num_distinct_predicates": len(predicates),
    }
    return stats, predicates


def plot_predicate_frequency(predicates, top_n=15):
    most_common = predicates.most_common(top_n)
    names = [str(p).split("/")[-1].split("#")[-1] for p, _ in most_common]
    counts = [c for _, c in most_common]

    plt.figure(figsize=(8, 5))
    plt.barh(names[::-1], counts[::-1])
    plt.xlabel("Number of triples")
    plt.title(f"Top {top_n} most frequent predicates")
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_FIGURES_DIR}/predicate_frequency.png", dpi=150)
    plt.close()


def plot_class_distribution(labels_df):
    counts = labels_df["label"].value_counts()
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.ylabel("Number of entities")
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_FIGURES_DIR}/class_distribution.png", dpi=150)
    plt.close()


def make_split(labels_df):
    random.seed(config.RANDOM_SEED)
    train_rows, test_rows = [], []

    for label in labels_df["label"].unique():
        subset = labels_df[labels_df["label"] == label].sample(
            frac=1.0, random_state=config.RANDOM_SEED
        )
        cut = int(len(subset) * config.TRAIN_TEST_SPLIT)
        train_rows.append(subset.iloc[:cut])
        test_rows.append(subset.iloc[cut:])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df


def main():
    print("[step2] Loading graph ...")
    graph = load_graph()

    print("[step2] Computing dataset statistics ...")
    stats, predicates = compute_stats(graph)
    pd.DataFrame([stats]).to_csv(f"{config.RESULTS_TABLES_DIR}/dataset_stats.csv", index=False)
    print(stats)

    plot_predicate_frequency(predicates)

    labels_df = pd.read_csv(config.LABELS_FILE, sep="\t")
    plot_class_distribution(labels_df)

    print("[step2] Creating stratified train/test split ...")
    train_df, test_df = make_split(labels_df)
    train_df.to_csv(config.TRAIN_FILE, sep="\t", index=False)
    test_df.to_csv(config.TEST_FILE, sep="\t", index=False)
    print(f"[step2] Train: {len(train_df)} entities, Test: {len(test_df)} entities")
    print("[step2] Done. See results/tables/dataset_stats.csv and results/figures/*.png")


if __name__ == "__main__":
    main()