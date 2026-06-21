"""
Quick inspection script for the data produced by step 1 and step 2.
Not part of the graded pipeline (not called by run_all.sh) -- just for
you to sanity-check the data before moving on to step 3.

Run with:
    python src/inspect_data.py
"""

import gzip

import pandas as pd
import rdflib

import config

print("=" * 60)
print("1. labels.tsv  (all entities + their class)")
print("=" * 60)
labels_df = pd.read_csv(config.LABELS_FILE, sep="\t")
print(labels_df.head(10))
print(f"\nTotal entities: {len(labels_df)}")
print("\nClass distribution:")
print(labels_df["label"].value_counts())

print("\n" + "=" * 60)
print("2. train.tsv / test.tsv")
print("=" * 60)
train_df = pd.read_csv(config.TRAIN_FILE, sep="\t")
test_df = pd.read_csv(config.TEST_FILE, sep="\t")
print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
print("\nTrain class distribution:")
print(train_df["label"].value_counts())
print("\nTest class distribution:")
print(test_df["label"].value_counts())

print("\n" + "=" * 60)
print("3. dataset_stats.csv")
print("=" * 60)
stats_df = pd.read_csv(f"{config.RESULTS_TABLES_DIR}/dataset_stats.csv")
print(stats_df.T)

print("\n" + "=" * 60)
print("4. Raw RDF graph (dbpedia_subset.nt.gz)")
print("=" * 60)
graph = rdflib.Graph()
with gzip.open(config.GRAPH_FILE, "rb") as f:
    graph.parse(file=f, format="nt")
print(f"Total triples in graph: {len(graph)}")

print("\nFirst 10 raw triples:")
for i, (s, p, o) in enumerate(graph):
    if i >= 10:
        break
    print(f"  {s}\n    -> {p}\n    -> {o}\n")

print("=" * 60)
print("5. All triples for ONE sample entity")
print("=" * 60)
sample_entity = labels_df.iloc[0]["entity"]
print(f"Entity: {sample_entity} (label: {labels_df.iloc[0]['label']})")
count = 0
for s, p, o in graph.triples((rdflib.URIRef(sample_entity), None, None)):
    print(f"  {p}  ->  {o}")
    count += 1
print(f"\nTotal outgoing triples for this entity: {count}")

print("\n" + "=" * 60)
print("6. Sanity check: is the label-revealing triple still in the graph?")
print("=" * 60)
label_predicate = rdflib.URIRef(config.LABEL_PREDICATE)
found = list(graph.triples((rdflib.URIRef(sample_entity), label_predicate, None)))
class_uris = set(config.TARGET_CLASSES.values())
leaking = [t for t in found if str(t[2]) in class_uris]
if leaking:
    print(f"[WARNING] Found {len(leaking)} label-revealing triples still in the raw graph "
          f"(this is fine -- step 3 removes them at training time, not here):")
    for t in leaking:
        print(f"  {t}")
else:
    print("No direct label-revealing triple found for this entity (it may have been the only "
          "rdf:type triple and got removed, or it simply has other types only).")