"""
Step 1 -- Extract a small, task-specific subgraph from DBpedia.

We do NOT download the full DBpedia dump (it's billions of triples).
Instead we query the public SPARQL endpoint for a few hundred entities
per target class, plus their 1-hop neighborhood, and cache the result
locally. This script is idempotent: if the cache files already exist it
skips straight to the end, so re-running run_all.sh doesn't hammer the
public endpoint again.

Output:
    data/raw/dbpedia_subset.nt.gz   -- the RDF graph (gzipped N-Triples)
    data/raw/labels.tsv             -- entity -> class label
"""

import gzip
import os
import time

import rdflib
from SPARQLWrapper import SPARQLWrapper, JSON

import config


def run_select(endpoint, query, retries=3, backoff=5):
    """Run a SELECT query and return JSON bindings, retrying on failure.

    The public DBpedia endpoint occasionally times out or rate-limits.
    A simple retry with backoff makes the extraction much more robust
    than a single bare request.
    """
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    for attempt in range(1, retries + 1):
        try:
            return sparql.query().convert()["results"]["bindings"]
        except Exception as exc:  # noqa: BLE001 -- we want to retry on anything
            print(f"  [warn] query failed (attempt {attempt}/{retries}): {exc}")
            if attempt == retries:
                raise
            time.sleep(backoff)


def get_entities_of_class(class_uri, limit):
    """Return up to `limit` entity URIs that are rdf:type `class_uri`."""
    query = f"""
    SELECT DISTINCT ?entity WHERE {{
        ?entity a <{class_uri}> .
    }} LIMIT {limit}
    """
    bindings = run_select(config.SPARQL_ENDPOINT, query)
    return [b["entity"]["value"] for b in bindings]


def fetch_triples_for_entities(entities, batch_size):
    """Fetch all outgoing triples for a list of entities, in batches.

    Returns a list of (subject, predicate, object_value, object_type)
    tuples, where object_type is "uri" or "literal" -- taken directly
    from the SPARQL JSON response. This is the only reliable way to
    tell URIs and literals apart: guessing from the string (e.g.
    "starts with http") is wrong, because some DBpedia literals are
    plain-text values that also start with "http" (e.g. a description
    string containing a URL plus extra text).
    """
    triples = []
    for start in range(0, len(entities), batch_size):
        batch = entities[start:start + batch_size]
        values_clause = " ".join(f"<{e}>" for e in batch)
        query = f"""
        SELECT ?s ?p ?o WHERE {{
            VALUES ?s {{ {values_clause} }}
            ?s ?p ?o .
            FILTER(isURI(?o) || isLiteral(?o))
        }}
        """
        bindings = run_select(config.SPARQL_ENDPOINT, query)
        for b in bindings:
            triples.append((b["s"]["value"], b["p"]["value"], b["o"]["value"], b["o"]["type"]))
        print(f"  fetched triples for entities {start + len(batch)}/{len(entities)}")
    return triples


def add_triples_to_graph(graph, triples, class_name):
    """Add (subject, predicate, object_value, object_type) tuples to the
    rdflib graph, skipping any triple the endpoint returned that rdflib
    genuinely cannot represent (rare, but better to skip one bad triple
    than lose the whole extraction at serialization time)."""
    skipped = 0
    for s, p, o_value, o_type in triples:
        try:
            obj = rdflib.URIRef(o_value) if o_type == "uri" else rdflib.Literal(o_value)
            graph.add((rdflib.URIRef(s), rdflib.URIRef(p), obj))
        except Exception:  # noqa: BLE001
            skipped += 1
    if skipped:
        print(f"  [warn] skipped {skipped} malformed triples for class {class_name}")


def main():
    if os.path.exists(config.GRAPH_FILE) and os.path.exists(config.LABELS_FILE):
        print(f"[step1] Cached graph already exists at {config.GRAPH_FILE}, skipping extraction.")
        print("        Delete data/raw/ if you want to re-extract from DBpedia.")
        return

    graph = rdflib.Graph()
    labels = []  # list of (entity, class_name)

    for class_name, class_uri in config.TARGET_CLASSES.items():
        print(f"[step1] Querying entities of type {class_name} ...")
        entities = get_entities_of_class(class_uri, config.ENTITIES_PER_CLASS)
        print(f"[step1] Got {len(entities)} entities, fetching their triples ...")
        triples = fetch_triples_for_entities(entities, config.SPARQL_BATCH_SIZE)

        add_triples_to_graph(graph, triples, class_name)

        for e in entities:
            labels.append((e, class_name))

    print(f"[step1] Total triples collected: {len(graph)}")
    print(f"[step1] Total labeled entities: {len(labels)}")

    try:
        with gzip.open(config.GRAPH_FILE, "wb") as f:
            graph.serialize(destination=f, format="nt", encoding="utf-8")
    except Exception as exc:
        print(f"[step1] ERROR during serialization: {exc}")
        print(f"[step1] Graph still has {len(graph)} triples collected in memory.")
        print("[step1] Nothing was written to disk -- fix the issue and rerun.")
        raise

    with open(config.LABELS_FILE, "w") as f:
        f.write("entity\tlabel\n")
        for e, lab in labels:
            f.write(f"{e}\t{lab}\n")

    print(f"[step1] Saved graph to {config.GRAPH_FILE}")
    print(f"[step1] Saved labels to {config.LABELS_FILE}")


if __name__ == "__main__":
    main()