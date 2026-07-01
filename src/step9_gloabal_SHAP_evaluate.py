"""
Step 9 -- Global SHAP Evaluation
              step8 = explainer  (SHAP + surrogates)
              step9 = evaluation (fidelity/sparsity + human-readable report)

What this file does
--------------------
PART A -- Relation-level fidelity validation (works on ALL classes together)
  1. Builds a relation -> embedding-dimension influence matrix.
  2. Ranks RDF predicates by SHAP-weighted global importance.
  3. Validates the ranking with a fidelity+/fidelity-/sparsity ablation
     sweep (remove top-K predicates, measure confidence drop).
  This answers: "which predicates does the model rely on, globally,
  across ALL classes?"  (wikiPageWikiLink, sameAs, subject, etc.)

PART B -- Class-level differentiation (answers "why THIS class not another")
  Predicate names alone cannot separate classes (every entity uses the
  same wikiPageWikiLink predicate).  What differs is the TARGET of those
  links.  This part:
  4. Extracts keywords from wikiPageWikiLink target page/category names
     per entity (e.g. "Olympic", "engineer", "musician", "constituency").
  5. Scores each keyword's lift per class (how many times more often it
     appears for this class vs the other three).
  6. Validates by stripping edges to top keyword-matched targets and
     measuring the real confidence drop per class.

PART C -- Natural language report
  Combines both parts into one Markdown report a non-technical reader
  can follow end to end.

Output
------
  results/tables/shap_relation_importance.csv
  results/tables/shap_fidelity_evaluation.csv
  results/tables/class_distinctive_keywords.csv
  results/figures/shap_relation_importance.png
  results/figures/shap_fidelity_curve.png
  results/figures/class_distinctive_keywords_<Class>.png
  results/tables/global_explanation_report.md
"""

import gzip
import pickle
import re
import textwrap
import warnings
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdflib
import torch
import torch.nn.functional as F

import config
from step3_train_rgcn import FastRGCN

warnings.filterwarnings("ignore")


# =============================================================================
# Shared constants
# =============================================================================

WIKILINK_PRED = "http://dbpedia.org/ontology/wikiPageWikiLink"

RELATION_PLAIN_ENGLISH = {
    "wikiPageWikiLink":  "a hyperlink between Wikipedia articles",
    "sameAs":            "an owl:sameAs identity link to another knowledge base (e.g. Wikidata, Freebase)",
    "subject":           "a SKOS category tag (e.g. 'Category:American_athletes')",
    "description":       "a short textual description literal",
    "label":             "an rdfs:label string (the entity's display name)",
    "type":              "an rdf:type class assertion",
    "wasBornIn":         "a birthplace link",
    "birthPlace":        "a birthplace link",
    "nationality":       "a nationality link",
    "country":           "a country link",
    "sport":             "a sport link",
    "team":               "a team membership link",
    "genre":             "a music/film genre link",
    "recordLabel":       "a record-label affiliation link",
    "almaMater":         "an alma-mater (university attended) link",
    "field":             "a research field link",
    "award":             "an award link",
    "party":             "a political party membership link",
    "office":            "a held-office link",
}

CLASS_DOMAIN_CONTEXT = {
    "Athlete": (
        "athletes typically appear on Wikipedia pages that link to sports teams, "
        "competitions, stadiums, and coaches.  Their pages receive incoming links "
        "from sports-event articles and team roster pages."
    ),
    "MusicalArtist": (
        "musicians are connected to record labels, albums, music genres, and "
        "concert venues.  Their pages are heavily cross-linked from discography "
        "articles, genre pages, and festival line-up pages."
    ),
    "Politician": (
        "politicians are linked to political parties, government offices, "
        "constituencies, and legislation.  Their pages appear in election results, "
        "parliamentary records, and policy articles."
    ),
    "Scientist": (
        "scientists are connected to universities, research fields, publications, "
        "and prizes.  Their pages are cited from discipline-specific category pages "
        "and from the pages of co-authors or research institutions."
    ),
}

# Keyword extraction stoplist (Part B)
STOPWORDS = {
    "category", "people", "living", "births", "deaths", "from", "the", "of",
    "and", "in", "at", "by", "to", "a", "an", "for", "with", "on", "missing",
    "place", "birth", "death", "year", "21st", "20th", "19th", "century",
    "possibly", "american", "british", "indian",
    "men", "women", "male", "female",
}
YEAR_RE = re.compile(r"^\d{3,4}(s|births|deaths)?$")


def plain_english(rel_name: str) -> str:
    return RELATION_PLAIN_ENGLISH.get(rel_name, f"a '{rel_name}' predicate in DBpedia")


def _wrap(text, width=88):
    return "\n".join(
        textwrap.fill(line, width=width) if line.strip() else line
        for line in text.splitlines()
    )


def short_name(uri: str) -> str:
    return uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


# =============================================================================
# I/O helpers
# =============================================================================

def load_model_and_data():
    checkpoint = torch.load(config.MODEL_FILE, weights_only=False)
    with open(config.MODEL_FILE + ".mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
    model = FastRGCN(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["data"], mappings


def load_shap_importance():
    path = f"{config.RESULTS_TABLES_DIR}/shap_global_importance.csv"
    return pd.read_csv(path)


def load_graph_and_labels():
    print("[step9] Loading raw graph ...")
    graph = rdflib.Graph()
    with gzip.open(config.GRAPH_FILE, "rb") as f:
        graph.parse(file=f, format="nt")
    labels_df = pd.read_csv(config.LABELS_FILE, sep="\t")
    return graph, labels_df


# =============================================================================
# PART A -- Relation-level global fidelity validation
# =============================================================================

def extract_embeddings_masked(model, data, keep_mask):
    ei = data.edge_index[:, keep_mask]
    et = data.edge_type[keep_mask]
    with torch.no_grad():
        emb = model.conv1(data.x, ei, et).relu()
    return emb.cpu().numpy()


def build_relation_influence_matrix(model, data, mappings):
    """influence_matrix[r, d] = mean |Δ embedding_d| when relation r is removed."""
    print("[step9] Building relation → embedding influence matrix ...")
    inv_relations = {v: k for k, v in mappings["relations_dict"].items()}

    full_emb = extract_embeddings_masked(
        model, data, torch.ones(data.edge_type.shape[0], dtype=torch.bool)
    )
    num_directed = int(data.edge_type.max().item()) + 1
    num_base = num_directed // 2

    matrix = np.zeros((num_base, full_emb.shape[1]))
    for base_rel in range(num_base):
        mask = ~((data.edge_type == 2 * base_rel) | (data.edge_type == 2 * base_rel + 1))
        ablated_emb = extract_embeddings_masked(model, data, mask)
        matrix[base_rel] = np.abs(full_emb - ablated_emb).mean(axis=0)

    base_relation_names = [short_name(inv_relations[r]) for r in range(num_base)]
    return matrix, base_relation_names, num_base


def describe_embedding_dimensions(influence_matrix, base_relation_names, top_k=3):
    descriptions = []
    for d in range(influence_matrix.shape[1]):
        col = influence_matrix[:, d]
        top_idx = np.argsort(col)[::-1][:top_k]
        top_rels = [base_relation_names[i] for i in top_idx if col[i] > 0]
        descriptions.append(
            f"a hidden signal mainly shaped by {', '.join(top_rels)} edges"
            if top_rels else "a hidden signal with low overall activation"
        )
    return descriptions


def rank_relations_by_shap(influence_matrix, shap_rf, shap_lr):
    score_rf = influence_matrix @ shap_rf
    score_lr = influence_matrix @ shap_lr
    score_combined = (score_rf + score_lr) / 2.0
    return score_rf, score_lr, score_combined


def class_prob(model, data, edge_index, edge_type, node_index, class_id):
    with torch.no_grad():
        log_probs = model(data.x, edge_index, edge_type)
        return float(F.softmax(log_probs[node_index], dim=0)[class_id])


def evaluate_fidelity_at_k(model, data, sorted_base_relations, k):
    important_all = set(
        t for r in sorted_base_relations[:k] for t in (2 * r, 2 * r + 1)
    )
    important_mask = torch.tensor(
        [int(et) in important_all for et in data.edge_type.tolist()],
        dtype=torch.bool,
    )
    sparsity = float(important_mask.sum()) / important_mask.shape[0]

    ei_keep = data.edge_index[:, ~important_mask]
    et_keep = data.edge_type[~important_mask]
    ei_only = data.edge_index[:, important_mask]
    et_only = data.edge_type[important_mask]

    with torch.no_grad():
        full_preds = model(data.x, data.edge_index, data.edge_type).argmax(dim=1)

    fp_list, fm_list = [], []
    for node_idx in data.test_idx.tolist():
        pred_class = int(full_preds[node_idx])
        full_p  = class_prob(model, data, data.edge_index, data.edge_type, node_idx, pred_class)
        minus_p = class_prob(model, data, ei_keep, et_keep, node_idx, pred_class)
        only_p  = class_prob(model, data, ei_only, et_only, node_idx, pred_class)
        fp_list.append(full_p - minus_p)
        fm_list.append(full_p - only_p)

    return {
        "k": k,
        "fidelity_plus":  float(np.mean(fp_list)),
        "fidelity_minus": float(np.mean(fm_list)),
        "sparsity":       sparsity,
    }


def plot_relation_importance(rel_df, save_path, top_n=20):
    subset = rel_df.head(top_n).copy()
    labels = [
        f"{row['relation_name']}\n({plain_english(row['relation_name'])})"
        for _, row in subset.iterrows()
    ]
    plt.figure(figsize=(11, 7))
    plt.barh(labels[::-1], subset["score_combined"].values[::-1], color="#4C72B0")
    plt.xlabel("Weighted SHAP importance score")
    plt.title(f"Top {top_n} most important RDF relation types\n"
              f"(SHAP-weighted, averaged over RF and LR surrogates)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[step9] Saved {save_path}")


def plot_fidelity_curve(fidelity_df, save_path):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(fidelity_df["k"], fidelity_df["fidelity_plus"],
             marker="o", label="Fidelity+ (remove top-K relations)", color="#d62728", linewidth=2)
    ax1.plot(fidelity_df["k"], fidelity_df["fidelity_minus"],
             marker="s", label="Fidelity− (keep only top-K relations)", color="#2ca02c", linewidth=2)
    ax1.set_xlabel("Number of top relation types selected by SHAP (K)")
    ax1.set_ylabel("Mean drop in predicted-class probability")
    ax1.legend(loc="upper left")
    ax1.set_title("SHAP-guided Fidelity Curve\nHow much do top-K relation types drive predictions?")

    ax2 = ax1.twinx()
    ax2.bar(fidelity_df["k"], fidelity_df["sparsity"],
            alpha=0.20, color="#1f77b4", label="Fraction of edges in top-K types")
    ax2.set_ylabel("Fraction of graph edges covered by top-K relation types")
    ax2.legend(loc="lower right")

    best_k  = int(fidelity_df.loc[fidelity_df["fidelity_plus"].idxmax(), "k"])
    best_fp = float(fidelity_df.loc[fidelity_df["fidelity_plus"].idxmax(), "fidelity_plus"])
    ax1.annotate(f"Peak fidelity+\nK={best_k}", xy=(best_k, best_fp),
                 xytext=(best_k + 0.4, best_fp + 0.01),
                 arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[step9] Saved {save_path}")


def run_part_a(model, data, mappings, shap_importance_df):
    """Relation-level global importance + fidelity validation."""
    influence_matrix, base_relation_names, num_base = build_relation_influence_matrix(
        model, data, mappings
    )
    dim_descriptions = describe_embedding_dimensions(influence_matrix, base_relation_names)

    shap_rf = shap_importance_df.sort_values("feature")["mean_abs_shap_rf"].values
    shap_lr = shap_importance_df.sort_values("feature")["mean_abs_shap_lr"].values
    score_rf, score_lr, score_combined = rank_relations_by_shap(influence_matrix, shap_rf, shap_lr)

    rel_df = pd.DataFrame({
        "relation_name":  base_relation_names,
        "score_rf":       score_rf,
        "score_lr":       score_lr,
        "score_combined": score_combined,
    }).sort_values("score_combined", ascending=False).reset_index(drop=True)
    rel_df.to_csv(f"{config.RESULTS_TABLES_DIR}/shap_relation_importance.csv", index=False)

    plot_relation_importance(rel_df, f"{config.RESULTS_FIGURES_DIR}/shap_relation_importance.png")

    sorted_base_rels = [base_relation_names.index(n) for n in rel_df["relation_name"].tolist()]
    k_values = list(range(1, min(11, num_base + 1)))
    print(f"[step9] Running fidelity evaluation for K = {k_values} ...")
    records = []
    for k in k_values:
        print(f"  K={k} ...", end=" ", flush=True)
        row = evaluate_fidelity_at_k(model, data, sorted_base_rels, k)
        records.append(row)
        print(f"fidelity+={row['fidelity_plus']:.4f}  fidelity-={row['fidelity_minus']:.4f}  "
              f"sparsity={row['sparsity']:.4f}")

    fidelity_df = pd.DataFrame(records)
    fidelity_df.to_csv(f"{config.RESULTS_TABLES_DIR}/shap_fidelity_evaluation.csv", index=False)
    plot_fidelity_curve(fidelity_df, f"{config.RESULTS_FIGURES_DIR}/shap_fidelity_curve.png")

    return rel_df, fidelity_df, dim_descriptions, num_base


# =============================================================================
# PART B -- Keyword-based per-class differentiation
# =============================================================================

def clean_target_name(uri: str):
    name = uri.rsplit("/", 1)[-1]
    is_category = name.startswith("Category:")
    name = name.replace("Category:", "").replace("_", " ")
    return name, is_category


def tokenize(name: str):
    tokens = re.findall(r"[A-Za-z]+", name.lower())
    return [t for t in tokens if t not in STOPWORDS and not YEAR_RE.match(t) and len(t) > 2]


def collect_keyword_counts(graph, labels_df):
    pred = rdflib.URIRef(WIKILINK_PRED)
    class_keyword_counts  = defaultdict(Counter)
    class_entity_counts   = Counter()
    class_example_targets = defaultdict(lambda: defaultdict(list))

    label_lookup = dict(zip(labels_df["entity"], labels_df["label"]))
    total = len(label_lookup)

    for i, (ent_str, cls) in enumerate(label_lookup.items()):
        if i % 200 == 0:
            print(f"  ... {i}/{total} entities processed")
        class_entity_counts[cls] += 1
        ent = rdflib.URIRef(ent_str)
        entity_keywords_seen = set()
        for _, _, obj in graph.triples((ent, pred, None)):
            target_name, _ = clean_target_name(str(obj))
            for kw in tokenize(target_name):
                if kw not in entity_keywords_seen:
                    entity_keywords_seen.add(kw)
                    class_keyword_counts[cls][kw] += 1
                if len(class_example_targets[cls][kw]) < 3 and target_name not in class_example_targets[cls][kw]:
                    class_example_targets[cls][kw].append(target_name)

    return class_keyword_counts, class_entity_counts, class_example_targets


def compute_distinctive_keywords(class_keyword_counts, class_entity_counts,
                                 class_names, min_count=5, top_n=12):
    results = {}
    alpha = 1.0
    for cls in class_names:
        this_counts = class_keyword_counts[cls]
        this_n      = class_entity_counts[cls]
        other_counts = Counter()
        other_n      = 0
        for c2 in class_names:
            if c2 == cls:
                continue
            other_counts.update(class_keyword_counts[c2])
            other_n += class_entity_counts[c2]

        rows = []
        for kw, count_this in this_counts.items():
            if count_this < min_count:
                continue
            rate_this  = count_this / max(this_n, 1)
            count_other = other_counts.get(kw, 0)
            rate_other = (count_other + alpha) / max(other_n + alpha, 1)
            lift = rate_this / rate_other
            rows.append({
                "keyword": kw, "count_this": count_this, "pct_this": 100 * rate_this,
                "count_other": count_other, "pct_other": 100 * rate_other, "lift": lift,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("lift", ascending=False).head(top_n).reset_index(drop=True)
        results[cls] = df
    return results


def measure_keyword_removal_impact(model, data, mappings, distinctive_by_class,
                                   class_names, top_n_keywords=5):
    nodes_dict  = mappings["nodes_dict"]
    labels_dict = mappings["labels_dict"]

    node_id_keywords = {}
    for node_str, node_id in nodes_dict.items():
        name, _ = clean_target_name(node_str)
        node_id_keywords[node_id] = set(tokenize(name))

    with torch.no_grad():
        full_log_probs = model(data.x, data.edge_index, data.edge_type)

    results = {}
    for cls in class_names:
        cls_id = labels_dict[cls]
        df = distinctive_by_class[cls]
        if df.empty:
            results[cls] = None
            continue
        top_keywords = set(df.head(top_n_keywords)["keyword"].tolist())
        target_node_ids = {nid for nid, kws in node_id_keywords.items() if kws & top_keywords}
        if not target_node_ids:
            results[cls] = None
            continue

        dst_nodes = data.edge_index[1]
        strip_mask = torch.tensor(
            [int(d) in target_node_ids for d in dst_nodes.tolist()], dtype=torch.bool
        )
        if strip_mask.sum().item() == 0:
            results[cls] = None
            continue

        keep_mask = ~strip_mask
        ei_keep = data.edge_index[:, keep_mask]
        et_keep = data.edge_type[keep_mask]
        with torch.no_grad():
            stripped_log_probs = model(data.x, ei_keep, et_keep)

        test_idx_this_cls = [
            int(idx) for idx in data.test_idx.tolist()
            if int(data.test_y[(data.test_idx == idx).nonzero()[0, 0]]) == cls_id
        ]
        if not test_idx_this_cls:
            results[cls] = None
            continue

        drops = []
        for node_idx in test_idx_this_cls:
            full_p  = float(F.softmax(full_log_probs[node_idx], dim=0)[cls_id])
            strip_p = float(F.softmax(stripped_log_probs[node_idx], dim=0)[cls_id])
            drops.append(full_p - strip_p)

        results[cls] = {
            "mean_prob_drop":  float(np.mean(drops)),
            "n_test_nodes":    len(test_idx_this_cls),
            "n_edges_removed": int(strip_mask.sum()),
            "top_keywords":    sorted(top_keywords),
        }
    return results


CLASS_COLORS = {
    "Athlete": "#2196F3", "MusicalArtist": "#E91E63",
    "Politician": "#FF9800", "Scientist": "#4CAF50",
}


def plot_distinctive_keywords(distinctive_by_class, save_prefix):
    for cls, df in distinctive_by_class.items():
        if df.empty:
            continue
        color = CLASS_COLORS.get(cls, "#607D8B")
        subset = df.head(10)
        labels = [f"{row['keyword']}  (×{row['lift']:.1f})" for _, row in subset.iterrows()]
        plt.figure(figsize=(9, 5))
        plt.barh(labels[::-1], subset["pct_this"].values[::-1], color=color, alpha=0.85)
        plt.xlabel(f"% of {cls} entities linked to a page containing this word")
        plt.title(f"What '{cls}' Wikipedia pages link to\n"
                 f"(top keywords from linked page/category names, with lift vs other classes)")
        plt.tight_layout()
        path = f"{save_prefix}_{cls}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[step9] Saved {path}")


def run_part_b(model, data, mappings, class_names):
    """Keyword-based per-class differentiation + validation."""
    graph, labels_df = load_graph_and_labels()

    print("[step9] Collecting wikiPageWikiLink target keywords per entity ...")
    class_keyword_counts, class_entity_counts, class_example_targets = \
        collect_keyword_counts(graph, labels_df)

    print("[step9] Computing distinctive keywords (lift score) ...")
    distinctive_by_class = compute_distinctive_keywords(
        class_keyword_counts, class_entity_counts, class_names, min_count=5, top_n=12
    )
    for cls in class_names:
        print(f"\n[step9] Top distinctive keywords for {cls}:")
        if distinctive_by_class[cls].empty:
            print("  (none found above min_count threshold)")
        else:
            print(distinctive_by_class[cls].head(8).to_string(index=False))

    all_rows = []
    for cls, df in distinctive_by_class.items():
        if df.empty:
            continue
        df2 = df.copy()
        df2.insert(0, "class", cls)
        all_rows.append(df2)
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(
            f"{config.RESULTS_TABLES_DIR}/class_distinctive_keywords.csv", index=False
        )

    plot_distinctive_keywords(
        distinctive_by_class, f"{config.RESULTS_FIGURES_DIR}/class_distinctive_keywords"
    )

    print("[step9] Validating: does removing distinctive keyword-edges hurt accuracy?")
    validation_results = measure_keyword_removal_impact(
        model, data, mappings, distinctive_by_class, class_names, top_n_keywords=5
    )
    for cls, val in validation_results.items():
        if val:
            print(f"  {cls}: prob drop = {val['mean_prob_drop']:.4f} "
                  f"({val['n_edges_removed']} edges removed, keywords={val['top_keywords']})")
        else:
            print(f"  {cls}: validation skipped (no matching data)")

    return distinctive_by_class, validation_results, class_entity_counts, class_example_targets


# =============================================================================
# PART C -- Natural language report
# =============================================================================

def generate_report(
    mappings, rel_df, fidelity_df, shap_importance_df, surrogate_perf_df,
    dim_descriptions, num_base, class_names,
    distinctive_by_class, validation_results, class_entity_counts, class_example_targets,
):
    inv_labels  = {v: k for k, v in mappings["labels_dict"].items()}
    num_classes = len(class_names)

    rf_fid = float(surrogate_perf_df.loc[
        surrogate_perf_df["model"] == "RandomForest", "surrogate_fidelity"].values[0])
    lr_fid = float(surrogate_perf_df.loc[
        surrogate_perf_df["model"] == "LogisticRegression", "surrogate_fidelity"].values[0])
    rf_acc = float(surrogate_perf_df.loc[
        surrogate_perf_df["model"] == "RandomForest", "true_accuracy"].values[0])
    lr_acc = float(surrogate_perf_df.loc[
        surrogate_perf_df["model"] == "LogisticRegression", "true_accuracy"].values[0])
    agree_word = (
        "strongly agree" if abs(rf_fid - lr_fid) < 0.05 else
        "broadly agree"  if abs(rf_fid - lr_fid) < 0.12 else
        "diverge noticeably"
    )

    top5_rels = rel_df.head(5)
    opt_k     = int(fidelity_df.loc[fidelity_df["fidelity_plus"].idxmax(), "k"])
    opt_row   = fidelity_df[fidelity_df["k"] == opt_k].iloc[0]

    lines = []
    A = lines.append

    A("# Global Explanation Report — R-GCN Node Classifier on DBpedia")
    A("")
    A("*Generated by `step9_global_SHAP_evaluate.py`*")
    A("")
    A("---")
    A("")

    # --- 1. Overview ---------------------------------------------------------
    A("## 1. What This Report Explains")
    A("")
    A(_wrap(
        f"This report answers two questions in plain language about the trained "
        f"R-GCN classifier on {num_classes} DBpedia entity classes "
        f"({', '.join(class_names)}): "
        f"**(A) which types of graph connections does the model rely on overall**, "
        f"and **(B) what specifically makes the model pick one class over another**. "
        f"Part A uses SHAP on two surrogate models (Random Forest, Logistic "
        f"Regression) trained to mimic the GNN. Part B goes further and looks "
        f"at the actual Wikipedia pages each entity links to, because a "
        f"predicate name alone (e.g. 'wikiPageWikiLink') cannot distinguish "
        f"classes — every entity uses the same predicates. What differs is "
        f"*where those links point*."
    ))
    A("")

    # --- 2. Surrogate trust ----------------------------------------------------
    A("## 2. Can We Trust the Explanations?")
    A("")
    A("| Surrogate | Agreement with GNN | Correct classification rate |")
    A("|-----------|-------------------|----------------------------|")
    A(f"| Random Forest       | **{rf_fid:.1%}** | {rf_acc:.1%} |")
    A(f"| Logistic Regression | **{lr_fid:.1%}** | {lr_acc:.1%} |")
    A("")
    if rf_fid >= 0.85 and lr_fid >= 0.85:
        A(_wrap(
            "Both surrogates agree with the GNN on more than 85% of entities — "
            "the SHAP explanations below accurately reflect the GNN's reasoning."
        ))
    elif max(rf_fid, lr_fid) >= 0.70:
        A(_wrap(
            f"The best surrogate agrees {max(rf_fid, lr_fid):.1%} of the time — "
            f"a reasonable approximation; treat exact scores with some caution."
        ))
    else:
        A(_wrap("Both surrogates struggle to replicate the GNN; treat findings as rough signals."))
    A("")
    A(_wrap(
        f"The two surrogates {agree_word} (RF: {rf_fid:.1%}, LR: {lr_fid:.1%})."
    ))
    A("")

    # --- 3. Global relation importance -----------------------------------------
    A("## 3. Part A — Which Link Types Does the Model Rely On Globally?")
    A("")
    A(_wrap(
        "We trace SHAP importance back through the model's hidden layer to "
        "rank the actual RDF predicates by how much they shape predictions "
        "across ALL classes."
    ))
    A("")
    A("| Rank | Link type | What it is | Combined importance |")
    A("|------|-----------|------------|---------------------|")
    for rank, (_, row) in enumerate(top5_rels.iterrows(), 1):
        A(f"| {rank} | `{row['relation_name']}` | {plain_english(row['relation_name'])} | "
          f"{row['score_combined']:.4f} |")
    A("")
    top_rel = top5_rels.iloc[0]["relation_name"]
    A(_wrap(
        f"**`{top_rel}`** ({plain_english(top_rel)}) dominates globally — "
        f"its sheer volume and target pattern give the strongest structural signal. "
        f"But this only tells us the model 'looks at hyperlinks' — it does NOT "
        f"tell us why one entity gets classified as an Athlete and another as a "
        f"Scientist.  That requires Part B below."
    ))
    A("")

    A("### Validating the ranking: removing top-K predicates")
    A("")
    A("| K removed | Confidence drop (Fidelity+) | Confidence drop, only top-K kept (Fidelity−) | % edges covered |")
    A("|-----------|------------------------------|-----------------------------------------------|------------------|")
    for _, row in fidelity_df.iterrows():
        A(f"| Top {int(row['k'])} | {row['fidelity_plus']:.4f} | {row['fidelity_minus']:.4f} | "
          f"{row['sparsity']*100:.1f}% |")
    A("")
    A(_wrap(
        f"The largest confidence drop occurs at K = {opt_k} "
        f"(Fidelity+ = {opt_row['fidelity_plus']:.4f}), covering "
        f"{opt_row['sparsity']*100:.1f}% of graph edges. "
        + (
            f"Fidelity− at this K ({float(opt_row['fidelity_minus']):.4f}) is low, "
            f"meaning these top predicates are nearly sufficient on their own."
            if float(opt_row["fidelity_minus"]) < 0.15 else
            f"Fidelity− at this K ({float(opt_row['fidelity_minus']):.4f}) remains "
            f"high, meaning the model also relies on many other predicates — "
            f"no small set of predicate TYPES alone is fully sufficient."
        )
    ))
    A("")

    # --- 4. Per-class differentiation (Part B) ---------------------------------
    A("## 4. Part B — Why THIS Class and Not Another?")
    A("")
    A(_wrap(
        "Since every entity uses the same predicates, we instead look at what "
        "those `wikiPageWikiLink` edges actually point to. For each entity, we "
        "extract keywords from the names of linked pages/categories (e.g. "
        "'olympic', 'engineer', 'musician', 'constituency'), then measure how "
        "much more often each keyword appears for one class versus the other "
        "three combined. We then validate by removing edges to matching pages "
        "and checking the GNN's actual confidence drop."
    ))
    A("")

    for cls in class_names:
        A(f"### 4.{class_names.index(cls) + 1}  {cls}")
        A("")
        dom = CLASS_DOMAIN_CONTEXT.get(cls, "")
        if dom:
            A(_wrap(f"**Background:** {dom}"))
            A("")

        n_entities = class_entity_counts.get(cls, 0)
        df = distinctive_by_class.get(cls)

        if df is None or df.empty:
            A(f"No sufficiently frequent distinctive keywords were found for "
              f"`{cls}` (out of {n_entities} entities).")
            A("")
            continue

        A(f"Out of {n_entities} `{cls}` entities, their pages link disproportionately "
          f"often to pages/categories containing these words:")
        A("")
        A("| Keyword | % of this class | % of other classes | How much more likely | Example linked pages |")
        A("|---------|------------------|----------------------|------------------------|------------------------|")
        for _, row in df.head(6).iterrows():
            examples = class_example_targets.get(cls, {}).get(row["keyword"], [])
            examples_str = ", ".join(examples[:2])
            A(f"| **{row['keyword']}** | {row['pct_this']:.1f}% | {row['pct_other']:.1f}% | "
              f"**{row['lift']:.1f}×** | {examples_str} |")
        A("")

        top_kw, top_lift, top_pct = df.iloc[0]["keyword"], df.iloc[0]["lift"], df.iloc[0]["pct_this"]
        A(_wrap(
            f"**{top_pct:.0f}% of `{cls}` entities** link to a page containing "
            f"**'{top_kw}'** — **{top_lift:.1f}× more often** than the other "
            f"three classes. This is the concrete, checkable signal that makes "
            f"the model say '{cls}', not just 'has wikiPageWikiLink edges' "
            f"(every entity has those)."
        ))
        A("")

        val = validation_results.get(cls)
        if val:
            kw_list = ", ".join(f"'{k}'" for k in val["top_keywords"])
            A(_wrap(
                f"**Model validation:** removing edges pointing to pages "
                f"containing the top-5 keywords ({kw_list}) — "
                f"{int(val['n_edges_removed'])} edges across "
                f"{val['n_test_nodes']} test entities — drops the model's "
                f"confidence in the correct `{cls}` prediction by "
                f"**{val['mean_prob_drop']:.4f}** probability points. "
                + (
                    "This confirms the GNN genuinely relies on this specific "
                    "keyword pattern."
                    if val["mean_prob_drop"] > 0.02 else
                    "This modest drop suggests other redundant keywords "
                    "further down the list also contribute."
                )
            ))
        else:
            A(f"**Model validation:** insufficient matching data for `{cls}`.")
        A("")

    # --- 5. Summary --------------------------------------------------------
    A("## 5. Summary: What Has the Model Learned?")
    A("")
    A("**The model reads link structure, not text.**  "
      f"It reaches {max(rf_fid, lr_fid):.0%} surrogate agreement purely from "
      f"hyperlink patterns — it never sees entity names or descriptions as text.")
    A("")
    A(f"**Globally, `{top_rel}` dominates** — but the *specific targets* of "
      f"those links are what actually separate the four classes:")
    for cls in class_names:
        df = distinctive_by_class.get(cls)
        if df is not None and not df.empty:
            top_kw = df.iloc[0]["keyword"]
            A(f"- **{cls}** → pages containing **'{top_kw}'**")
        else:
            A(f"- **{cls}** → no single dominant keyword found")
    A("")
    A(f"**A small number of predicate types carry most global signal** — "
      f"the top-{opt_k} predicates cover {opt_row['sparsity']*100:.0f}% of edges "
      f"and account for the largest confidence drop when removed.")
    A("")
    struct = "non-linear" if rf_fid > lr_fid + 0.05 else "approximately linear"
    A(f"**The model's internal representation is {struct}** "
      f"(RF {rf_fid:.1%} vs LR {lr_fid:.1%} surrogate agreement).")
    A("")
    A("---")
    A("")
    A("*Figures: `results/figures/shap_relation_importance.png`, "
      "`shap_fidelity_curve.png`, `class_distinctive_keywords_<Class>.png`.  "
      "Raw tables in `results/tables/`.*")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    print("[step9] Loading trained R-GCN model ...")
    model, data, mappings = load_model_and_data()

    inv_labels  = {v: k for k, v in mappings["labels_dict"].items()}
    class_names = [inv_labels[i] for i in sorted(inv_labels)]

    print("[step9] Loading SHAP importance + surrogate performance from step 8 ...")
    shap_importance_df = load_shap_importance()
    surrogate_perf_df  = pd.read_csv(f"{config.RESULTS_TABLES_DIR}/surrogate_performance.csv")

    # ---- PART A: global relation importance + fidelity ----------------------
    print("\n[step9] === PART A: Global relation-level fidelity evaluation ===")
    rel_df, fidelity_df, dim_descriptions, num_base = run_part_a(
        model, data, mappings, shap_importance_df
    )

    # ---- PART B: per-class keyword differentiation ---------------------------
    print("\n[step9] === PART B: Per-class keyword differentiation ===")
    distinctive_by_class, validation_results, class_entity_counts, class_example_targets = \
        run_part_b(model, data, mappings, class_names)

    # ---- PART C: report -------------------------------------------------------
    print("\n[step9] === PART C: Generating natural language report ===")
    report_md = generate_report(
        mappings=mappings,
        rel_df=rel_df,
        fidelity_df=fidelity_df,
        shap_importance_df=shap_importance_df,
        surrogate_perf_df=surrogate_perf_df,
        dim_descriptions=dim_descriptions,
        num_base=num_base,
        class_names=class_names,
        distinctive_by_class=distinctive_by_class,
        validation_results=validation_results,
        class_entity_counts=class_entity_counts,
        class_example_targets=class_example_targets,
    )
    report_path = f"{config.RESULTS_TABLES_DIR}/global_explanation_report.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"[step9] Saved report → {report_path}")
    print("[step9] Done.")


if __name__ == "__main__":
    main()