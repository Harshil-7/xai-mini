"""
Central configuration for the whole pipeline.

Every script imports from here so that there is exactly ONE place to
change the dataset size, hyperparameters, or file paths.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_FIGURES_DIR = os.path.join(ROOT_DIR, "results", "figures")
RESULTS_TABLES_DIR = os.path.join(ROOT_DIR, "results", "tables")

GRAPH_FILE = os.path.join(DATA_RAW_DIR, "dbpedia_subset.nt.gz")
LABELS_FILE = os.path.join(DATA_PROCESSED_DIR, "labels.tsv")
TRAIN_FILE = os.path.join(DATA_PROCESSED_DIR, "train.tsv")
TEST_FILE = os.path.join(DATA_PROCESSED_DIR, "test.tsv")
MODEL_FILE = os.path.join(MODELS_DIR, "rgcn_model.pt")

for d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_FIGURES_DIR, RESULTS_TABLES_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# DBpedia extraction settings
# ---------------------------------------------------------------------------
SPARQL_ENDPOINT = "https://dbpedia.org/sparql"

TARGET_CLASSES = {
    "Scientist": "http://dbpedia.org/ontology/Scientist",
    "Politician": "http://dbpedia.org/ontology/Politician",
    "Athlete": "http://dbpedia.org/ontology/Athlete",
    "MusicalArtist": "http://dbpedia.org/ontology/MusicalArtist",
}

ENTITIES_PER_CLASS = 600
SPARQL_BATCH_SIZE = 25
LABEL_PREDICATE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

TRAIN_TEST_SPLIT = 0.8
RANDOM_SEED = 0

# ---------------------------------------------------------------------------
# Model / training hyperparameters
# ---------------------------------------------------------------------------
HIDDEN_CHANNELS = 16
NUM_BASES = 30
NUM_EPOCHS = 50
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0005
MAX_DEGREE_FEATURES = 50


# ---------------------------------------------------------------------------
# Explanation settings
# ---------------------------------------------------------------------------
GNN_EXPLAINER_EPOCHS = 200
TOP_K_EDGES = 5
NUM_NODES_TO_EXPLAIN = 5


# ---------------------------------------------------------------------------
# Global explanation settings (step 6)
# ---------------------------------------------------------------------------
# Number of trees in the Random Forest surrogate
RF_N_ESTIMATORS = 200

# Max iterations for Logistic Regression surrogate convergence
LR_MAX_ITER = 1000

# How many top embedding dimensions to show in per-class bar charts
SHAP_TOP_K_FEATURES = 16
