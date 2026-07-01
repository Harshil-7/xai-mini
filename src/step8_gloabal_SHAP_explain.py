"""
Step 8 -- Global explanation via SHAP on a Surrogate Model.

Strategy
--------
The R-GCN is a black-box graph model whose inputs are adjacency structure
+ one-hot degree features. SHAP works best on tabular models, so we:

  1. Extract the *penultimate* node embeddings from the trained R-GCN
     (the 16-dimensional hidden vector after conv1 + ReLU) for every
     labeled node (train + test combined).
  2. Train TWO surrogate classifiers on those embeddings:
        - Random Forest  (more expressive, captures non-linear interactions)
        - Logistic Regression (linear, faster, SHAP exact via LinearExplainer)
  3. Run SHAP on both surrogates to get global feature importances.
  4. Save figures + CSVs.

Why embeddings instead of raw features?
  The raw input is a one-hot degree vector (low-information).
  The hidden embedding already encodes neighbourhood structure learned by
  the R-GCN, so the surrogate actually has something interesting to
  explain.  The surrogate fidelity score tells us how well the surrogate
  approximates the GNN on those embeddings.

Output
------
  results/figures/shap_rf_summary.png
  results/figures/shap_lr_summary.png
  results/figures/shap_rf_bar.png
  results/figures/shap_lr_bar.png
  results/tables/surrogate_performance.csv
  results/tables/shap_global_importance.csv
"""

import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import config
from step3_train_rgcn import FastRGCN

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_and_data():
    checkpoint = torch.load(config.MODEL_FILE, weights_only=False)
    with open(config.MODEL_FILE + ".mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
    model = FastRGCN(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["data"], mappings


def extract_embeddings(model, data):
    with torch.no_grad():
        embeddings = model.conv1(data.x, data.edge_index, data.edge_type).relu()
    return embeddings.cpu().numpy()


def gather_labeled_nodes(data, mappings):
    all_idx = torch.cat([data.train_idx, data.test_idx]).cpu().numpy()
    all_y   = torch.cat([data.train_y,   data.test_y  ]).cpu().numpy()
    inv_labels  = {v: k for k, v in mappings["labels_dict"].items()}
    label_names = [inv_labels[i] for i in sorted(inv_labels)]
    return all_idx, all_y, label_names


def rgcn_predictions(model, data, node_indices):
    with torch.no_grad():
        log_probs = model(data.x, data.edge_index, data.edge_type)
        preds = log_probs.argmax(dim=1).cpu().numpy()
    return preds[node_indices]


# ---------------------------------------------------------------------------
# Surrogate training
# ---------------------------------------------------------------------------

def train_surrogates(X_train, y_train):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(
        n_estimators=200, random_state=config.RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    lr = LogisticRegression(
        max_iter=1000, random_state=config.RANDOM_SEED, solver="lbfgs"
    )
    lr.fit(X_scaled, y_train)

    return rf, lr, scaler


def surrogate_fidelity(surrogate, X, gnn_preds, scaler=None):
    X_in = scaler.transform(X) if scaler is not None else X
    return accuracy_score(gnn_preds, surrogate.predict(X_in))


def surrogate_accuracy(surrogate, X, true_y, scaler=None):
    X_in = scaler.transform(X) if scaler is not None else X
    return accuracy_score(true_y, surrogate.predict(X_in))


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def shap_for_rf(rf, X, feature_names):
    print("[step8] Computing SHAP values for Random Forest ...")
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def shap_for_lr(lr, X_scaled, feature_names):
    print("[step8] Computing SHAP values for Logistic Regression ...")
    explainer   = shap.LinearExplainer(lr, X_scaled)
    shap_values = explainer.shap_values(X_scaled)
    return explainer, shap_values


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _feature_names(n):
    return [f"emb_{i}" for i in range(n)]


def to_1d_shap_importance(sv):
    sv = np.asarray(sv)
    if sv.ndim == 2:
        return np.mean(np.abs(sv), axis=0)
    if sv.ndim == 3:
        return np.mean(np.abs(sv), axis=(0, 2))
    return np.abs(sv).ravel()


def plot_shap_summary(shap_values, X, feature_names, title, save_path):
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values)

    plt.figure(figsize=(9, 5))
    shap.summary_plot(mean_abs, X, feature_names=feature_names,
                      plot_type="dot", show=False, color_bar=True)
    plt.title(title, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[step8] Saved {save_path}")


def plot_shap_bar(shap_values, feature_names, title, save_path):
    """Global mean |SHAP| bar chart — robust to list-of-arrays and 2-D/3-D tensors."""

    # ---- 1. Collapse everything to a clean 1-D array of length n_features ----
    if isinstance(shap_values, list):
        # list of (n_samples, n_features) arrays — one per class
        stacked   = np.stack([np.abs(sv) for sv in shap_values], axis=0)  # (n_classes, n_samples, n_features)
        mean_abs  = stacked.mean(axis=(0, 1))                              # (n_features,)
    else:
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            mean_abs = np.abs(sv).mean(axis=(0, 2))   # (n_samples, n_features, n_classes) → (n_features,)
        elif sv.ndim == 2:
            mean_abs = np.abs(sv).mean(axis=0)        # (n_samples, n_features) → (n_features,)
        else:
            mean_abs = np.abs(sv).ravel()

    mean_abs = np.asarray(mean_abs).ravel()           # guarantee 1-D

    # ---- 2. Sanity-check length matches feature_names ----------------------
    n = len(feature_names)
    if len(mean_abs) != n:
        # trim or pad silently so we never crash
        mean_abs = mean_abs[:n] if len(mean_abs) > n else np.pad(mean_abs, (0, n - len(mean_abs)))

    # ---- 3. Sort and pick top-N -------------------------------------------
    order   = np.argsort(mean_abs)[::-1]
    top_n   = min(16, n)
    top_idx = order[:top_n][::-1]                     # reversed so highest is at top of barh

    plt.figure(figsize=(8, 5))
    plt.barh(
        [feature_names[int(i)] for i in top_idx],
        [float(mean_abs[int(i)]) for i in top_idx],
    )
    plt.xlabel("Mean |SHAP value|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[step8] Saved {save_path}")


def plot_shap_per_class(shap_values, X, feature_names, class_names, model_tag):
    """
    One bar chart per class  +  saves per-class mean |SHAP| to CSV
    so step 9 can use genuine per-class weights.
    """
    if not isinstance(shap_values, list):
        return

    for c, cls_name in enumerate(class_names):
        sv_c     = np.asarray(shap_values[c])   # (n_samples, n_features)
        mean_abs = np.abs(sv_c).mean(axis=0)    # (n_features,)
        order    = np.argsort(mean_abs)[::-1]
        top_n    = min(16, len(feature_names))
        top_idx  = order[:top_n][::-1]

        # --- figure ---
        plt.figure(figsize=(8, 4))
        plt.barh(
            [feature_names[i] for i in top_idx],
            [mean_abs[i]       for i in top_idx],
        )
        plt.xlabel("Mean |SHAP value|")
        plt.title(f"{model_tag} — SHAP importance for class '{cls_name}'")
        plt.tight_layout()
        fig_path = (f"{config.RESULTS_FIGURES_DIR}/"
                    f"shap_{model_tag.lower()}_class_{cls_name}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[step8] Saved {fig_path}")

        # --- CSV (new) — sorted by feature name so step9 can align easily ---
        csv_df = pd.DataFrame({
            "feature":        feature_names,
            "mean_abs_shap":  mean_abs,
        }).sort_values("feature").reset_index(drop=True)
        csv_path = (f"{config.RESULTS_TABLES_DIR}/"
                    f"shap_perclass_{model_tag.lower()}_{cls_name}.csv")
        csv_df.to_csv(csv_path, index=False)
        print(f"[step8] Saved {csv_path}")


# ---------------------------------------------------------------------------
# Global importance table
# ---------------------------------------------------------------------------

def build_importance_df(shap_values_rf, shap_values_lr, feature_names):
    rf_imp = (
        np.mean([to_1d_shap_importance(sv) for sv in shap_values_rf], axis=0)
        if isinstance(shap_values_rf, list)
        else to_1d_shap_importance(shap_values_rf)
    )
    lr_imp = (
        np.mean([to_1d_shap_importance(sv) for sv in shap_values_lr], axis=0)
        if isinstance(shap_values_lr, list)
        else to_1d_shap_importance(shap_values_lr)
    )
    rf_imp = np.asarray(rf_imp).ravel()
    lr_imp = np.asarray(lr_imp).ravel()

    df = pd.DataFrame({
        "feature":          feature_names,
        "mean_abs_shap_rf": rf_imp,
        "mean_abs_shap_lr": lr_imp,
    })
    df["rank_rf"] = df["mean_abs_shap_rf"].rank(ascending=False).astype(int)
    df["rank_lr"] = df["mean_abs_shap_lr"].rank(ascending=False).astype(int)
    return df.sort_values("mean_abs_shap_rf", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[step8] Loading trained R-GCN model ...")
    model, data, mappings = load_model_and_data()

    print("[step8] Extracting embeddings for all labeled nodes ...")
    all_embeddings        = extract_embeddings(model, data)
    all_idx, all_y, class_names = gather_labeled_nodes(data, mappings)

    X      = all_embeddings[all_idx]
    y_true = all_y
    y_gnn  = rgcn_predictions(model, data, all_idx)

    n_features   = X.shape[1]
    feature_names = _feature_names(n_features)
    print(f"[step8] {len(all_idx)} labeled nodes | classes: {class_names} | "
          f"embedding dim: {n_features}")

    print("[step8] Training surrogate models ...")
    rf, lr, scaler = train_surrogates(X, y_gnn)
    X_scaled = scaler.transform(X)

    perf_df = pd.DataFrame({
        "model": ["RandomForest", "LogisticRegression"],
        "surrogate_fidelity": [
            surrogate_fidelity(rf, X, y_gnn),
            surrogate_fidelity(lr, X, y_gnn, scaler),
        ],
        "true_accuracy": [
            surrogate_accuracy(rf, X, y_true),
            surrogate_accuracy(lr, X, y_true, scaler),
        ],
    })
    perf_df.to_csv(f"{config.RESULTS_TABLES_DIR}/surrogate_performance.csv", index=False)
    print("\n[step8] Surrogate performance:")
    print(perf_df.to_string(index=False))

    _, shap_values_rf = shap_for_rf(rf, X, feature_names)
    _, shap_values_lr = shap_for_lr(lr, X_scaled, feature_names)

    # global summary plots
    plot_shap_summary(shap_values_rf, X, feature_names,
                      "SHAP Summary — Random Forest Surrogate",
                      f"{config.RESULTS_FIGURES_DIR}/shap_rf_summary.png")
    plot_shap_summary(shap_values_lr, X_scaled, feature_names,
                      "SHAP Summary — Logistic Regression Surrogate",
                      f"{config.RESULTS_FIGURES_DIR}/shap_lr_summary.png")
    plot_shap_bar(shap_values_rf, feature_names,
                  "Mean |SHAP| per Embedding Dim — Random Forest",
                  f"{config.RESULTS_FIGURES_DIR}/shap_rf_bar.png")
    plot_shap_bar(shap_values_lr, feature_names,
                  "Mean |SHAP| per Embedding Dim — Logistic Regression",
                  f"{config.RESULTS_FIGURES_DIR}/shap_lr_bar.png")

    # per-class plots + CSVs (used by step 9)
    plot_shap_per_class(shap_values_rf, X,        feature_names, class_names, "RF")
    plot_shap_per_class(shap_values_lr, X_scaled, feature_names, class_names, "LR")

    imp_df = build_importance_df(shap_values_rf, shap_values_lr, feature_names)
    imp_df.to_csv(f"{config.RESULTS_TABLES_DIR}/shap_global_importance.csv", index=False)
    print("\n[step8] Top-5 embedding dims (RF SHAP):")
    print(imp_df.head(5).to_string(index=False))
    print("[step8] Done.")


if __name__ == "__main__":
    main()