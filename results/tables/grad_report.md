# Grad Explanation Report

## 1. Dataset Overview

- Nodes: 5
- Avg explanation size: 2.80
- Relations discovered: 8

## 2. What This Model Learns

The model learns patterns from how real-world entities are connected in Wikipedia. It uses graph structure instead of raw text.

## 3. How Predictions Are Made

Predictions are based on relational patterns such as occupation, education, and domain-specific links.

## 4. Visual Summary

### Prediction Distribution
![](../figures/grad_prediction_distribution.png)

### Fidelity vs Sparsity
![](../figures/grad_fidelity_vs_sparsity.png)

## 5. Case Study

Entity: Augustin Maior

Predicted Label: Scientist

Key connections:

- Augustin Maior --[nationality]--> Romanian
- Augustin Maior --[origin]--> Physics
- Augustin Maior --[subject]--> Physics

The model aggregates multiple signals instead of relying on a single feature.

## 6. Key Insight

Graph structure provides distributed evidence across multiple relation types, improving robustness of predictions.
