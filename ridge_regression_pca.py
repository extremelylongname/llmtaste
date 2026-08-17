"""
performing ridge regression between input taste dimensions and model activations
"""

from common import load_arrays, probe_panel

activations, taste_vectors, stds, pca_scores, raw_names, names, token_counts = load_arrays()

probe_panel(activations, pca_scores, [f"PC{i+1}" for i in range(pca_scores.shape[1])], "plots/ridgeregression_pca.png")
