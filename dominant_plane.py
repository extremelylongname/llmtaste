'''
correlation of the model's top-2 PCs with taste and token length
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from common import load_arrays, zscore_eps

activations, taste_vectors, stds, pca_scores, raw_names, names, token_counts = load_arrays()

r = np.zeros((activations.shape[1], 2, 3))
for layer in range(activations.shape[1]):
	scores = PCA(n_components=2).fit_transform(zscore_eps(activations[:, layer, :]))
	for a in range(2):
		r[layer, a] = [np.abs(np.corrcoef(scores[:, a], pca_scores[:, 0])[0, 1]), np.abs(np.corrcoef(scores[:, a], pca_scores[:, 1])[0, 1]), np.abs(np.corrcoef(scores[:, a], token_counts)[0, 1])]

layers = np.arange(activations.shape[1])
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
for a in range(2):
	for t in range(2):
		ax[a][t].plot(layers, r[:, a, t], c="firebrick", label=f"taste PC{t+1}")
		ax[a][t].plot(layers, r[:, a, 2], c="steelblue", label="token count")
		ax[a][t].set_title(f"act-PC{a+1}")
		ax[a][t].set_ylim(0, 0.85)
		ax[a][t].legend()
		ax[a][t].set_xlabel("layer")
		ax[a][t].set_ylabel(r"$|r|$")
plt.savefig("plots/dominant_plane_correlations.png")

plt.figure(figsize=(9, 6))
plt.plot(layers, r[:, 0, 0], c="firebrick", label="taste PC1")
plt.plot(layers, r[:, 0, 2], c="steelblue", label="token count")
plt.xlabel("layer")
plt.ylabel(r"$|r|$")
plt.legend()
plt.savefig("plots/dominant_plane_pc1.png")

plt.figure(figsize=(9, 5))
plt.plot(layers, r[:, 0, 0]**2 + r[:, 0, 2]**2, label="act-PC1")
plt.plot(layers, r[:, 1, 0]**2 + r[:, 1, 2]**2, label="act-PC2")
plt.xlabel("layer")
plt.ylabel(r"$r^2_{taste} + r^2_{len}$")
plt.ylim(0, 1)
plt.legend()
plt.savefig("plots/dominant_plane_confinement.png")
