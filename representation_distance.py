'''
RSA between taste space and activation space, with mantel permutation test
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, zscore
from scipy.spatial.distance import squareform, pdist
from common import rng, load_arrays

activations, taste_vectors, stds, pca_scores, raw_names, names, token_counts = load_arrays()

taste_distances = pdist(zscore(taste_vectors, axis=0))
expanded = squareform(taste_distances)

p_values = []
correlations = []
runs = 1000

for layer in range(activations.shape[1]):
	sig_count = 0
	perm_corr_tot = 0
	layer_distances = pdist(zscore(activations[:, layer, :], axis=0))
	p_ta = spearmanr(layer_distances, taste_distances).statistic
	for i in range(runs):
		perm = rng.permutation(expanded.shape[0])
		perm_corr = spearmanr(layer_distances, squareform(expanded[np.ix_(perm, perm)], checks=False)).statistic
		if perm_corr > p_ta:
			sig_count += 1
		perm_corr_tot += perm_corr
	print(f"p-value: {(1 + sig_count)/(1 + runs)}")
	correlations.append([p_ta, perm_corr_tot/runs])
	p_values.append((1 + sig_count)/(1 + runs))
correlations = np.array(correlations)
np.savez("mantel_results.npz", correlations=correlations, p_values=p_values)

fig, ax1 = plt.subplots(figsize=(9, 6))
ax2 = ax1.twinx()
ax1.plot(np.arange(activations.shape[1]), correlations[:, 0], c="steelblue", label=r"observed $\rho$")
ax1.plot(np.arange(activations.shape[1]), correlations[:, 1], c="lightsteelblue", label=r"permuted $\rho$")
ax2.plot(np.arange(activations.shape[1]), p_values, c="firebrick", ls="--", label=r"$p$")
ax2.set_yscale("log")
ax1.set_xlabel("layer")
ax1.set_ylabel(r"$\rho$")
ax2.set_ylabel(r"$p$")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2)
plt.savefig("plots/representation_correlation.png")
