'''
classify top-20 variance dims per layer
'''

import numpy as np
import matplotlib.pyplot as plt
from common import load_arrays

activations, taste_vectors, stds, pca_scores, raw_names, names, token_counts = load_arrays()

taste_counts, length_counts, entangled_counts, other_counts = [], [], [], []
for layer in range(activations.shape[1]):
	xl = activations[:, layer, :]
	top = np.argsort(xl.var(axis=0))[-20:]
	r_taste = np.array([np.max([np.abs(np.corrcoef(xl[:, d], pca_scores[:, pc])[0, 1]) for pc in range(pca_scores.shape[1])]) for d in top])
	r_len = np.array([np.abs(np.corrcoef(xl[:, d], token_counts)[0, 1]) for d in top])
	taste_counts.append(np.sum((r_taste >= 0.2) & (r_len < 0.3)))
	length_counts.append(np.sum((r_len >= 0.3) & (r_taste < 0.2)))
	entangled_counts.append(np.sum((r_taste >= 0.2) & (r_len >= 0.3)))
	other_counts.append(20 - taste_counts[-1] - length_counts[-1] - entangled_counts[-1])

plt.figure(figsize=(10, 6))
plt.stackplot(np.arange(activations.shape[1]), taste_counts, length_counts, entangled_counts, other_counts, labels=["taste only", "length only", "entangled", "neither"], colors=["firebrick", "steelblue", "mediumpurple", "lightgray"])
plt.xlabel("layer")
plt.ylabel("count")
plt.legend(loc="upper left")
plt.savefig("plots/dim_census.png")
