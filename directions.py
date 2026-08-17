'''
fit difference and haufe taste directions at the steering layer
'''

import numpy as np
from common import load_arrays, haufe

activations, taste_vectors, stds, pca_scores, raw_names, names, token_counts = load_arrays()

steering_layer = 28
quartile = pca_scores.shape[0] // 4
x = activations[:, steering_layer, :]
difference_vectors = []
haufe_vectors = []

for pc in range(pca_scores.shape[1]):
	order = np.argsort(pca_scores[:, pc])
	difference_vectors.append(np.mean(activations[order[-quartile:], steering_layer, :], axis=0) - np.mean(activations[order[:quartile], steering_layer, :], axis=0))
	haufe_vectors.append(haufe(x, pca_scores[:, pc]))

difference_vectors = np.array([vec/np.linalg.norm(vec) for vec in difference_vectors])
haufe_vectors = np.array([vec/np.linalg.norm(vec) for vec in haufe_vectors])

print("cross-method cosines (difference vs haufe):", [round(float(np.dot(difference_vectors[pc], haufe_vectors[pc])), 3) for pc in range(pca_scores.shape[1])])

np.save("haufe_directions.npy", haufe_vectors)
np.save("difference_directions.npy", difference_vectors)
