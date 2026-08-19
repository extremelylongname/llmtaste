'''
kill all text length contamination
'''

import numpy as np
from common import load_arrays, haufe, rng, groups

activations, taste_vectors, stds, pca_scores, raw_names, names, token_counts = load_arrays()

steering_layer = 28
x = activations[:, steering_layer, :]
dirs = np.load("haufe_directions.npy")

top = np.argsort(x.var(axis=0))[-20:]
length_dims = top[(np.abs([np.corrcoef(x[:, d], token_counts)[0, 1] for d in top]) >= 0.3) & (np.abs([np.corrcoef(x[:, d], pca_scores[:, 0])[0, 1] for d in top]) < 0.2)]
np.save("length_dims.npy", length_dims)

onehots = np.zeros((x.shape[1], len(length_dims)))
onehots[length_dims, np.arange(len(length_dims))] = 1

N = np.linalg.qr(np.column_stack([onehots, haufe(x, token_counts)]))[0]
v_cleans = []

for pc, v in enumerate(dirs):
	v_clean = v - N @ (N.T @ v)
	v_clean /= np.linalg.norm(v_clean)
	v_cleans.append(v_clean)
	print(f"PC{pc+1} retained: {np.dot(v_clean, v)} corr: {np.corrcoef(x @ v_clean, token_counts)[0, 1]} own-taste: {np.corrcoef(x @ v_clean, pca_scores[:, pc])[0, 1]}")

B = np.array(v_cleans).T
Q = np.linalg.qr(B)[0]
for k in range(Q.shape[1]):
	if np.dot(Q[:, k], B[:, k]) < 0:
		Q[:, k] *= -1
np.save("steering_vectors.npy", Q.T)

def cleaned_directions(rows):
	xs = x[rows]
	raw = np.array([vec/np.linalg.norm(vec) for vec in [haufe(xs, pca_scores[rows, pc]) for pc in range(pca_scores.shape[1])]])
	n = np.linalg.qr(np.column_stack([onehots, haufe(xs, token_counts[rows])]))[0]
	cleaned = raw - (raw @ n) @ n.T
	b = np.array([vec/np.linalg.norm(vec) for vec in cleaned]).T
	q = np.linalg.qr(b)[0]
	for k in range(q.shape[1]):
		if np.dot(q[:, k], b[:, k]) < 0:
			q[:, k] *= -1
	return q.T

half_cos = []
for s in range(5):
	half = rng.permutation(np.unique(groups))[: len(np.unique(groups)) // 2]
	m = np.isin(groups, half)
	d1, d2 = cleaned_directions(np.where(m)[0]), cleaned_directions(np.where(~m)[0])
	half_cos.append([np.dot(d1[k], d2[k]) for k in range(d1.shape[0])])
print([f"PC{k+1}": round(float(c), 3) for k, c in enumerate(np.mean(half_cos, axis=0))])
