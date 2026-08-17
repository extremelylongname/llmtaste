'''
human vs model taste planes: procrustes, out-of-fold, unsupervised emergence
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from common import load_arrays, haufe, zscore_eps, rng, groups

activations, taste_vectors, stds, pca_scores, raw_names, names, token_counts = load_arrays()

steering_layer = 28
steering_vectors = np.load("steering_vectors.npy")
x = activations[:, steering_layer, :]

human_plane = pca_scores[:, :2]
landmarks = ["Milk chocolate", "Chocolate cake", "Paella", "Parmesan (cheese)", "Fruit candy", "Meringue", "Beef bourguignon", "Cola soda", "Expresso coffee", "Champagne", "Yogurt (plain, without sugar)", "Sweet wines (Sauternes, Monbazillac, Jurançon, etc.)"]

def panel_figure(h_plane, m_plane, colors, names_arr, save_path):
	fig, ax = plt.subplots(1, 3, figsize=(21, 7))
	ax[0].scatter(h_plane[:, 0], h_plane[:, 1], c=colors, cmap="coolwarm", s=12)
	ax[0].set_title("human taste plane")
	ax[1].scatter(m_plane[:, 0], m_plane[:, 1], c=colors, cmap="coolwarm", s=12)
	ax[1].set_title("model taste plane")
	for name in landmarks:
		idx = np.where(names_arr == name)[0]
		if len(idx):
			ax[0].annotate(name.split(" (")[0], h_plane[idx[0]], fontsize=7)
			ax[1].annotate(name.split(" (")[0], m_plane[idx[0]], fontsize=7)
	h_std, m_std, disparity = procrustes(h_plane, m_plane)
	ax[2].scatter(h_std[:, 0], h_std[:, 1], c=colors, cmap="coolwarm", s=10)
	ax[2].scatter(m_std[:, 0], m_std[:, 1], c=colors, cmap="coolwarm", s=10, marker="x")
	for i in range(h_std.shape[0]):
		ax[2].plot([h_std[i, 0], m_std[i, 0]], [h_std[i, 1], m_std[i, 1]], c="gray", lw=0.3, alpha=0.5)
	ax[2].set_title("procrustes overlay")
	plt.savefig(save_path)
	print(f"{save_path} disparity: {disparity:.3f}")
	order = np.argsort(np.linalg.norm(h_std - m_std, axis=1))
	print("largest disagreements:", names_arr[order[-10:]][::-1])
	return disparity

c = pca_scores[:, 0]
panel_figure(human_plane, np.column_stack([x @ steering_vectors[0], x @ steering_vectors[1]]), c, raw_names, "plots/manifold_panels.png")

def unsup_plane(layer):
	return PCA(n_components=2).fit_transform(zscore_eps(activations[:, layer, :]))

unsup_disparities = [procrustes(human_plane, unsup_plane(layer))[2] for layer in range(activations.shape[1])]

strip_layers = [0, 3, 7, 15, 25, 33]
fig, ax = plt.subplots(1, len(strip_layers) + 1, figsize=(7*(len(strip_layers) + 1) - 5, 6))
for i, layer in enumerate(strip_layers):
	m_std = procrustes(human_plane, unsup_plane(layer))[1]
	ax[i].scatter(m_std[:, 0], m_std[:, 1], c=c, cmap="coolwarm", s=8)
	ax[i].set_title(f"layer {layer}")
ax[-1].plot(np.arange(activations.shape[1]), unsup_disparities)
ax[-1].set_xlabel("layer")
ax[-1].set_ylabel("procrustes disparity")
plt.savefig("plots/manifold_emergence_unsupervised.png")

def cleaned_pair(rows, layer):
	xs = activations[rows, layer, :]
	raw = np.array([vec/np.linalg.norm(vec) for vec in [haufe(xs, pca_scores[rows, pc]) for pc in range(pca_scores.shape[1])]])
	top = np.argsort(xs.var(axis=0))[-20:]
	ldims = top[(np.abs([np.corrcoef(xs[:, d], token_counts[rows])[0, 1] for d in top]) >= 0.3) & (np.array([np.max([np.abs(np.corrcoef(xs[:, d], pca_scores[rows, pc])[0, 1]) for pc in range(pca_scores.shape[1])]) for d in top]) < 0.2)]
	oh = np.zeros((xs.shape[1], len(ldims)))
	oh[ldims, np.arange(len(ldims))] = 1
	n = np.linalg.qr(np.column_stack([oh, haufe(xs, token_counts[rows])]))[0]
	cleaned = raw - (raw @ n) @ n.T
	b = np.array([vec/np.linalg.norm(vec) for vec in cleaned]).T
	q = np.linalg.qr(b)[0]
	for k in range(q.shape[1]):
		if np.dot(q[:, k], b[:, k]) < 0:
			q[:, k] *= -1
	return q.T[:2]

m = np.isin(groups, rng.permutation(np.unique(groups))[: len(np.unique(groups)) // 2])
train, test = np.where(m)[0], np.where(~m)[0]
pair = cleaned_pair(train, steering_layer)
x_test = activations[test, steering_layer, :]
panel_figure(human_plane[test], np.column_stack([x_test @ pair[0], x_test @ pair[1]]), c[test], raw_names[test], "plots/manifold_panels_oof.png")
