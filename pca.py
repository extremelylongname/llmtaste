'''
taste-label PCA, per-layer activation PCA with shuffled null, cross-correlation
'''

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from common import rng, labels, load_arrays, zscore_eps

activations, taste_vectors, stds, pca_scores, raw_names, names, token_counts = load_arrays()

plt.figure(figsize=(15, 15))
sns.heatmap(np.corrcoef(taste_vectors, rowvar=False), annot=True, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=[l.replace(" Mean", "") for l in labels], yticklabels=[l.replace(" Mean", "") for l in labels])
plt.savefig("plots/corrcoef.png")

pca_scaled = PCA(n_components=None)
scaled_X_pca = pca_scaled.fit_transform(StandardScaler().fit_transform(taste_vectors))
print(pca_scaled.explained_variance_ratio_)
print(pca_scaled.components_)
np.save("top_taste_pca.npy", scaled_X_pca)

plt.figure(figsize=(30, 15))
sns.heatmap(pca_scaled.components_, annot=True, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=labels, yticklabels=[f"PC {i+1}" for i in range(pca_scaled.components_.shape[0])])
plt.savefig("plots/taste_pca_heatmap.png")

fig, ax = plt.subplots(1, 2)
ax[0].scatter(PCA(n_components=None).fit_transform(taste_vectors)[:, 0], PCA(n_components=None).fit_transform(taste_vectors)[:, 1])
ax[0].set_title("unscaled")
ax[1].scatter(scaled_X_pca[:, 0], scaled_X_pca[:, 1])
ax[1].set_title("scaled")
plt.savefig("plots/taste_pca.png")

activation_pca = PCA(n_components=None)
activation_eigs, activation_pcas, shuffled_eigs, counts = [], [], [], []
for layer in range(activations.shape[1]):
	selected_layer = activations[:, layer, :]
	activation_pcas.append(activation_pca.fit_transform(zscore_eps(selected_layer)))
	activation_eigs.append(activation_pca.explained_variance_ratio_)
	shuffled_eigs_layer = []
	for i in range(10):
		activation_pca.fit_transform(zscore_eps(rng.permuted(selected_layer, axis=0)))
		shuffled_eigs_layer.append(activation_pca.explained_variance_ratio_)
	shuffled_eigs.append(np.mean(shuffled_eigs_layer, axis=0))
	counts.append(np.sum(activation_eigs[-1] > shuffled_eigs[-1]))

plt.figure()
plt.yscale("log")
plt.ylabel(r"$\lambda$")
plt.xlabel("PCA rank")
plt.plot(np.array(activation_eigs).T, c='blue', label="original")
plt.plot(np.array(shuffled_eigs).T, c='orange', label="shuffled")
handles, ll = plt.gca().get_legend_handles_labels()
plt.legend(dict(zip(ll, handles)).values(), dict(zip(ll, handles)).keys())
plt.savefig("plots/activation_eigs.png")

plt.figure()
plt.plot(np.arange(activations.shape[1]), counts)
plt.xlabel("layer")
plt.ylabel("count")
plt.savefig("plots/internal_pca_counts.png")

activation_eigs = np.array(activation_eigs)
activation_pcas = np.array(activation_pcas)
n_taste = scaled_X_pca.shape[1]
layers = [3, 7, 15, 25, 33]
fig, ax = plt.subplots(1, len(layers), figsize=(7*len(layers) + 5, len(layers)))
i = 0
most_aligned, human_alignments, best_rs, high_corrs = [], [], [], []
for layer in range(activation_eigs.shape[0]):
	cross_corr = np.corrcoef(activation_pcas[layer][:, :50].T, np.hstack([scaled_X_pca, zscore(taste_vectors, axis=0)]).T)
	most_aligned.append(np.argmax(np.abs(cross_corr[:50, 50])))
	human_alignments.append([[np.argmax(np.abs(cross_corr[:10, j])), np.max(np.abs(cross_corr[:10, j]))] for j in range(50 + n_taste, 50 + 2*n_taste)])
	best_rs.append(np.max(np.abs(cross_corr[:50, 50])))
	high_corrs.append(np.sum(np.abs(cross_corr[:50, 50:50 + n_taste]) >= 0.3))
	if layer in layers:
		sns.heatmap(cross_corr[:50, 50:50 + n_taste], cmap="coolwarm", vmin=-1, vmax=1, ax=ax[i])
		i += 1
plt.savefig("plots/crosscorr_heatmap.png")

fig, ax = plt.subplots(2, 1)
ax[0].plot(np.arange(activations.shape[1]), best_rs)
ax[0].set_ylabel(r"$|r|$")
ax[1].plot(np.arange(activations.shape[1]), high_corrs)
ax[1].set_ylabel("count")
ax[1].set_xlabel("layer")
plt.savefig("plots/crosscorr_counts.png")

human_alignments = np.array(human_alignments)
human_loadings = pca_scaled.components_[0]
human_rank = {j: r for r, j in enumerate(np.argsort(np.abs(human_loadings))[::-1])}
for layer in layers:
	strengths = human_alignments[layer, :, 1]
	print(f"\n=== layer {layer}  (weighting agreement ρ = {spearmanr(strengths, np.abs(human_loadings)).statistic:.2f}) ===")
	print(f"{'axis':<12}{'model |r|':>10}{'at PC':>7}{'model rank':>12}{'human rank':>12}{'PC1 loading':>13}{'raw std':>9}")
	for m_rank, j in enumerate(np.argsort(strengths)[::-1]):
		print(f"{labels[j].replace(' Mean', ''):<12}{strengths[j]:>10.2f}{human_alignments[layer, j, -2].astype(int):>7}{m_rank + 1:>12}{human_rank[j] + 1:>12}{human_loadings[j]:>+13.2f}{taste_vectors.std(axis=0)[j]:>9.2f}")
