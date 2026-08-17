'''
prompt-stability of steering-layer taste directions vs group-split floor
'''

import os
import numpy as np
from common import rng, foods, groups, load_arrays, extract_activations, haufe, principal_angles

activations, taste_vectors, stds, pca_scores, raw_names, names, token_counts = load_arrays()

STEERING_LAYER = 28
N_SPLITS = 5
TEMPLATES = {
	"bare": "{x}",
	"taste_of": "The taste of {x} is",
	"chat": "<start_of_turn>user\nFood: {x}<end_of_turn>\n",
}

def haufe_directions(xa, rows=None):
	if rows is None:
		rows = np.arange(xa.shape[0])
	dirs = [haufe(xa[rows], pca_scores[rows, pc]) for pc in range(pca_scores.shape[1])]
	return np.array([a / np.linalg.norm(a) for a in dirs])

base_acts = activations[:, STEERING_LAYER, :]
assert len(foods) == base_acts.shape[0] == pca_scores.shape[0] == len(groups), "row misalignment"

acts = {"base": base_acts}
for name, fmt in TEMPLATES.items():
	cache = f"prompt_stability_acts_{name}_l{STEERING_LAYER}.npy"
	if os.path.exists(cache):
		acts[name] = np.load(cache)
	else:
		acts[name], _ = extract_activations(fmt, layer=STEERING_LAYER)
		np.save(cache, acts[name])

dirs = {name: haufe_directions(a) for name, a in acts.items()}

floor_angles, floor_cos = [], []
for _ in range(N_SPLITS):
	m = np.isin(groups, rng.permutation(np.unique(groups))[: len(np.unique(groups)) // 2])
	d1, d2 = haufe_directions(base_acts, np.where(m)[0]), haufe_directions(base_acts, np.where(~m)[0])
	floor_angles.append(principal_angles(d1, d2))
	floor_cos.append([np.dot(d1[p], d2[p]) for p in range(pca_scores.shape[1])])

pc_names = [f"PC{p + 1}" for p in range(pca_scores.shape[1])]
print(f"split-half floor ({N_SPLITS} group-aware splits):")
print("  principal angles (deg):", np.round(np.mean(floor_angles, axis=0), 1))
print("  per-direction cos     :", dict(zip(pc_names, np.round(np.mean(floor_cos, axis=0), 3))))

for name in TEMPLATES:
	print(f"base vs {name!r}:")
	print("  principal angles (deg):", np.round(principal_angles(dirs["base"], dirs[name]), 1))
	print("  per-direction cos     :", dict(zip(pc_names, np.round([np.dot(dirs["base"][p], dirs[name][p]) for p in range(pca_scores.shape[1])], 3))))

np.savez("prompt_stability_directions.npz", **dirs)
