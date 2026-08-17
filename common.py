import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_score

rng = np.random.default_rng()

human_tastes = pd.read_csv("food_taste_database/food_taste_database.csv")
human_tastes = human_tastes.sample(frac=1, random_state=42).reset_index(drop=True) #shuffling bc df is sorted alphabetically
labels = [name for name in human_tastes.columns if "Mean" in name]

#group assignment for groupkfold
STOPWORDS = {
	"green", "red", "black", "white", "brown", "dark", "light",
	"sweet", "sour", "hot", "dry", "fresh", "soft", "hard", "strong",
	"whole", "non", "mixed", "sparkling", "fortified", "bottled",
	"fried", "baked", "cooked", "smoked", "salted", "candied",
	"mashed", "minced", "marinated", "pickled", "roasted", "stewed",
	"stuffed", "breaded", "dry-cured", "whole-grain", "country-style",
}

def stem(word):
	word = word.lower().strip(".,").replace("'s", "").replace("®", "")
	if word.endswith("ies"):
		return word[:-3] + "y"
	if word.endswith("s") and not word.endswith("ss"):
		return word[:-1]
	return word

human_tastes["group"] = [next((stem(w) for w in food.split("(")[0].split() if stem(w) not in STOPWORDS), stem(food.split()[0])) for food in human_tastes["Food"]]

mask = (human_tastes["n"] > 1) & (human_tastes.isna().sum(axis=1) == 0)
foods = human_tastes.loc[mask, "Food"].tolist()
groups = np.array(human_tastes.loc[mask, "group"])

def load_arrays():
	activations = np.load("activations.npy")
	taste_vectors = np.load("taste_vectors.npy")
	stds = np.load("taste_stds.npy")
	try:
		pca_scores = np.load("top_taste_pca.npy")
	except FileNotFoundError:
		pca_scores = None
	with open("raw_names.pkl", "rb") as f:
		raw_names = np.array(pickle.load(f))
	with open("names.pkl", "rb") as f:
		names = pickle.load(f)
	token_counts = np.array([len(t) for t in names])
	return activations, taste_vectors, stds, pca_scores, raw_names, names, token_counts

_processor = None
_model = None

def get_model():
	global _processor, _model
	if _model is None:
		import torch
		from transformers import AutoProcessor, Gemma3ForConditionalGeneration
		_processor = AutoProcessor.from_pretrained("google/gemma-3-4b-pt")
		_model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-pt", device_map="auto", torch_dtype=torch.bfloat16).eval()
	return _processor, _model

def extract_activations(template, layer=None):
	import torch
	processor, model = get_model()
	acts, name_tokens = [], []
	for food in foods:
		prompt = template.format(x=food)
		# token span of the food name itself, template-agnostic:
		# keep tokens whose char range overlaps [start, end) of the name.
		start = prompt.index(food)
		end = start + len(food)
		enc = processor.tokenizer(prompt, return_offsets_mapping=True)
		keep = [k for k, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start]
		assert keep, f"empty name span for {food!r} in template {template!r}"
		inputs = processor(text=prompt, return_tensors="pt").to(model.device)
		with torch.inference_mode():
			out = model(**inputs, output_hidden_states=True, use_cache=False)
		if layer is None:
			h = torch.stack([hs[0] for hs in out.hidden_states[:-1]]).float().cpu().numpy()
			acts.append(h[:, keep, :].mean(axis=1)) #average across tokens
		else:
			h = out.hidden_states[layer][0].float().cpu().numpy()
			acts.append(h[keep].mean(axis=0))
		name_tokens.append([enc["input_ids"][k] for k in keep])
	return np.array(acts, dtype=np.float32), name_tokens

def zscore_eps(m, eps=1e-4):
	return (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + eps)

def haufe(xs, ys, alphas=np.logspace(-1, 8, 30)):
	w = RidgeCV(alphas=alphas).fit(xs, ys).coef_
	xc = xs - xs.mean(axis=0)
	return xc.T @ (xc @ w) / (xs.shape[0] - 1)

def principal_angles(b1, b2):
	q1 = np.linalg.qr(b1.T)[0]
	q2 = np.linalg.qr(b2.T)[0]
	sv = np.linalg.svd(q1.T @ q2, compute_uv=False)
	return np.degrees(np.arccos(np.clip(sv, 0.0, 1.0)))

def cv_stats(results):
	return [np.min(results), np.max(results), np.std(results), np.mean(results)]

def plot_probe_axis(ax, axis_results, axis_results_shuffled, axis_name):
	axis_results = np.array(axis_results)
	axis_results_shuffled = np.array(axis_results_shuffled)
	ax.fill_between(np.arange(axis_results.shape[0]), axis_results[:, -1]-axis_results[:, -2], axis_results[:, -1]+axis_results[:, -2], alpha=0.4, label=axis_name)
	ax.fill_between(np.arange(axis_results_shuffled.shape[0]), axis_results_shuffled[:, -1]-axis_results_shuffled[:, -2], axis_results_shuffled[:, -1]+axis_results_shuffled[:, -2], alpha=0.4, label=axis_name + " shuffled")
	ax.plot(axis_results[:, -1])
	ax.plot(axis_results_shuffled[:, -1])
	ax.set_ylim(-1, 1)
	ax.legend()
	ax.set_xlabel("layer")

def probe_panel(X, Y, axis_names, save_path, weights=None):
	fig, ax = plt.subplots(1, Y.shape[1], figsize=(7*Y.shape[1] - 5, 5))
	for axis in range(Y.shape[1]):
		axis_name = axis_names[axis]
		print(axis_name)
		if weights is not None:
			stds_axis = weights[:, axis].copy()
			stds_axis[stds_axis <= 0] = np.median(weights[:, axis])
			w = 1/(stds_axis)**2
			params = {'sample_weight': w}
		else:
			w = None
			params = None
		axis_results = []
		axis_results_shuffled = []
		for layer in range(X.shape[1]):
			x = X[:, layer, :]
			y = Y[:, axis]
			ridge_model = RidgeCV(alphas=np.logspace(-1, 8, 30))
			cross_val_results = cross_val_score(ridge_model, x, y, cv=GroupKFold(n_splits=10), params=params, groups=groups, scoring='r2')
			axis_results.append(cv_stats(cross_val_results))
			if w is not None:
				yw = np.array([y, w]).T
				yw_shuffled = np.random.default_rng().permutation(yw)
				y_shuffled, params_shuffled = yw_shuffled[:, 0], {'sample_weight': yw_shuffled[:, 1]}
			else:
				y_shuffled, params_shuffled = np.random.default_rng().permutation(y), None
			cross_val_results_shuffled = cross_val_score(ridge_model, x, y_shuffled, cv=GroupKFold(n_splits=10), params=params_shuffled, groups=groups, scoring='r2')
			axis_results_shuffled.append(cv_stats(cross_val_results_shuffled))
		plot_probe_axis(ax[axis], axis_results, axis_results_shuffled, axis_name)
	ax[0].set_ylabel(r"$r^2$")
	plt.savefig(save_path)
