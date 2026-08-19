import numpy as np
from common import load_arrays, get_model, extract_activations
import matplotlib.pyplot as plt
import torch

activations, taste_vectors, stds, pca_scores, raw_names, names, token_counts = load_arrays()
steering_vectors = np.load("steering_vectors.npy")
processor, model = get_model()

v1 = steering_vectors[0]
stds_proj = []
corr_proj = []
covar = []

for layer in range(activations.shape[1]):
	proj = activations[:, layer, :] @ v1
	std = np.std(proj)
	corr = np.corrcoef(proj, pca_scores[:, 0])[0, 1]
	print(layer, std, corr)
	stds_proj.append(std)
	corr_proj.append(corr)
	covar.append(std * corr)

fig, ax = plt.subplots(1, 3, figsize=(20,5))
ax[0].plot(np.arange(activations.shape[1]), stds_proj)
ax[1].plot(np.arange(activations.shape[1]), corr_proj)
ax[2].plot(np.arange(activations.shape[1]), covar)
ax[2].set_xlabel("layers")
ax[0].set_ylabel("std")
ax[0].set_xlabel("layers")
ax[1].set_xlabel("layers")
ax[1].set_ylabel("r")
ax[2].set_ylabel("delta_covar")
plt.savefig("plots/proj_onto_v1.png")


interesting_jumps = [7] + list(range(21, 28)) + [28] + [3, 12]
interesting_jumps = np.array(interesting_jumps)
captured = {}
def catcher(key):
	def hook(module, inputs, outputs):
		captured[key]= outputs[0].float().cpu().numpy()
	return hook
#baseline runs (before ablation)
handles = []
for layer in interesting_jumps:
	block = model.model.language_model.layers[layer]
	handles.append(block.post_attention_layernorm.register_forward_hook(catcher((layer, "attention"))))
	handles.append(block.post_feedforward_layernorm.register_forward_hook(catcher((layer, "mlp"))))

blocks = {(layer, type): [] for layer in interesting_jumps for type in ["attention", "mlp"]}
vec_sums = {(layer, type): np.zeros(2560) for layer in interesting_jumps for type in ["attention", "mlp"]}
try:
	for food in raw_names:
		prompt = "Food: " + food
		enc = processor.tokenizer(prompt, return_offsets_mapping=True)
		keep = [k for k, (a, b) in enumerate(enc["offset_mapping"]) if (a < prompt.index(food)+len(food)) and (b > prompt.index(food))]
		inputs = processor(text=prompt, return_tensors="pt").to(model.device)
		with torch.inference_mode():
			model(**inputs, use_cache=False)
		for key, h in captured.items():
			blocks[key].append(h[keep].mean(axis=0) @ v1)
			vec_sums[key] += h[keep].mean(axis=0)
finally:
	for h in handles:
		h.remove()
resid_cov = [np.cov(activations[:, l, :] @ v1, pca_scores[:, 0])[0, 1] for l in range(activations.shape[1])]
attention_contributions = []
mlp_contributions = []
for layer in interesting_jumps:
	attention_contribution = np.cov(np.array(blocks[(layer, "attention")]), pca_scores[:, 0])[0, 1]
	mlp_contribution = np.cov(np.array(blocks[(layer, "mlp")]), pca_scores[:, 0])[0, 1]
	print(f"layer {layer} |  attention:{attention_contribution}, mlp:{mlp_contribution}, residual:{resid_cov[layer+1] - resid_cov[layer] - attention_contribution - mlp_contribution}")
	attention_contributions.append(attention_contribution)
	mlp_contributions.append(mlp_contribution)

attention_contributions = np.array(attention_contributions)
mlp_contributions = np.array(mlp_contributions)

key_attentions = interesting_jumps[attention_contributions > 0.5*np.std(attention_contributions) + np.mean(attention_contributions)]
print(key_attentions)
key_mlps = interesting_jumps[mlp_contributions > 0.5*np.std(mlp_contributions) + np.mean(mlp_contributions)]
print(key_mlps)

candidates = [(l, "attention") for l in key_attentions] + [(l, "mlp") for l in key_mlps] + [(3, "mlp"), (12, "attention")]
mean_vecs = {key: vec_sums[key] / len(raw_names) for key in vec_sums}

probe_prompts = ["Water tastes very", "Pizza tastes very", "Cake tastes very", "Bread tastes very"]
ppl_text = "The moon orbits the earth once every twenty seven days, and its gravity is the main cause of the ocean tides. Astronomers have mapped its surface in detail since the invention of the telescope."
sweet_ids = [processor.tokenizer.encode(w, add_special_tokens=False)[0] for w in [" sweet", " sugary", " syrup"]]
savory_ids = [processor.tokenizer.encode(w, add_special_tokens=False)[0] for w in [" salty", " savory", " greasy", " meaty"]]
behavior_idx = np.argsort(pca_scores[:, 0])[::12]

def contrast(text):
	inputs = processor(text=text, return_tensors="pt").to(model.device)
	with torch.inference_mode():
		logp = torch.log_softmax(model(**inputs).logits[0, -1].float(), dim=-1)
	return np.mean([float(logp[t]) for t in sweet_ids]) - np.mean([float(logp[t]) for t in savory_ids])

def lm_loss(text):
	inputs = processor(text=text, return_tensors="pt").to(model.device)
	with torch.inference_mode():
		return float(model(**inputs, labels=inputs["input_ids"]).loss)

def behavioral():
	food_contrasts = [contrast(f"The taste of {raw_names[i]} is mostly") for i in behavior_idx]
	return {"contrast": np.mean([contrast(p) for p in probe_prompts]), "correlation": np.corrcoef(food_contrasts, pca_scores[behavior_idx, 0])[0, 1], "legibility": lm_loss(ppl_text)}

def top5(text):
	inputs = processor(text=text, return_tensors="pt").to(model.device)
	with torch.inference_mode():
		logp = torch.log_softmax(model(**inputs).logits[0, -1].float(), dim=-1)
	return processor.tokenizer.convert_ids_to_tokens(torch.topk(logp, 5).indices.tolist())

baseline_behavior = behavioral()
print("baseline")
print(baseline_behavior)
print(top5("Water tastes very"))
print(top5("Cake tastes very"))

#ablation runs
for key in candidates:
	block = model.model.language_model.layers[key[0]]
	if key[1] == "attention":
		mod = block.post_attention_layernorm
	else:
		mod = block.post_feedforward_layernorm

	def ablate(module, inputs, outputs, key=key):
		return torch.tensor(mean_vecs[key]).to(outputs.device, outputs.dtype).expand_as(outputs)

	handle = mod.register_forward_hook(ablate)
	try:
		projs, _ = extract_activations("Food: {x}", layer=28)
		behavior = behavioral()
		steered_top5 = top5("Water tastes very"), top5("Cake tastes very")
	finally:
		handle.remove()
	print("--------------------------------------------------------------")
	print(key)
	print(f"delta covar: {np.cov(projs @ v1, pca_scores[:, 0])[0, 1] - resid_cov[28]}")
	print([f"{k}: {behavior[k]-baseline_behavior[k]}" for k in behavior.keys()])
	print(f"{key}, water: {steered_top5[0]}")
	print(f"{key}, cake: {steered_top5[1]}")

#full ablation
def make_ablate(key):
	def ablate(module, inputs, outputs):
		return torch.tensor(mean_vecs[key]).to(outputs.device, outputs.dtype).expand_as(outputs)
	return ablate

joint = [(l, "mlp") for l in key_mlps] + [(l, "attention") for l in key_attentions]
handles_full = []
for key in joint:
	block = model.model.language_model.layers[key[0]]
	handles_full.append((block.post_attention_layernorm if key[1] == "attention" else block.post_feedforward_layernorm).register_forward_hook(make_ablate(key)))
try:
	projs, _ = extract_activations("Food: {x}", layer=28)
	projs33, _ = extract_activations("Food: {x}", layer=33)
	behavior = behavioral()
	steered_top5 = top5("Water tastes very"), top5("Cake tastes very")
finally:
	for h in handles_full:
		h.remove()

print("--------------------------------------------------------------")
print(f"joint {joint}")
print(f"delta covar: {np.cov(projs @ v1, pca_scores[:, 0])[0, 1] - resid_cov[28]}")
print(f"correlation: {np.corrcoef(projs @ v1, pca_scores[:, 0])[0, 1]}")
print([f"{k}: {behavior[k]-baseline_behavior[k]}" for k in behavior.keys()])
print(f"water: {steered_top5[0]}")
print(f"cake: {steered_top5[1]}")

print(f"layer 33 delta covar: {np.cov(projs @ v1, pca_scores[:, 0])[0, 1] - resid_cov[33]}")
print(f"layer 33 correlation: {np.corrcoef(projs @ v1, pca_scores[:, 0])[0, 1]}")
