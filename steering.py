import numpy as np
from common import load_arrays, get_model, haufe
import torch

activations, taste_vectors, stds, pca_scores, raw_names, names, token_counts = load_arrays()
processor, model = get_model()

layer = 28
steering_vectors = np.load("steering_vectors.npy")
x28 = activations[:, layer, :]
s = np.std(x28 @ steering_vectors[0])
rng0 = np.random.default_rng(0)

for i in range(20):
	shuffled_dir = haufe(x28, rng0.permutation(pca_scores[:, 0]))
	shuffled_dir /= np.linalg.norm(shuffled_dir)
	if np.max(np.abs(steering_vectors[:3] @ shuffled_dir)) < 0.1:
		break
else:
	shuffled_dir -= steering_vectors[:3].T @ (steering_vectors[:3] @ shuffled_dir)
	shuffled_dir /= np.linalg.norm(shuffled_dir)
print("shuffled cos vs pc1-3:", np.round(steering_vectors[:3] @ shuffled_dir, 3))
random_dir = rng0.standard_normal(x28.shape[1])
random_dir /= np.linalg.norm(random_dir)

directions = {"pc1": steering_vectors[0], "pc2": steering_vectors[1], "pc3": steering_vectors[2], "random": random_dir, "shuffled": shuffled_dir}

word_sets = {
	"sweet": [" sweet", " sugary", " syrup"],
	"drink": [" drink", " juice", " beverage", " tea"],
	"savory": [" salty", " savory", " greasy", " meaty"],
	"bland": [" bland", " plain", " watery"]}

set_ids = {k: [processor.tokenizer.encode(w, add_special_tokens=False)[0] for w in ws if len(processor.tokenizer.encode(w, add_special_tokens=False)) == 1] for k, ws in word_sets.items()}
print({k: processor.tokenizer.convert_ids_to_tokens(ids) for k, ids in set_ids.items()})

prompts = ["Water tastes very", "Pizza tastes very", "Cake tastes very", "Bread tastes very"]
ppl_text = "The moon orbits the earth once every twenty seven days, and its gravity is the main cause of the ocean tides. Astronomers have mapped its surface in detail since the invention of the telescope."
mags = [-16, -12, -8, -4, 0, 4, 8, 12, 16]

layer_module = model.model.language_model.layers[layer-1]

def forward_logp(text, labels=False):
	inputs = processor(text=text, return_tensors="pt").to(model.device)
	with torch.inference_mode():
		out = model(**inputs, **({"labels": inputs["input_ids"]} if labels else {}))
	return float(out.loss) if labels else torch.log_softmax(out.logits[0, -1].float(), dim=-1)

def steered(fn, vec, a):
	if a == 0:
		return fn()
	vt = torch.tensor(vec)
	def steer(module, inputs, output):
		h = output[0] if isinstance(output, tuple) else output
		h += a * vt.to(h.device, h.dtype)
		return output
	assert not layer_module._forward_hooks
	handle = layer_module.register_forward_hook(steer)
	try:
		return fn()
	finally:
		handle.remove()

results = np.zeros((len(directions), len(mags), len(prompts), len(word_sets)))
losses = np.zeros((len(directions), len(mags)))

for di, (dname, dvec) in enumerate(directions.items()):
	for mi, mag in enumerate(mags):
		for pi, prompt in enumerate(prompts):
			logp = steered(lambda: forward_logp(prompt), dvec, mag * s)
			for si, ids in enumerate(set_ids.values()):
				results[di, mi, pi, si] = np.mean([float(logp[t]) for t in ids])
		losses[di, mi] = steered(lambda: forward_logp(ppl_text, labels=True), dvec, mag * s)
	print(f"{dname} done")

np.savez("steering_results.npz", results=results, losses=losses, mags=mags, directions=list(directions.keys()), prompts=prompts, sets=list(word_sets.keys()))
