'''
per-head taste contributions at the key attention blocks
'''

import numpy as np
import torch
from common import load_arrays, get_model

activations, taste_vectors, stds, pca_scores, raw_names, names, token_counts = load_arrays()
steering_vectors = np.load("steering_vectors.npy")
processor, model = get_model()
v1 = steering_vectors[0]

interesting_attention_layers = [24, 26]
n_heads, head_dim = 8, 256

captured = {}
def catch_oproj_input(layer):
	def hook(module, inputs):
		captured[(layer, "pre")] = inputs[0][0].float().cpu().numpy()
	return hook

def catch_norm_scale(layer):
	def hook(module, inputs, outputs):
		inp = inputs[0][0].float().cpu().numpy()
		captured[(layer, "scale")] = outputs[0].float().cpu().numpy() / np.where(np.abs(inp) < 1e-9, 1e-9, inp)
	return hook

handles = []
for layer in interesting_attention_layers:
	block = model.model.language_model.layers[layer]
	handles.append(block.self_attn.o_proj.register_forward_pre_hook(catch_oproj_input(layer)))
	handles.append(block.post_attention_layernorm.register_forward_hook(catch_norm_scale(layer)))

head_projs = {(layer, h): [] for layer in interesting_attention_layers for h in range(n_heads)}
try:
	for food in raw_names:
		prompt = "Food: " + food
		enc = processor.tokenizer(prompt, return_offsets_mapping=True)
		keep = [k for k, (a, b) in enumerate(enc["offset_mapping"]) if (a < prompt.index(food)+len(food)) and (b > prompt.index(food))]
		inputs = processor(text=prompt, return_tensors="pt").to(model.device)
		with torch.inference_mode():
			model(**inputs, use_cache=False)
		for layer in interesting_attention_layers:
			pre = captured[(layer, "pre")]
			scale = captured[(layer, "scale")]
			masked = np.zeros((n_heads, pre.shape[0], n_heads*head_dim), dtype=np.float32)
			for h in range(n_heads):
				masked[h, :, h*head_dim:(h+1)*head_dim] = pre[:, h*head_dim:(h+1)*head_dim]
			with torch.inference_mode():
				outs = model.model.language_model.layers[layer].self_attn.o_proj(torch.tensor(masked).to(model.device, torch.bfloat16)).float().cpu().numpy()
			for h in range(n_heads):
				head_projs[(layer, h)].append(((outs[h] * scale)[keep]).mean(axis=0) @ v1)
finally:
	for h in handles:
		h.remove()

covs = {key: np.cov(np.array(p), pca_scores[:, 0])[0, 1] for key, p in head_projs.items()}
print("head ranking (taste covariance along v1):")
for key, c in sorted(covs.items(), key=lambda kv: -abs(kv[1])):
	print(f"layer {key[0]} head {key[1]}: {c:+.1f}")
print("per-layer head sums:", {layer: round(sum(covs[(layer, h)] for h in range(n_heads)), 1) for layer in interesting_attention_layers})

#attention patterns + source-position path trace for the dominant heads and their kv partners
model.model.language_model.config._attn_implementation = "eager"
target_heads = [(24, 4), (24, 5), (26, 2), (26, 3)]
bucket_names = ["bos", "prefix"] + [f"name_{i}" for i in range(6)] + ["name_6plus"]

def catch_v(layer):
	def hook(module, inputs, outputs):
		captured[(layer, "v")] = outputs[0].float().cpu().numpy()
	return hook

handles = []
for layer in interesting_attention_layers:
	block = model.model.language_model.layers[layer]
	handles.append(block.self_attn.o_proj.register_forward_pre_hook(catch_oproj_input(layer)))
	handles.append(block.post_attention_layernorm.register_forward_hook(catch_norm_scale(layer)))
	handles.append(block.self_attn.v_proj.register_forward_hook(catch_v(layer)))

masses = {key: np.zeros(len(bucket_names)) for key in target_heads}
mass_counts = {key: np.zeros(len(bucket_names)) for key in target_heads}
writes = {(key, b): [] for key in target_heads for b in bucket_names}
write_idx = {(key, b): [] for key in target_heads for b in bucket_names}
checked = False
try:
	for fi, food in enumerate(raw_names):
		prompt = "Food: " + food
		enc = processor.tokenizer(prompt, return_offsets_mapping=True)
		keep = [k for k, (a, b) in enumerate(enc["offset_mapping"]) if (a < prompt.index(food)+len(food)) and (b > prompt.index(food))]
		inputs = processor(text=prompt, return_tensors="pt").to(model.device)
		with torch.inference_mode():
			out = model(**inputs, use_cache=False, output_attentions=True)
		seq = out.attentions[0].shape[-1]
		buckets = {"bos": [0], "prefix": [p for p in range(seq) if p != 0 and p not in keep]}
		for i, k in enumerate(keep[:6]):
			buckets[f"name_{i}"] = [k]
		if len(keep) > 6:
			buckets["name_6plus"] = keep[6:]
		truth_pre = {layer: captured[(layer, "pre")].copy() for layer in interesting_attention_layers} if not checked else None
		for layer, h in target_heads:
			A = out.attentions[layer][0, h].float().cpu().numpy()
			vg = captured[(layer, "v")][:, (h//2)*head_dim:(h//2+1)*head_dim]
			if not checked:
				recon = A @ vg
				truth = truth_pre[layer][:, h*head_dim:(h+1)*head_dim]
				print(f"reconstruction check L{layer}H{h}: cos {np.sum(recon*truth)/(np.linalg.norm(recon)*np.linalg.norm(truth)):.4f}")
			present = [b for b in bucket_names if b in buckets]
			for b in present:
				masses[(layer, h)][bucket_names.index(b)] += A[np.ix_(keep, buckets[b])].sum(axis=1).mean()
				mass_counts[(layer, h)][bucket_names.index(b)] += 1
			pre_buckets = np.zeros((len(present), seq, n_heads*head_dim), dtype=np.float32)
			for pi, b in enumerate(present):
				pre_buckets[pi, :, h*head_dim:(h+1)*head_dim] = A[:, buckets[b]] @ vg[buckets[b]]
			with torch.inference_mode():
				outs = model.model.language_model.layers[layer].self_attn.o_proj(torch.tensor(pre_buckets).to(model.device, torch.bfloat16)).float().cpu().numpy()
			for pi, b in enumerate(present):
				writes[((layer, h), b)].append(((outs[pi] * captured[(layer, "scale")])[keep]).mean(axis=0) @ v1)
				write_idx[((layer, h), b)].append(fi)
		checked = True
finally:
	for h in handles:
		h.remove()

for key in target_heads:
	print(f"L{key[0]}H{key[1]}  mean attention mass (n foods):")
	print("  " + ", ".join(f"{b} {masses[key][bi]/mass_counts[key][bi]:.3f} (n={int(mass_counts[key][bi])})" for bi, b in enumerate(bucket_names) if mass_counts[key][bi] > 0))
	print(f"L{key[0]}H{key[1]}  taste write by source:")
	print("  " + ", ".join(f"{b} {np.cov(np.array(writes[(key, b)]), pca_scores[write_idx[(key, b)], 0])[0, 1]:+.1f}" for b in bucket_names if len(writes[(key, b)]) > 2))

#stage 9: substitute the learned attention of the key heads with a symbolic rule
SUB = {"on": False, "mode": "name", "heads": [(24, 4), (26, 2)], "keep": []}
resid_cov28 = np.cov(activations[:, 28, :] @ v1, pca_scores[:, 0])[0, 1]

def name_span(text, name):
	s = text.index(name)
	return [k for k, (a, b) in enumerate(processor.tokenizer(text, return_offsets_mapping=True)["offset_mapping"]) if (a < s + len(name)) and (b > s)]

def sub_hook(layer):
	def hook(module, args):
		if not SUB["on"]:
			return
		x = args[0].clone()
		seq = x.shape[1]
		src = SUB["keep"] if SUB["mode"] == "name" else list(range(seq))
		A_rule = np.zeros((seq, seq), dtype=np.float32)
		for q in range(seq):
			vis = [j for j in src if j <= q]
			if vis:
				A_rule[q, vis] = 1.0 / len(vis)
		rows = A_rule.sum(axis=1) > 0
		for l, h in SUB["heads"]:
			if l == layer:
				repl = torch.tensor(A_rule @ captured[(layer, "v")][:, (h//2)*head_dim:(h//2+1)*head_dim]).to(x.device, x.dtype)
				x[0, rows, h*head_dim:(h+1)*head_dim] = repl[rows]
		return (x,)
	return hook

handles = []
for layer in interesting_attention_layers:
	block = model.model.language_model.layers[layer]
	handles.append(block.self_attn.v_proj.register_forward_hook(catch_v(layer)))
	handles.append(block.self_attn.o_proj.register_forward_pre_hook(sub_hook(layer)))
	handles.append(block.self_attn.o_proj.register_forward_pre_hook(catch_oproj_input(layer)))
	handles.append(block.post_attention_layernorm.register_forward_hook(catch_norm_scale(layer)))

probe_prompts = [("Water tastes very", "Water"), ("Pizza tastes very", "Pizza"), ("Cake tastes very", "Cake"), ("Bread tastes very", "Bread")]
ppl_text = "The moon orbits the earth once every twenty seven days, and its gravity is the main cause of the ocean tides. Astronomers have mapped its surface in detail since the invention of the telescope."
sweet_ids = [processor.tokenizer.encode(w, add_special_tokens=False)[0] for w in [" sweet", " sugary", " syrup"]]
savory_ids = [processor.tokenizer.encode(w, add_special_tokens=False)[0] for w in [" salty", " savory", " greasy", " meaty"]]
behavior_idx = np.argsort(pca_scores[:, 0])[::12]

def forward_ctx(text, name, labels=False):
	SUB["keep"] = name_span(text, name) if name else []
	SUB["on"] = True
	inputs = processor(text=text, return_tensors="pt").to(model.device)
	with torch.inference_mode():
		out = model(**inputs, use_cache=False, **({"labels": inputs["input_ids"]} if labels else {}))
	SUB["on"] = False
	return out

def contrast(text, name):
	logp = torch.log_softmax(forward_ctx(text, name).logits[0, -1].float(), dim=-1)
	return np.mean([float(logp[t]) for t in sweet_ids]) - np.mean([float(logp[t]) for t in savory_ids])

def behavioral():
	food_contrasts = [contrast(f"The taste of {raw_names[i]} is mostly", raw_names[i]) for i in behavior_idx]
	return {"contrast": np.mean([contrast(p, n) for p, n in probe_prompts]), "correlation": np.corrcoef(food_contrasts, pca_scores[behavior_idx, 0])[0, 1], "legibility": float(forward_ctx(ppl_text, None, labels=True).loss)}

def top5(text, name):
	logp = torch.log_softmax(forward_ctx(text, name).logits[0, -1].float(), dim=-1)
	return processor.tokenizer.convert_ids_to_tokens(torch.topk(logp, 5).indices.tolist())

def run_condition(label, mode, heads_list):
	SUB["mode"], SUB["heads"] = mode, heads_list
	head_writes = {key: [] for key in heads_list}
	for food in raw_names:
		prompt = "Food: " + food
		out = forward_ctx(prompt, food)
		keep = name_span(prompt, food)
		for l, h in heads_list:
			pre = captured[(l, "pre")]
			masked = np.zeros((1, pre.shape[0], n_heads*head_dim), dtype=np.float32)
			masked[0, :, h*head_dim:(h+1)*head_dim] = pre[:, h*head_dim:(h+1)*head_dim]
			with torch.inference_mode():
				outs = model.model.language_model.layers[l].self_attn.o_proj(torch.tensor(masked).to(model.device, torch.bfloat16)).float().cpu().numpy()
			head_writes[(l, h)].append(((outs[0] * captured[(l, "scale")])[keep]).mean(axis=0) @ v1)
	print(f"----- {label} -----")
	for key in heads_list:
		print(f"L{key[0]}H{key[1]} write under rule: {np.cov(np.array(head_writes[key]), pca_scores[:, 0])[0, 1]:+.1f}")
	print(f"behavior: {behavioral()}")
	print(f"water: {top5('Water tastes very', 'Water')}")
	print(f"cake: {top5('Cake tastes very', 'Cake')}")

try:
	forward_ctx("Food: test", None)
	SUB["heads"] = []
	print(f"baseline behavior: {behavioral()}")
	run_condition("uniform over name span, heads 24.4 + 26.2", "name", [(24, 4), (26, 2)])
	run_condition("uniform over ALL tokens, heads 24.4 + 26.2", "all", [(24, 4), (26, 2)])
	run_condition("null-head control (24.0), name rule", "name", [(24, 0)])
finally:
	for h in handles:
		h.remove()
