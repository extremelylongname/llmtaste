"""
performing ridge regression between input taste dimensions and model activations
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

processor = AutoProcessor.from_pretrained("google/gemma-3-4b-pt")
model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-pt", device_map="auto", torch_dtype=torch.bfloat16).eval()

human_tastes = pd.read_csv("food_taste_database/food_taste_database.csv")

print(human_tastes.head())

def fetch_hidden_states(text):
	inputs = processor(text=text, return_tensors="pt").to(model.device)
	with torch.inference_mode():
		output = model(**inputs, output_hidden_states=True, use_cache=False)

	return torch.stack([h[0].float().cpu() for h in output.hidden_states])

prefix_toks = processor.tokenizer("Food:", return_tensors="pt", add_special_tokens=False)["input_ids"].cpu().numpy()

Y=[]
X=[]
stds = []

for i, row in human_tastes.iterrows():
	if (row["n"] > 1) and (row.isna().sum() == 0):
		prompt = "Food:" + row["Food"]
		Y.append([row[col] for col in row.index if "Mean" in col])
		stds.append([row[col]/np.sqrt(row["n"]) for col in row.index if "STD" in col])
		X.append(np.mean(fetch_hidden_states(prompt)[:, 1+prefix_toks.shape[-1]:, :].numpy(), axis=1)) #average across tokens

X = np.array(X)
Y = np.array(Y)
stds = np.array(stds)

fig, ax = plt.subplots(figsize=(10, 10))

for axis in range(Y.shape[1]):
	axis_name=[col for col in human_tastes.columns if "Mean" in col][axis]
	print(axis_name)
	stds_axis = stds[:, axis]
	stds_axis[stds_axis <= 0] = np.median(stds[:, axis])
	axis_results = []
	for layer in range(X.shape[1]):
		x = X[:, layer, :]
		y = Y[:, axis]
		w = 1/(stds_axis)**2
		ridge_model = RidgeCV(alphas=np.logspace(-1, 8, 30))
		cross_val_results = cross_val_score(ridge_model, x, y, cv=10, params={'sample_weight': w}, scoring='r2')
		print(f"{np.mean(cross_val_results)} +/- {np.std(cross_val_results)}")
		axis_results.append([np.min(cross_val_results), np.max(cross_val_results), np.std(cross_val_results), np.mean(cross_val_results)]) 
	axis_results = np.array(axis_results)
	plt.fill_between(np.arange(X.shape[1]), axis_results[:, -1]-axis_results[:, -2], axis_results[:, -1]+axis_results[:, -2], alpha=0.4, label=axis_name)
	plt.plot(axis_results[:, -1])

plt.legend()
plt.xlabel("layer")
plt.ylabel("r^2")
plt.savefig("plots/ridgeregression.png") 


