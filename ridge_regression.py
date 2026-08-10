"""
performing ridge regression between input taste dimensions and model activations
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from sklearn.linear_model import Ridge
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
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


X=[]
Y=[]

for i, row in human_tastes.iterrows():
	prompt = "Food:" + row["Food"]
	x_i = [row[col] for col in row.index if "Mean" in col]
	X.append(x_i)

	hidden = np.mean(fetch_hidden_states(prompt)[1:, 1:, :], axis=1) #average across tokens
	Y.append(hidden) #shape: [layers, d]

model = Ridge(alpha=1.0)
model.fit(X, Y)

print(model.coef_)
print(model.intercept_)
