'''
extract activations and labels for the food dataset
'''

import pickle
import numpy as np
from common import human_tastes, mask, foods, groups, extract_activations

X, names = extract_activations("Food: {x}")

filtered = human_tastes.loc[mask]
Y = filtered[[col for col in human_tastes.columns if "Mean" in col]].to_numpy()
stds = filtered[[col for col in human_tastes.columns if "STD" in col]].to_numpy() / np.sqrt(filtered["n"].to_numpy())[:, None]

with open("names.pkl", "wb") as f:
	pickle.dump(names, f)

with open("raw_names.pkl", "wb") as f:
	pickle.dump(foods, f)

np.save("groups.npy", groups)
np.save("activations.npy", X)
np.save("taste_vectors.npy", Y)
np.save("taste_stds.npy", stds)
