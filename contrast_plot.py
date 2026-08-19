'''
steering sweep plot
'''

import numpy as np
import matplotlib.pyplot as plt

d = np.load("steering_results.npz")
results, losses, mags = d["results"], d["losses"], list(d["mags"])
directions, sets = list(d["directions"]), list(d["sets"])
zero = mags.index(0)

fig, ax = plt.subplots(1, len(sets) + 1, figsize=(6*(len(sets) + 1) - 4, 5))
for si, sname in enumerate(sets):
	for di, dname in enumerate(directions):
		ax[si].plot(mags, results[di, :, :, si].mean(axis=1) - results[di, zero, :, si].mean(), label=dname)
	ax[si].set_title(rf"$\Delta \log p$ ({sname})")
	ax[si].axhline(0, c="gray", lw=0.5)
	ax[si].set_xlabel(r"$\alpha$ (SD)")
	ax[si].legend()
for di, dname in enumerate(directions):
	ax[-1].plot(mags, losses[di] - losses[di, zero], label=dname)
ax[-1].set_title(r"$\Delta$ LM loss")
ax[-1].axhline(0, c="gray", lw=0.5)
ax[-1].set_xlabel(r"$\alpha$ (SD)")
ax[-1].legend()
plt.savefig("plots/steering_dose_response.png")

si, sj = sets.index("sweet"), sets.index("savory")
plt.figure(figsize=(9, 6))
for di, dname in enumerate(directions):
	contrast = results[di, :, :, si].mean(axis=1) - results[di, :, :, sj].mean(axis=1)
	plt.plot(mags, contrast - contrast[zero], label=dname)
plt.axhline(0, c="gray", lw=0.5)
plt.xlabel(r"$\alpha$ (SD)")
plt.ylabel(r"$\Delta(\log p_{sweet} - \log p_{savory})$")
plt.legend()
plt.savefig("plots/steering_contrast.png")
