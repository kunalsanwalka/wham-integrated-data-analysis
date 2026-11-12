import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

d = np.load("/home/sanwalka/integrated_data_analysis/warpx_data/fields_rtz.npz")                      # fields have shape (dump, r, theta, z)
rz = lambda k: d[k].mean(axis=(0, 2))              # average over dumps and theta -> (r, z)
ne = rz("n_electrons")

fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
for ax, (k, label, log, cmap) in zip(axs, [("n_electrons", "n$_e$ (m$^{-3}$)", 1, "viridis"),
                                           ("T_electrons", "T$_e$ (eV)", 0, "inferno"),
                                           ("n_neutrals", "n$_0$ (m$^{-3}$)", 1, "cividis")]):
    a = rz(k) if k != "T_electrons" else np.ma.masked_where(ne < 1e-3 * ne.max(), rz(k))  # hide T where no plasma
    im = ax.pcolormesh(d["z"], d["r"], a, shading="auto", cmap=cmap, norm=LogNorm(a.max() / 1e4, a.max()) if log else None)
    fig.colorbar(im, ax=ax, label=label); ax.set_ylabel("r (m)")
axs[-1].set_xlabel("z (m)")
fig.suptitle(f"theta-averaged, mean of t = {d['t'][0]*1e6:.0f}-{d['t'][-1]*1e6:.0f} µs")
plt.savefig("/home/sanwalka/integrated_data_analysis/plots/rtz_profiles.png", dpi=150)