"""
This script holds the plot style and helpers shared by every comparison script, so that the
plots of the different diagnostics look the same and can be combined into one figure.
"""

import os
import matplotlib.pyplot as plt

import config

# Use TkAgg backend for interactive plotting
plt.switch_backend('TkAgg')

plt.rcParams.update({'font.size': 28})

# Color of the simulated / synthetic signal of each kind of measurement, so the panels of the same
# quantity match across diagnostics. The measurements themselves are drawn in black.
# Density (SEE, n_i vs. Thomson, ELA n). density_comparison.py draws simulation i in f'C{i}', so
# its first simulation matches this.
densityColor = 'C0'
# Temperature (T_e vs. Thomson, ELA T_e and T_i)
temperatureColor = 'C1'
# H-alpha (AXUV)
halphaColor = 'C3'
# Flux loops
fluxLoopColor = 'C2'

def save_plot(fig, filename):
    """
    Save the figure in config.plotDir, making the directory if it does not exist yet.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure.
    filename : str
        Name of the file inside config.plotDir.
    """

    os.makedirs(config.plotDir, exist_ok=True)

    savePath = config.plotDir + filename
    fig.savefig(savePath, dpi=600)
    print(f'Saved plot to {savePath}')

    return

def plot_mirrored_profile(ax, r, profile, numRSim=None, color='C0', label=None, marker=None):
    """
    Plot a midplane radial profile mirrored about r = 0, so that it shares an x axis with the
    line-integrated diagnostics, which are plotted against signed impact parameter.

    If the profile has been extended past the simulation grid (see extend_midplane_profile() in
    axuv_extension.py), the simulated part is drawn solid and the extension dashed, with the
    last simulated point repeated to close the gap between them.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis the profile is drawn on.
    r : np.array
        1D array of the midplane radius of each point. [m]
    profile : np.array
        1D array of the profile on r.
    numRSim : int
        Number of points of r on the simulation grid, with the rest being the extension.
        Default is None, which draws the whole profile solid.
    color : str
        Color of the profile.
        Default is 'C0'.
    label : str
        Legend label of the simulated part. The extension is labelled 'AXUV extension'.
        Default is None, which leaves both out of the legend.
    marker : str
        Marker drawn at each point of the simulated part, e.g. 'o' to show the flux tubes.
        Default is None, which draws no markers.
    """

    if numRSim is None:
        numRSim = len(r)

    for sign in (1, -1):

        ax.plot(sign*r[:numRSim], profile[:numRSim],
                color=color,
                linewidth=3,
                marker=marker,
                ms=10,
                label=label if sign == 1 else None)

        if numRSim < len(r):
            ax.plot(sign*r[numRSim-1:], profile[numRSim-1:],
                    color=color,
                    linewidth=3,
                    linestyle='dashed',
                    label='AXUV extension' if (sign == 1 and label is not None) else None)

    return
