"""
This script compares the measured flux loop signals against the synthetic ones of the simulation.

The synthetic signal of each loop is the change in the poloidal flux at its (R, Z) location from
the eqdsk the simulation is initialized with (the first IPS timestep) to the eqdsk of the IPS
timestep that covers simTime (synthetic_flux_loop_signal() in fluxloops.py).
"""

import numpy as np
import matplotlib.pyplot as plt

from plotting import save_plot, fluxLoopColor
from cql3d_simulation import ips_timestep_index, ips_timestep_dirs, load_flux_tube_data
from fluxloops import load_flux_loops, synthetic_flux_loop_signal
import config

def plot_fluxloop_comparison(simulationName, simTime, timestepIdx=None, refTimestepIdx=0,
                             showPlot=True, saveplot=False, axes=None):
    """
    Plot the measured flux of each flux loop against the synthetic one of the simulation, vs. the
    axial position of the loop.

    The flux loop data is not tied to a shot, as with the Thomson scattering data, so it is read
    from config.fluxLoopDataPath.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    simTime : float
        The simulation time. [s]
        The eqdsk of the IPS timestep whose CQL3D run covers it is used.
    timestepIdx : int
        Index of the IPS timestep to compare, in the order of ips_timestep_dirs(). Overrides
        simTime when given.
        Default is None, which uses simTime.
    refTimestepIdx : int
        Index of the IPS timestep of the reference eqdsk.
        Default is 0, the first IPS timestep (the initialization).
    showPlot : bool
        Show the figure before returning. Set to False to add to the figure afterwards.
        Default is True.
    saveplot : bool
        Save the plot in config.plotDir as f'{simulationName}_fluxloops_comparison.png'.
        Default is False.
    axes : matplotlib.axes.Axes
        The axis to draw into instead of making a new figure, e.g. to place this plot in a larger
        figure.
        Default is None, which makes a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The flux loop axis.
    """

    #### Measured flux loops

    R, Z, flux, fluxSigma, expTime = load_flux_loops()

    #### Synthetic flux loops

    if timestepIdx is None:
        timestepIdx = ips_timestep_index(simulationName, simTime)

    fluxSim = synthetic_flux_loop_signal(simulationName, R, Z, timestepIdx, refTimestepIdx)

    # Time at the end of the IPS timestep, for the legend
    timeArr, _, _ = load_flux_tube_data(simulationName)
    stepName = ips_timestep_dirs(config.simulationScanDir + simulationName + '/')[timestepIdx].split('/')[-2]
    stepStart = timeArr[timestepIdx-1] if timestepIdx % len(timeArr) > 0 else 0.0

    print('Flux loops [uWb]')
    for r, z, fMeas, fSigma, fSim in zip(R, Z, flux, fluxSigma, fluxSim):
        print(f'  (R, Z) = ({r:g}, {z:g}) m: measured {fMeas*1e6:.1f} +- {fSigma*1e6:.1f}, '
              f'simulated {fSim*1e6:.1f}')

    if axes is None:
        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = axes
        fig = ax.figure

    # [Wb] to [uWb]
    ax.errorbar(Z, flux*1e6,
                yerr=fluxSigma*1e6,
                color='k',
                fmt='o',
                ms=12,
                elinewidth=2,
                capsize=6)

    ax.plot(Z, fluxSim*1e6,
            color=fluxLoopColor,
            linestyle='none',
            marker='s',
            ms=14,
            label=('Simulation'))

    # Label each loop with its radius
    for r, z, top in zip(R, Z, np.maximum(flux + fluxSigma, fluxSim)*1e6):
        ax.annotate(f'R = {r:g} m', (z, top), xytext=(0, 12), textcoords='offset points',
                    ha='center', fontsize='small')

    ax.set_xlabel('Z [m]')
    ax.set_ylabel(r'Flux [$\mu$Wb]')
    ax.set_title(f'Flux loops, {simulationName}')

    # Headroom for the labels and the legend
    ax.set_ylim(0, 1.4 * max(np.max(flux + fluxSigma), np.max(fluxSim)) * 1e6)
    zPad = 0.25 * (np.max(Z) - np.min(Z))
    ax.set_xlim(np.min(Z) - zPad, np.max(Z) + zPad)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower center', fontsize='small')

    if saveplot:
        save_plot(fig, f'{simulationName}_fluxloops_comparison.png')

    if showPlot:
        plt.show()

    return fig, ax

if __name__ == "__main__":

    # Simulation to compare
    simName = 'nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW_ionDrrOn'

    # Time to plot (in seconds)
    simTime = 3e-3

    plot_fluxloop_comparison(simName, simTime, saveplot=True)
