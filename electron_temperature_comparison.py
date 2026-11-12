"""
This script compares the simulated midplane electron temperature against the Thomson scattering
temperature profile.

The Thomson data is a single file (config.thomsonDataPath) rather than one per shot, so only the
simulation is chosen here, with no shot number.
"""

import numpy as np
import matplotlib.pyplot as plt

from plotting import save_plot, plot_mirrored_profile, temperatureColor
from cql3d_simulation import load_temperature_data, parse_simulation_name, density_label
from thomson import load_thomson_temperature

def add_thomson_temperature_to_axis(ax):
    """
    Add the Thomson scattering electron temperature profile (loaded from config.thomsonDataPath)
    to a midplane radial profile axis.

    The Thomson data is stored as R = |x|, so it is plotted on the positive radius side only.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The radial profile axis. Its y axis is in eV.
    """

    # Load the Thomson scattering data
    tsRadius, tsTe, tsTeSigma, tsTime = load_thomson_temperature()

    ax.errorbar(tsRadius, tsTe,
                yerr=tsTeSigma,
                label=f'Thomson T$_e$, {tsTime*1e3:.2f} ms',
                color='black',
                fmt='o',
                ms=10,
                elinewidth=3)

    # relim() ignores the error bars, so the top of the axis is set from them directly, with
    # headroom for the legend
    yMax = max(ax.get_ylim()[1], np.max(tsTe + tsTeSigma))
    ax.set_ylim(0, 1.25 * yMax)

    return

def plot_electron_temperature_comparison(simulationName, simTime, redoAnalysis=False,
                                         showPlot=True, saveplot=False, axes=None,
                                         color=temperatureColor):
    """
    Plot the simulated midplane electron temperature against the Thomson scattering profile.

    The simulated T_e is (2/3)<E> of the electron distribution function at the midplane (see
    load_temperature_data() in cql3d_simulation.py), on the CQL3D grid alone. It is mirrored
    about r = 0 so the panel can share an x axis with the line-integrated diagnostics.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    simTime : float
        The simulation time. [s]
        The closest available simulated timestep is used.
    redoAnalysis : bool
        Force the temperature profiles to be reloaded from the WHAM.nc files instead of the cache.
        Default is False.
    showPlot : bool
        Show the figure before returning. Set to False to add to the figure afterwards.
        Default is True.
    saveplot : bool
        Save the plot in config.plotDir as
        f'{simulationName}_electron_temperature_comparison.png'.
        Default is False.
    axes : matplotlib.axes.Axes
        Axis to draw into instead of making a new figure, e.g. to place this plot in a larger
        figure.
        Default is None, which makes a new figure.
    color : str
        Color of the simulated profile.
        Default is temperatureColor.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axTe : matplotlib.axes.Axes
        The electron temperature axis.
    """

    #### Simulated electron temperature

    Te, _, rMid, teTimes = load_temperature_data(simulationName, redoAnalysis=redoAnalysis)

    teTimeIdx = np.argmin(np.abs(teTimes - simTime))

    if axes is None:
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        axTe = fig.add_subplot(1, 1, 1)
    else:
        axTe = axes
        fig = axTe.figure

    # [keV] to [eV]
    plot_mirrored_profile(axTe, rMid, Te[teTimeIdx]*1e3, color=color,
                          label=rf'CQL3D, t$_{{Sim}}$ = {teTimes[teTimeIdx]*1e3:.4g} ms')

    #### Thomson scattering

    add_thomson_temperature_to_axis(axTe)

    axTe.set_xlabel('Radius [m]')
    axTe.set_ylabel(r'T$_e$ [eV]')

    # Symmetric about 0, covering both the simulation grid and the Thomson points
    xLim = np.max(np.abs(axTe.get_xlim()))
    axTe.set_xlim(-xLim, xLim)

    # Label the panel with the neutral densities of the simulation
    mainVesselDens, gasBoxDens = parse_simulation_name(simulationName)
    if mainVesselDens is not None:
        axTe.set_title(f'Gas Box '+r'$n_n$'+f': {density_label(gasBoxDens)}, '
                       f'Main Vessel '+r'$n_n$'+f': {density_label(mainVesselDens)}',
                       fontsize='small')

    axTe.legend(loc='upper left', fontsize='small')

    if saveplot:
        save_plot(fig, f'{simulationName}_electron_temperature_comparison.png')

    if showPlot:
        plt.show()

    return fig, axTe

if __name__ == "__main__":

    # Simulation to compare
    simName = 'nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW_ionDrrOn'

    # Time to plot (in seconds)
    simTime = 3e-3

    plot_electron_temperature_comparison(simName, simTime, saveplot=True)
