"""
This script compares the end loss analyzer (ELA) density, electron temperature and ion
temperature profiles against the simulation.

The simulated density, electron temperature and ion temperature are taken at the mirror throat and
mapped out along the flux tubes to the ELAs in the expander (see map_throat_to_ela() in ela.py).
The temperatures are taken to not change between the throat and the ELAs, so only their radius is
mapped.
"""

import numpy as np
import matplotlib.pyplot as plt

from plotting import save_plot
from cql3d_simulation import load_throat_density, load_throat_temperature
from ela import load_ela_density, load_ela_temperature, map_throat_to_ela, elaMirrorRatio

def plot_ela_comparison(simulationName, shotDatePrefix, simTime, expTime=None,
                        mirrorRatio=elaMirrorRatio, showPlot=True, saveplot=False, axes=None):
    """
    Plot the ELA density, electron temperature and ion temperature against the radius of each
    analyzer, one curve per time point, with the simulated density, electron temperature and ion
    temperature at the ELAs.

    The time points are colored from light (earliest) to dark (latest). If expTime is given, only
    the time point closest to it is drawn, in black. The simulated profiles are
    drawn over their whole radial extent, with a marker at the radius of each ELA. ELAs inside the
    innermost simulated flux tube take its value.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    shotDatePrefix : int
        Date prefix of the shots, e.g. 260709.
    simTime : float
        The simulation time. [s]
        The closest available simulated timestep is used.
    expTime : float
        The experimental time. [s]
        Only the ELA time point closest to it is drawn.
        Default is None, which draws every time point.
    mirrorRatio : float
        B_throat / B_ELA, used to map the simulated profiles from the throat to the ELAs.
        Default is elaMirrorRatio.
    showPlot : bool
        Show the figure before returning. Set to False to add to the figure afterwards.
        Default is True.
    saveplot : bool
        Save the plot in config.plotDir as f'{shotDatePrefix}_{simulationName}_ela_comparison.png'.
        Default is False.
    axes : tuple of matplotlib.axes.Axes
        The three axes (density, T_e, T_i) to draw into instead of making a new figure, e.g. to
        place this plot in a larger figure.
        Default is None, which makes a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axs : tuple of matplotlib.axes.Axes
        The density, electron temperature and ion temperature axes.
    """

    #### ELA data

    times, radius, analyzerNum, dens, densSigma, windowHalfWidth = load_ela_density(shotDatePrefix)
    tempTimes, tempRadius, _, Te, TeSigma, Ti, TiSigma, _ = load_ela_temperature(shotDatePrefix)

    if not (np.allclose(times, tempTimes) and np.allclose(radius, tempRadius)):
        raise ValueError(f'The density and temperature files of {shotDatePrefix} are on '
                         'different times or radii')

    # One shade of a single hue per time, light to dark
    colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(times)))

    if expTime is not None:
        # Kept as a list so every [time, analyzer] array stays 2D
        expTimeIdx = [np.argmin(np.abs(times - expTime))]
        times, radius, analyzerNum = times[expTimeIdx], radius[expTimeIdx], analyzerNum[expTimeIdx]
        dens, densSigma = dens[expTimeIdx], densSigma[expTimeIdx]
        Te, TeSigma = Te[expTimeIdx], TeSigma[expTimeIdx]
        Ti, TiSigma = Ti[expTimeIdx], TiSigma[expTimeIdx]
        colors = ['k']

    #### Simulated density at the ELAs

    densThroat, rThroat, _, simTimes = load_throat_density(simulationName)

    simTimeIdx = np.argmin(np.abs(simTimes - simTime))

    rEla, densEla = map_throat_to_ela(rThroat, densThroat[simTimeIdx], mirrorRatio)

    # The analyzers do not move, so the radius at the first time is used
    densSimAtEla = np.interp(radius[0], rEla, densEla)

    #### Simulated temperatures at the ELAs

    TeThroat, TiThroat, rThroatTemp, tempSimTimes = load_throat_temperature(simulationName)

    tempSimTimeIdx = np.argmin(np.abs(tempSimTimes - simTime))

    # The temperature does not change along the flux tube, so only the radius is mapped
    rElaTemp = rThroatTemp * np.sqrt(mirrorRatio)
    # [keV] to [eV]
    TeSim = TeThroat[tempSimTimeIdx] * 1e3
    TiSim = TiThroat[tempSimTimeIdx] * 1e3

    TeSimAtEla = np.interp(radius[0], rElaTemp, TeSim)
    TiSimAtEla = np.interp(radius[0], rElaTemp, TiSim)

    if axes is None:
        fig = plt.figure(figsize=(12, 24), tight_layout=True)
        axDens = fig.add_subplot(3, 1, 1)
        axTe = fig.add_subplot(3, 1, 2, sharex=axDens)
        axTi = fig.add_subplot(3, 1, 3, sharex=axDens)
    else:
        axDens, axTe, axTi = axes
        fig = axDens.figure

    panels = ((axDens, dens, densSigma),
              (axTe, Te, TeSigma),
              (axTi, Ti, TiSigma))

    for timeIdx in range(len(times)):
        for ax, value, sigma in panels:
            ax.errorbar(radius[timeIdx], value[timeIdx],
                        yerr=sigma[timeIdx],
                        label=f'{times[timeIdx]*1e3:g} ms',
                        color=colors[timeIdx],
                        fmt='-o',
                        linewidth=2,
                        ms=10,
                        elinewidth=2,
                        capsize=4)

    axDens.plot(rEla, densEla,
                color='C1',
                linewidth=3,
                label=rf'CQL3D, t$_{{Sim}}$ = {simTimes[simTimeIdx]*1e3:.4g} ms')
    axDens.plot(radius[0], densSimAtEla,
                color='C1',
                linestyle='none',
                marker='s',
                ms=12)

    axTe.plot(rElaTemp, TeSim,
              color='C1',
              linewidth=3,
              label=rf'CQL3D, t$_{{Sim}}$ = {tempSimTimes[tempSimTimeIdx]*1e3:.4g} ms')
    axTe.plot(radius[0], TeSimAtEla,
              color='C1',
              linestyle='none',
              marker='s',
              ms=12)

    axTi.plot(rElaTemp, TiSim,
              color='C1',
              linewidth=3,
              label=rf'CQL3D, t$_{{Sim}}$ = {tempSimTimes[tempSimTimeIdx]*1e3:.4g} ms')
    axTi.plot(radius[0], TiSimAtEla,
              color='C1',
              linestyle='none',
              marker='s',
              ms=12)

    # Label each point with its analyzer number, above the error bar of the densest time
    for r, num, top in zip(radius[0], analyzerNum[0], np.max(dens + densSigma, axis=0)):
        axDens.annotate(f'ELA {num}', (r, top), xytext=(0, 6), textcoords='offset points',
                        ha='center', fontsize='x-small')

    axDens.set_ylabel(r'n [m$^{-3}$]')
    axTe.set_ylabel(r'T$_e$ [eV]')
    axTi.set_ylabel(r'T$_i$ [eV]')
    axTi.set_xlabel('Radius [m]')

    for ax, value, sigma in panels:
        # relim() ignores the error bars, so the top of the axis is set from them directly, with
        # headroom for the legend
        ax.set_ylim(0, 1.3 * np.max(value + sigma))
        ax.grid(alpha=0.3)

    # The simulated temperatures can be above the measured ones, so the axes are stretched to fit
    # them
    for ax, value, sigma, simAtEla in ((axTe, Te, TeSigma, TeSimAtEla),
                                       (axTi, Ti, TiSigma, TiSimAtEla)):
        ax.set_ylim(0, 1.3 * max(np.max(value + sigma), np.max(simAtEla)))

    # The simulation runs well past the outermost ELA, so the axis is kept on the analyzers
    axDens.set_xlim(0, 1.1 * np.max(radius))

    axDens.legend(loc='lower center', fontsize='x-small', ncol=3)
    axDens.set_title(f'ELA, {shotDatePrefix}, '
                     rf'$\pm${windowHalfWidth*1e3:g} ms window, '
                     rf'B$_{{throat}}$/B$_{{ELA}}$ = {mirrorRatio:g}', fontsize='small')

    if saveplot:
        save_plot(fig, f'{shotDatePrefix}_{simulationName}_ela_comparison.png')

    if showPlot:
        plt.show()

    return fig, (axDens, axTe, axTi)

if __name__ == "__main__":

    # Simulation to compare
    simName = 'nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW_ionDrrOn'

    # Date prefix of the shots in config.elaDataDir
    shotDatePrefix = 260709

    # Time to plot (in seconds)
    simTime = 3e-3

    plot_ela_comparison(simName, shotDatePrefix, simTime, saveplot=True)
