"""
This script compares the end loss analyzer (ELA) density, electron temperature and ion
temperature profiles against the simulation at the mirror throat, where the simulated plasma is
Maxwellian.

The ELA density and radii are given at the midplane, mapped there with GDT scaling. They are taken
on to the mirror throat-
    - The density is multiplied by 1/sqrt(e), the GDT drop from the midplane to the throat
      (midplane_to_throat_density() in ela.py).
    - The radius of each analyzer is followed along its flux surface in the eqdsk of the IPS
      timestep that covers simTime, out to where the surface is closest to the axis
      (map_midplane_radius_to_throat() in cql3d_simulation.py).
    - The temperatures are left alone, since GDT takes the plasma to be isothermal along the flux
      tube.
Everything is plotted against the radius at the mirror throat.
"""

import numpy as np
import matplotlib.pyplot as plt

from plotting import save_plot, densityColor, temperatureColor
from cql3d_simulation import (load_throat_density, load_throat_temperature, load_eqdsk_flux,
                              ips_timestep_index, map_midplane_radius_to_throat)
from ela import load_ela_density, load_ela_temperature, midplane_to_throat_density

def plot_ela_comparison(simulationName, shotDatePrefix, simTime, expTime=None,
                        showPlot=True, saveplot=False, axes=None):
    """
    Plot the ELA density, electron temperature and ion temperature against the radius of each
    analyzer at the mirror throat, one curve per time point, with the simulated density, electron
    temperature and ion temperature at the mirror throat.

    The ELA density and radii are at the midplane in the data files. The density is mapped to the
    throat with GDT scaling (n_throat = n_mid / sqrt(e)), and the radii along the flux surfaces of
    the eqdsk of the IPS timestep that covers simTime. The temperatures are not changed.

    The time points are colored from light (earliest) to dark (latest). If expTime is given, only
    the time point closest to it is drawn, in black. The simulated profiles are drawn over their
    whole radial extent, with a marker at the throat radius of each ELA. ELAs inside the innermost
    simulated flux tube take its value.

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

    times, radiusMid, analyzerNum, densMid, densMidSigma, windowHalfWidth = load_ela_density(shotDatePrefix)
    tempTimes, tempRadius, _, Te, TeSigma, Ti, TiSigma, _ = load_ela_temperature(shotDatePrefix)

    if not (np.allclose(times, tempTimes) and np.allclose(radiusMid, tempRadius)):
        raise ValueError(f'The density and temperature files of {shotDatePrefix} are on '
                         'different times or radii')

    # One shade of grey per time, light to dark, so the measurements stay black-ish like those of
    # the other diagnostics and do not clash with the simulated density
    colors = plt.cm.Greys(np.linspace(0.4, 1.0, len(times)))

    if expTime is not None:
        # Kept as a list so every [time, analyzer] array stays 2D
        expTimeIdx = [np.argmin(np.abs(times - expTime))]
        times, radiusMid, analyzerNum = times[expTimeIdx], radiusMid[expTimeIdx], analyzerNum[expTimeIdx]
        densMid, densMidSigma = densMid[expTimeIdx], densMidSigma[expTimeIdx]
        Te, TeSigma = Te[expTimeIdx], TeSigma[expTimeIdx]
        Ti, TiSigma = Ti[expTimeIdx], TiSigma[expTimeIdx]
        colors = ['k']

    #### ELA data at the mirror throat

    # GDT scaling of the density from the midplane to the throat
    dens = midplane_to_throat_density(densMid)
    densSigma = midplane_to_throat_density(densMidSigma)

    # Radius of each analyzer at the throat, along the flux surfaces of the eqdsk that the
    # simulation used at simTime
    timestepIdx = ips_timestep_index(simulationName, simTime)
    Rmesh, Zmesh, psi = load_eqdsk_flux(simulationName, timestepIdx)
    radius = map_midplane_radius_to_throat(radiusMid.ravel(), Rmesh, Zmesh, psi)[0].reshape(radiusMid.shape)

    #### Simulated density at the mirror throat

    densThroat, rThroat, _, simTimes = load_throat_density(simulationName)

    simTimeIdx = np.argmin(np.abs(simTimes - simTime))
    densSim = densThroat[simTimeIdx]

    # The analyzers do not move, so the radius at the first time is used
    densSimAtEla = np.interp(radius[0], rThroat, densSim)

    #### Simulated temperatures at the mirror throat

    TeThroat, TiThroat, rThroatTemp, tempSimTimes = load_throat_temperature(simulationName)

    tempSimTimeIdx = np.argmin(np.abs(tempSimTimes - simTime))

    # [keV] to [eV]
    TeSim = TeThroat[tempSimTimeIdx] * 1e3
    TiSim = TiThroat[tempSimTimeIdx] * 1e3

    TeSimAtEla = np.interp(radius[0], rThroatTemp, TeSim)
    TiSimAtEla = np.interp(radius[0], rThroatTemp, TiSim)

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

    axDens.plot(rThroat, densSim,
                color=densityColor,
                linewidth=3,
                label=rf'CQL3D, t$_{{Sim}}$ = {simTimes[simTimeIdx]*1e3:.4g} ms')
    axDens.plot(radius[0], densSimAtEla,
                color=densityColor,
                linestyle='none',
                marker='s',
                ms=12)

    axTe.plot(rThroatTemp, TeSim,
              color=temperatureColor,
              linewidth=3,
              label=rf'CQL3D, t$_{{Sim}}$ = {tempSimTimes[tempSimTimeIdx]*1e3:.4g} ms')
    axTe.plot(radius[0], TeSimAtEla,
              color=temperatureColor,
              linestyle='none',
              marker='s',
              ms=12)

    axTi.plot(rThroatTemp, TiSim,
              color=temperatureColor,
              linewidth=3,
              label=rf'CQL3D, t$_{{Sim}}$ = {tempSimTimes[tempSimTimeIdx]*1e3:.4g} ms')
    axTi.plot(radius[0], TiSimAtEla,
              color=temperatureColor,
              linestyle='none',
              marker='s',
              ms=12)

    # Label each point with its analyzer number, above the error bar of the densest time
    for r, num, top in zip(radius[0], analyzerNum[0], np.max(dens + densSigma, axis=0)):
        axDens.annotate(f'ELA {num}', (r, top), xytext=(0, 6), textcoords='offset points',
                        ha='center', fontsize='x-small')

    axDens.set_ylabel(r'n$_{throat}$ [m$^{-3}$]')
    axTe.set_ylabel(r'T$_e$ [eV]')
    axTi.set_ylabel(r'T$_i$ [eV]')
    axTi.set_xlabel(r'Radius at the mirror throat, r$_{throat}$ [m]')

    for ax, value, sigma in panels:
        # relim() ignores the error bars, so the top of the axis is set from them directly, with
        # headroom for the legend
        ax.set_ylim(0, 1.3 * np.max(value + sigma))
        ax.grid(alpha=0.3)

    # The simulated profiles can be above the measured ones, so the axes are stretched to fit them
    for ax, value, sigma, simAtEla in ((axDens, dens, densSigma, densSimAtEla),
                                       (axTe, Te, TeSigma, TeSimAtEla),
                                       (axTi, Ti, TiSigma, TiSimAtEla)):
        ax.set_ylim(0, 1.3 * max(np.max(value + sigma), np.max(simAtEla)))

    # The simulation runs well past the outermost ELA, so the axis is kept on the analyzers
    axDens.set_xlim(0, 1.1 * np.max(radius))

    axDens.legend(loc='lower center', fontsize='x-small', ncol=3)
    axDens.set_title(f'ELA, {shotDatePrefix}, '
                     rf'$\pm${windowHalfWidth*1e3:g} ms window, '
                     r'n$_{throat}$ = n$_{mid}$/$\sqrt{e}$ (GDT)', fontsize='small')

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
