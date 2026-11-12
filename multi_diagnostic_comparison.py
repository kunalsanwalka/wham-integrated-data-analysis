"""
This script compares one simulation against several diagnostics of one shot in a single figure.

Only the headline comparison of each diagnostic is shown. The left column has the midplane
diagnostics, one panel each, from top to bottom-
    1. The synthetic SEE signal (line-integrated ion density) against the measured one.
    2. The simulated midplane ion density against the Thomson scattering profile.
    3. The simulated midplane electron temperature against the Thomson scattering profile.
    4. The synthetic H-alpha signal against the one measured by the AXUV array, both normalized
       to their peak.
The right column has the end loss analyzer (ELA) density, electron temperature and ion temperature
mapped to the mirror throat, against the simulated ones there, at the ELA time point closest to the
experimental time. Below them is the measured flux of each flux loop against the synthetic one.
The detailed plots of each diagnostic are made by density_comparison.py, axuv_comparison.py,
electron_temperature_comparison.py, ela_comparison.py and fluxloops_comparison.py.
"""

import numpy as np
import matplotlib.pyplot as plt

from plotting import save_plot
from density_comparison import plot_radial_and_synthetic, add_thomson_to_axis
from electron_temperature_comparison import plot_electron_temperature_comparison
from ela_comparison import plot_ela_comparison
from fluxloops_comparison import plot_fluxloop_comparison
from axuv_comparison import (load_simulated_profiles, halpha_emissivity, synthetic_halpha_signal,
                             add_halpha_signal_to_axis)
from axuv import load_axuv_signal, pinholePositions

def plot_multi_diagnostic_comparison(simulationName, shotnum, simTime, expTime,
                                     elaShotDatePrefix=None, useExtension=False,
                                     detectorsToDrop=None,
                                     radiusDiodeArrayNum=1, signalDiodeArrayNum=2, rateKHz=10.0,
                                     tanhSteepness=4.0, redoAnalysis=False,
                                     showPlot=True, saveplot=False):
    """
    Plot the headline comparison of each diagnostic for one simulation against one shot.

    The four midplane panels in the left column, from top to bottom, are-
        1. The synthetic SEE signal against the measured one (top panel of
           plot_radial_and_synthetic() in density_comparison.py).
        2. The simulated midplane ion density (bottom panel of plot_radial_and_synthetic()),
           against the Thomson scattering density profile.
        3. The simulated midplane electron temperature against the Thomson scattering profile
           (plot_electron_temperature_comparison() in electron_temperature_comparison.py).
        4. The synthetic H-alpha signal against the measured AXUV signal (bottom panel of
           plot_axuv_comparison() in axuv_comparison.py).
    All four are put on the same x axis.

    The right column has the ELA density, electron temperature and ion temperature at the ELA time
    point closest to expTime, mapped to the mirror throat, against the simulated ones there
    (plot_ela_comparison() in ela_comparison.py). These are against the radius at the mirror
    throat, so they have their own x axis. Below them is the measured flux of each flux loop
    against the synthetic one, vs. the Z of each loop (plot_fluxloop_comparison() in
    fluxloops_comparison.py).

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    shotnum : int
        The shot number the experimental data is taken from.
    simTime : float
        The simulation time. [s]
        The closest available simulated timestep is used.
    expTime : float
        The experimental time. [s]
        The ELA panels use the ELA time point closest to it.
    elaShotDatePrefix : int
        Date prefix of the ELA data files in config.elaDataDir, e.g. 260709.
        Default is None, which uses the date of shotnum (its first 6 digits).
    useExtension : bool
        Extend the simulated density out to the AXUV plasma radius for the SEE comparison. See
        plot_radial_and_synthetic(). The H-alpha comparison always uses the CQL3D grid alone.
        Default is False.
    detectorsToDrop : list of int
        Indices of the SEE detectors with bad data. See plot_radial_and_synthetic().
        Default is None, which discards defaultDetectorsToDrop.
    radiusDiodeArrayNum : int
        The diode array the AXUV plasma radius of the extension is taken from.
        Only used when useExtension is True.
        Default is 1.
    signalDiodeArrayNum : int
        The diode array whose line-integrated signal is compared against the synthetic H-alpha
        signal.
        Default is 2, the H-alpha filtered array (DIODEARRAY2).
    rateKHz : float
        Sets the width of the window the AXUV data is averaged over, 1/rateKHz. [kHz]
        Default is 10.
    tanhSteepness : float
        Steepness of the tanh fall-off of the extension.
        Only used when useExtension is True.
        Default is 4.
    redoAnalysis : bool
        Recompute the saved analysis of every comparison (the extension and its synthetic SEE
        signal, the temperature and the KN1D profiles) instead of loading it.
        Default is False.
    showPlot : bool
        Show the figure before returning.
        Default is True.
    saveplot : bool
        Save the plot in config.plotDir as
        f'{shotnum}_{simulationName}_multi_diagnostic_comparison.png'.
        Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axSee : matplotlib.axes.Axes
        The SEE signal axis.
    axDens : matplotlib.axes.Axes
        The density profile axis.
    axTe : matplotlib.axes.Axes
        The electron temperature axis.
    axHalpha : matplotlib.axes.Axes
        The H-alpha signal axis.
    elaAxs : tuple of matplotlib.axes.Axes
        The ELA density, electron temperature and ion temperature axes.
    axFluxLoop : matplotlib.axes.Axes
        The flux loop axis.
    """

    if elaShotDatePrefix is None:
        elaShotDatePrefix = shotnum // 1000

    fig = plt.figure(figsize=(24, 24), tight_layout=True)

    # 4 panels in each column
    grid = fig.add_gridspec(4, 2)

    # The axes are not shared, as plot_radial_and_synthetic() removes the x ticks of the SEE
    # panel, which would remove them from every shared axis. The x limits are matched below.
    axSee = fig.add_subplot(grid[0, 0])
    axDens = fig.add_subplot(grid[1, 0])
    axTe = fig.add_subplot(grid[2, 0])
    axHalpha = fig.add_subplot(grid[3, 0])

    axElaDens = fig.add_subplot(grid[0, 1])
    axElaTe = fig.add_subplot(grid[1, 1], sharex=axElaDens)
    axElaTi = fig.add_subplot(grid[2, 1], sharex=axElaDens)
    elaAxs = (axElaDens, axElaTe, axElaTi)

    # The flux loops are against Z, so they do not share the ELA x axis
    axFluxLoop = fig.add_subplot(grid[3, 1])

    #### SEE and density profile (with Thomson)

    plot_radial_and_synthetic(simulationName, shotnum, np.atleast_1d(simTime),
                              np.atleast_1d(expTime),
                              useExtension=useExtension, diodeArrayNum=radiusDiodeArrayNum,
                              rateKHz=rateKHz, tanhSteepness=tanhSteepness,
                              redoAnalysis=redoAnalysis, showPlot=False,
                              detectorsToDrop=detectorsToDrop, axes=(axDens, axSee))

    add_thomson_to_axis(axDens)

    # The neutral density label fills the lower right, so the legend goes in the empty space
    # above the flat topped profile
    axDens.legend(loc='upper left')

    #### Electron temperature (with Thomson)

    plot_electron_temperature_comparison(simulationName, simTime, redoAnalysis=redoAnalysis,
                                         showPlot=False, axes=axTe)

    # The neutral densities are already labelled on the density panel
    axTe.set_title('')

    #### H-alpha

    profiles = load_simulated_profiles(simulationName, simTime, redoAnalysis=redoAnalysis)
    emissivity = halpha_emissivity(profiles)

    impactParams, signal, signalSigma, description = load_axuv_signal(shotnum, signalDiodeArrayNum,
                                                                      expTime, rateKHz=rateKHz)

    synthSignal = synthetic_halpha_signal(emissivity, impactParams,
                                          pinholePositions[signalDiodeArrayNum])

    add_halpha_signal_to_axis(axHalpha, impactParams, signal, signalSigma, synthSignal,
                              signalDiodeArrayNum, description, expTime)

    # Same x axis for every panel, all of which are symmetric about 0
    xLim = max(np.max(np.abs(ax.get_xlim())) for ax in (axSee, axDens, axTe, axHalpha))
    for ax in (axSee, axDens, axTe, axHalpha):
        ax.set_xlim(-xLim, xLim)

    # Only the bottom panel keeps its x tick labels
    for ax in (axDens, axTe):
        ax.tick_params(labelbottom=False)
        ax.set_xlabel('')

    #### ELA

    plot_ela_comparison(simulationName, elaShotDatePrefix, simTime, expTime=expTime,
                        showPlot=False, axes=elaAxs)

    for ax in (axElaDens, axElaTe):
        ax.tick_params(labelbottom=False)

    #### Flux loops

    plot_fluxloop_comparison(simulationName, simTime, showPlot=False, axes=axFluxLoop)

    # The other panels are not titled with the simulation name either
    axFluxLoop.set_title('Flux loops')

    if saveplot:
        save_plot(fig, f'{shotnum}_{simulationName}_multi_diagnostic_comparison.png')

    if showPlot:
        plt.show()

    return fig, axSee, axDens, axTe, axHalpha, elaAxs, axFluxLoop

if __name__ == "__main__":

    # Simulation and shot to compare
    simName = 'nneut_3e16_gb_1e18_NBI_800kW_ECH_0kW_ionDrrOn'
    shotnum = 260709060

    # Times to plot (in seconds)
    simTime = 3e-3
    expTime = 7.83e-3

    # SEE detectors with bad data in this shot
    detectorsToDrop = [9, 13, 15]

    # Date prefix of the ELA data in config.elaDataDir
    elaShotDatePrefix = 260709

    plot_multi_diagnostic_comparison(simName, shotnum, simTime, expTime,
                                     elaShotDatePrefix=elaShotDatePrefix, useExtension=True,
                                     detectorsToDrop=detectorsToDrop, saveplot=True)
