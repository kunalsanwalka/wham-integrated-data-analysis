"""
This script plots the radial density profiles of the simulations against the diagnostics.

Each column of the plots is a simulated midplane density profile with the SEE synthetic
diagnostic run on it, drawn against the measured SEE signal, and optionally the Thomson
scattering density profile.
"""

import numpy as np
import matplotlib.pyplot as plt

from plotting import save_plot
from cql3d_simulation import load_density_data, parse_simulation_name, density_label
from axuv_extension import build_density_extension, extended_midplane_density
from see_array import (defaultDetectorsToDrop, drop_suffix, load_detector_dictionary,
                       load_experimental_data, time_dependent_see_detector,
                       time_dependent_see_detector_extended, exp_dict_to_numpy, sim_dict_to_numpy)
from thomson import load_thomson_density

def plot_radial_and_synthetic(simulationName, shotnum, timesToPlotSim, timesToPlotExp,
                              useExtension=False, diodeArrayNum=1, rateKHz=10.0,
                              tanhSteepness=4.0, axuvTMin=None, axuvTMax=None,
                              redoAnalysis=False, showPlot=True, detectorsToDrop=None, saveplot=False,
                              axes=None):
    """
    Plot the midplane radial density profile and the synthetic diagnostic for the given times.

    The simulated profile and its synthetic SEE signal are plotted against the measured signal,
    either as CQL3D solves them or extended out to the plasma radius measured by AXUV (see
    time_dependent_see_detector_extended() in see_array.py).

    In the extended case the density is 4D- the extension is shaped by the AXUV plasma radius,
    so both the profile and the signal depend on the experimental time as well as the simulated
    one. timesToPlotSim[i] and timesToPlotExp[i] are therefore taken as a pair, and curve i is
    the simulation at timesToPlotSim[i] extended with the plasma radius measured at
    timesToPlotExp[i], drawn against the experimental signal at timesToPlotExp[i]. The two
    lists have to be the same length in that case. This is the pairing the unextended plot
    already implies by drawing simulated curve i and experimental curve i in the same color.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    shotnum : int
        The shot number the experimental data is taken from.
    timesToPlotSim : np.array
        The simulation times at which to plot the radial profiles. [s]
        The closest available simulated timestep is used.
    timesToPlotExp : np.array
        The experimental times at which to plot the measured signals. [s]
        The closest available experimental timestep is used.
    useExtension : bool
        Use the simulation extended out to the AXUV plasma radius instead of the simulation as
        CQL3D solves it.
        Default is False.
    diodeArrayNum : int
        The diode array the AXUV plasma radius of the extension is taken from, i.e. 1 for
        DIODEARRAY1. Only used when useExtension is True.
        Default is 1.
    rateKHz : float
        Rate the native AXUV data is averaged down to. [kHz]
        This sets the experimental timebase of the extension.
        Only used when useExtension is True.
        Default is 10.
    tanhSteepness : float
        Steepness of the tanh fall-off of the extension.
        Only used when useExtension is True.
        Default is 4.
    axuvTMin : float
        Start of the experimental time window the AXUV data is trimmed to. [s]
        Only used when useExtension is True.
        Default is None, which keeps the data from the start of the shot.
    axuvTMax : float
        End of the experimental time window the AXUV data is trimmed to. [s]
        Only used when useExtension is True.
        Default is None, which keeps the data to the end of the shot.
    redoAnalysis : bool
        Force the extended synthetic diagnostic and the extension itself to be recomputed
        instead of loading the saved ones.
        Only used when useExtension is True.
        Default is False.
    showPlot : bool
        Show the figure before returning. Set to False to add to the figure afterwards.
        Default is True.
    detectorsToDrop : list of int
        Indices (in detDictList) of the detectors with bad data, which are left out of the plot.
        These are 0-indexed, i.e. detector 6 (no signal) is index 5.
        Default is None, which discards defaultDetectorsToDrop (the railed detector, index 15).
    saveplot : bool
        Save the plot in config.plotDir as
        f'{shotnum}_{simulationName}_withThomsonComparison{drop_suffix(detectorsToDrop)}_noThomson.png'.
        Default is False.
    axes : tuple of matplotlib.axes.Axes
        (radial profile axis, synthetic diagnostic axis) to draw into instead of making a new
        figure, e.g. to place this plot in a column of a larger figure.
        Default is None, which makes a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax1 : matplotlib.axes.Axes
        The radial profile axis.
    ax2 : matplotlib.axes.Axes
        The synthetic diagnostic axis.
    """

    if detectorsToDrop is None:
        detectorsToDrop = defaultDetectorsToDrop

    # Each extended curve is a pair of times rather than a simulated time alone
    if useExtension and len(timesToPlotSim) != len(timesToPlotExp):
        raise ValueError('The extended profile depends on the experimental time as well as the '
                         'simulated one, so timesToPlotSim and timesToPlotExp have to be the '
                         f'same length. Got {len(timesToPlotSim)} and {len(timesToPlotExp)}.')

    # Detectors the synthetic diagnostic is run on
    detDictList = load_detector_dictionary()

    if useExtension:

        # Load the extended synthetic SEE signal, [simTime x expTime x detector]
        syntheticSignal, simTimeArr, axuvTimeArr = time_dependent_see_detector_extended(
            simulationName, shotnum, detDictList, diodeArrayNum=diodeArrayNum,
            rateKHz=rateKHz, tanhSteepness=tanhSteepness, axuvTMin=axuvTMin, axuvTMax=axuvTMax,
            redoAnalysis=redoAnalysis)

        # [detector x simTime x expTime], so that it is indexed like the unextended
        # [detector x simTime] below
        simDataArr = np.transpose(syntheticSignal, (2, 0, 1))

        # The pieces the extended density profile is assembled from. This is the same extension
        # the signals above were line integrated through, so the two stay consistent. The full
        # [simTime x expTime x R x Z] profile is not built, as it runs to hundreds of MB and
        # only the midplane of a handful of timesteps is plotted.
        extData = build_density_extension(simulationName, shotnum, diodeArrayNum,
                                          rateKHz=rateKHz, tanhSteepness=tanhSteepness,
                                          tMin=axuvTMin, tMax=axuvTMax,
                                          redoAnalysis=redoAnalysis)

        # Midplane radii of the simulation grid, followed by those of the extension
        radialGrid = np.concatenate([extData['solrz'][:, 0], extData['rExtGrid']])

    else:

        # [Time x r x z], with the CQL3D restart slices dropped and the density clipped. The
        # synthetic signal below has the restart slices dropped as well, so this is what puts
        # dens and simTimeArr on the same timebase.
        dens, solrz, _, _ = load_density_data(simulationName)

        # Load the synthetic SEE signal
        detDictList = time_dependent_see_detector(simulationName, detDictList)
        # Convert to numpy arrays for easy plotting
        simTimeArr, simDataArr = sim_dict_to_numpy(detDictList)

        # Midplane radii of the simulation grid
        radialGrid = solrz[:, 0]

        # There is no AXUV timebase without the extension
        axuvTimeArr = None

    # Load the experimental data
    detDictList = load_experimental_data(shotnum, detectorsToDrop=detectorsToDrop)
    # Convert to numpy arrays for easy plotting
    expTimeArr, expDataArr, expDataSigmaArr, dataPresent = exp_dict_to_numpy(detDictList)

    # Impact Parameters
    impactParams = np.array([detDict['impact_param_vertical'] for detDict in detDictList]) / 1e3

    # Remove data from the broken digitizer and the discarded detectors (see load_experimental_data())
    impactParams = impactParams[dataPresent]
    simDataArr = simDataArr[dataPresent]

    # Sort based on the impact parameter
    sortIdx = np.argsort(impactParams)
    impactParams = impactParams[sortIdx]
    simDataArr = simDataArr[sortIdx]
    expDataArr = expDataArr[sortIdx]
    expDataSigmaArr = expDataSigmaArr[sortIdx]

    # The radial grid reaches past the outermost detector, more so with the extension, so both
    # are taken into account to keep the two panels on the same x axis without clipping either
    xLim = 1.1 * max(np.max(np.abs(impactParams)), np.max(radialGrid))

    if axes is None:

        # Create a figure for the radial profiles
        fig = plt.figure(figsize=(12, 10), tight_layout=True)

        # Synthetic diagnostic on top
        ax2 = fig.add_subplot(2, 1, 1)
        # Radial profile at the midplane on the bottom
        ax1 = fig.add_subplot(2, 1, 2)

    else:

        ax1, ax2 = axes
        fig = ax1.figure

    # Plot the simulated radial profiles and synthetic diagnostics for the specified times
    for i in range(len(timesToPlotSim)):

        # Find the index of the closest time in the simulation data
        timeIdx = np.argmin(np.abs(simTimeArr - timesToPlotSim[i]))

        if useExtension:

            # The extension is shaped by the AXUV plasma radius, so the profile and the signal
            # are both taken at the pair of times (timesToPlotSim[i], timesToPlotExp[i])
            axuvIdx = np.argmin(np.abs(axuvTimeArr - timesToPlotExp[i]))

            _, radialProfile = extended_midplane_density(extData, timeIdx, axuvIdx)

            simSignal = simDataArr[:, timeIdx, axuvIdx]

        else:

            radialProfile = dens[timeIdx, :, 0]

            simSignal = simDataArr[:, timeIdx]

        #### Radial profile at the midplane

        ax1.plot(radialGrid, radialProfile,
                 color=f'C{i}',
                 linewidth=3)
        ax1.plot(-radialGrid, radialProfile,
                 color=f'C{i}',
                 linewidth=3)

        #### Synthetic SEE signal
        ax2.plot(impactParams, simSignal,
                 label=f'Simulation {timesToPlotSim[i]*1e3:.2f}ms',
                 color=f'C{i}',
                 linewidth=3)

    # Plot the experimental data
    for i in range(len(timesToPlotExp)):

        # Find the index of the closest time in the experimental data
        timeIdx = np.argmin(np.abs(expTimeArr - timesToPlotExp[i]))

        ax2.plot(impactParams, expDataArr[:, timeIdx], 
                 color=f'k',
                 linewidth=3, 
                 linestyle='dashed')

        ax2.errorbar(impactParams, expDataArr[:, timeIdx], 
                     yerr=expDataSigmaArr[:, timeIdx],
                     color=f'k',
                     fmt='o',
                     ms=10,
                     elinewidth=7)
        
        ax2.errorbar(impactParams, expDataArr[:, timeIdx], 
                     yerr=3*expDataSigmaArr[:, timeIdx],
                     label=f'SEE Array {timesToPlotExp[i]*1e3:.2f}ms', 
                     color=f'k',
                     fmt='o',
                     ms=10,
                     elinewidth=3)

    ax1.set_ylabel(r'n$_i$ [m$^{-3}$]')
    ax1.set_ylim(0, None)

    ax1.set_xlim(-xLim, xLim)
    ax1.set_xlabel('Radius [m]')

    ax2.set_ylabel(r'$\int n_i \cdot dl$ [m$^{-2}$]')
    ax2.set_ylim(0, None)

    ax2.set_xticks([])
    ax2.set_xlim(-xLim, xLim)

    ax2.set_title(f'Shot #{shotnum}', loc='right')

    # Label the bottom panel with the neutral densities of the simulation
    mainVesselDens, gasBoxDens = parse_simulation_name(simulationName)
    if mainVesselDens is not None:
        densText = (f'Gas Box '+r'$n_n$'+f': {density_label(gasBoxDens)}\n'
                    f'Main Vessel '+r'$n_n$'+f': {density_label(mainVesselDens)}')
        ax1.text(0.98, 0.05, densText,
                 transform=ax1.transAxes,
                 ha='right', va='bottom',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Label the simulated and experimental signals with their times
    ax2.legend(loc='lower left')

    if saveplot:
        save_plot(fig, f'{shotnum}_{simulationName}_withThomsonComparison{drop_suffix(detectorsToDrop)}_noThomson.png')

    if showPlot:
        plt.show()

    return fig, ax1, ax2

def add_thomson_to_axis(ax):
    """
    Add the Thomson scattering density profile (loaded from config.thomsonDataPath) to a radial
    profile axis made by plot_radial_and_synthetic().

    The Thomson data is stored as R = |x|, so it is plotted on the positive radius side only.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The radial profile axis.
    """

    # Load the Thomson scattering data
    tsRadius, tsDens, tsDensSigma, tsTime = load_thomson_density()

    ax.errorbar(tsRadius, tsDens,
                yerr=tsDensSigma,
                label=f'Thomson n$_e$, {tsTime*1e3:.2f} ms',
                color='black',
                fmt='o',
                ms=10,
                elinewidth=3)

    # The y limits were fixed before the Thomson data was added, so rescale to fit its error bars
    ax.set_autoscaley_on(True)
    ax.relim()
    ax.autoscale_view(scalex=False)
    ax.set_ylim(0, None)

    ax.legend(loc='lower left')

    return

def plot_radial_and_synthetic_with_thomson(simulationName, shotnum, timesToPlotSim, timesToPlotExp,
                                           useExtension=False, diodeArrayNum=1, rateKHz=10.0,
                                           tanhSteepness=4.0, axuvTMin=None, axuvTMax=None,
                                           redoAnalysis=False, saveplot=False, detectorsToDrop=None):
    """
    Same plot as plot_radial_and_synthetic(), with the Thomson scattering density profile
    (loaded from config.thomsonDataPath) added to the radial profile panel.

    The Thomson data is stored as R = |x|, so it is plotted on the positive radius side only.

    Parameters
    ----------
    See plot_radial_and_synthetic()
    """

    if detectorsToDrop is None:
        detectorsToDrop = defaultDetectorsToDrop

    fig, ax1, ax2 = plot_radial_and_synthetic(simulationName, shotnum, timesToPlotSim, timesToPlotExp,
                                              useExtension=useExtension, diodeArrayNum=diodeArrayNum,
                                              rateKHz=rateKHz, tanhSteepness=tanhSteepness,
                                              axuvTMin=axuvTMin, axuvTMax=axuvTMax,
                                              redoAnalysis=redoAnalysis, showPlot=False,
                                              detectorsToDrop=detectorsToDrop)

    add_thomson_to_axis(ax1)

    if saveplot:
        save_plot(fig, f'{shotnum}_{simulationName}_withThomsonComparison{drop_suffix(detectorsToDrop)}.png')

    plt.show()

    return

def plot_profile_comparison(simName1, simName2, shotnum1, shotnum2, expTime1, expTime2,
                            simTime1, simTime2, withThomson1=False, withThomson2=False,
                            detectorsToDrop1=None, detectorsToDrop2=None,
                            useExtension=False, diodeArrayNum=1, rateKHz=10.0,
                            tanhSteepness=4.0, redoAnalysis=False, saveplot=False):
    """
    Plot two plot_radial_and_synthetic() figures side by side for comparison.

    The left column is (simName1, shotnum1, simTime1, expTime1) and the right column is
    (simName2, shotnum2, simTime2, expTime2). Each column is the same as the output of
    plot_radial_and_synthetic(), or plot_radial_and_synthetic_with_thomson() if the
    corresponding withThomson flag is set. Both columns share the same x axis and each row
    shares the same y axis, so the two can be compared directly.

    Parameters
    ----------
    simName1, simName2 : str
        The names of the simulations.
    shotnum1, shotnum2 : int
        The shot numbers the experimental data is taken from.
    expTime1, expTime2 : float or np.array
        The experimental times at which to plot the measured signals. [s]
    simTime1, simTime2 : float or np.array
        The simulation times at which to plot the radial profiles. [s]
    withThomson1, withThomson2 : bool
        Add the Thomson scattering density profile to the radial profile panel.
        Default is False.
    detectorsToDrop1, detectorsToDrop2 : list of int
        Indices of the detectors with bad data for each shot.
        Default is None, which discards defaultDetectorsToDrop.
    useExtension, diodeArrayNum, rateKHz, tanhSteepness, redoAnalysis :
        Used for both columns. See plot_radial_and_synthetic().
    saveplot : bool
        Save the plot in config.plotDir as
        f'{shotnum1}_{shotnum2}_profile_comparison.png'.
        Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axs : np.array of matplotlib.axes.Axes
        [row x column] axes. Row 0 is the synthetic diagnostic, row 1 the radial profile.
    """

    fig, axs = plt.subplots(2, 2, figsize=(20, 10), tight_layout=True)

    columns = [(simName1, shotnum1, simTime1, expTime1, withThomson1, detectorsToDrop1),
               (simName2, shotnum2, simTime2, expTime2, withThomson2, detectorsToDrop2)]

    for col, (simName, shotnum, simTime, expTime, withThomson, detectorsToDrop) in enumerate(columns):

        # Synthetic diagnostic on top, radial profile on the bottom
        axRad, axSyn = axs[1, col], axs[0, col]

        plot_radial_and_synthetic(simName, shotnum, np.atleast_1d(simTime), np.atleast_1d(expTime),
                                  useExtension=useExtension, diodeArrayNum=diodeArrayNum,
                                  rateKHz=rateKHz, tanhSteepness=tanhSteepness,
                                  redoAnalysis=redoAnalysis, showPlot=False,
                                  detectorsToDrop=detectorsToDrop, axes=(axRad, axSyn))

        if withThomson:
            add_thomson_to_axis(axRad)

    # Same x axis for all panels, which are symmetric about 0
    xLim = max(np.max(np.abs(ax.get_xlim())) for ax in axs.flat)
    for ax in axs.flat:
        ax.set_xlim(-xLim, xLim)

    # Same y axis along each row
    for row in range(2):
        yMax = max(ax.get_ylim()[1] for ax in axs[row])
        for ax in axs[row]:
            ax.set_ylim(0, yMax)

    # The y labels are shared along each row, so only keep them on the left
    for ax in axs[:, 1]:
        ax.set_ylabel('')

    if saveplot:
        save_plot(fig, f'{shotnum1}_{shotnum2}_profile_comparison.png')

    plt.show()

    return fig, axs

if __name__ == "__main__":

    # Compare the profiles from the pre and post Ti gettering simulations
    simName1 = 'nneut_3e16_gb_1e18_NBI_800kW_ECH_0kW_ionDrrOn'
    simName2 = 'nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW_ionDrrOn'
    shotnum1 = 260709061
    shotnum2 = 261012050
    expTime1 = 7.83e-3
    expTime2 = 7.83e-3
    simTime1 = 3e-3
    simTime2 = 3e-3
    withThomson1 = True
    withThomson2 = False
    detectorsToDrop1 = [15]
    detectorsToDrop2 = [9, 13, 15]
    useExtension = True

    # plot_profile_comparison(simName1, simName2, shotnum1, shotnum2, expTime1, expTime2,
    #                         simTime1, simTime2, withThomson1, withThomson2,
    #                         detectorsToDrop1, detectorsToDrop2,
    #                         useExtension, diodeArrayNum=1, rateKHz=10.0,
    #                         tanhSteepness=4.0, redoAnalysis=False, saveplot=True)

    # A single simulation against a single shot
    plot_radial_and_synthetic(simName2, shotnum2, np.array([simTime2]), np.array([expTime2]),
                              useExtension=True, diodeArrayNum=1, rateKHz=10.0,
                              saveplot=True, detectorsToDrop=detectorsToDrop2)
