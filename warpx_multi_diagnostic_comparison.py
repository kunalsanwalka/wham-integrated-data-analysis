"""
This script compares one WarpX simulation against several diagnostics of one shot in a single
figure. It is the WarpX version of ips_multi_diagnostic_comparison.py.

The WarpX fields are averaged over their dumps and theta (see warpx_simulation.py), so there is no
simulation time to choose. The left column has the midplane diagnostics, one panel each, from top
to bottom-
    1. The synthetic SEE signal (line-integrated n_i + n_fast) against the measured one.
    2. The midplane n_e and n_i + n_fast against the Thomson scattering n_e profile.
    3. The midplane electron temperature against the Thomson scattering profile.
    4. The synthetic H-alpha signal against the one measured by the AXUV array, both normalized
       to their peak (see warpx_axuv_comparison.py).
The right column has the end loss analyzer (ELA) density and electron temperature at the z > 0
mirror throat against the WarpX ones there, at the ELA time point closest to the experimental
time. The throat, and the radius of each ELA there, are taken from the eqdsk of an IPS simulation,
as WarpX has no equilibrium. Below them are the SEE and AXUV lines of sight on the WarpX (r, z)
grid (see warpx_lines_of_sight.py).

WarpX gives no ion temperature or equilibrium flux, so the ELA T_i and flux loop panels of
ips_multi_diagnostic_comparison.py are left out.
"""

import numpy as np
import matplotlib.pyplot as plt

from plotting import save_plot, densityColor, temperatureColor
from see_array import (defaultDetectorsToDrop, load_detector_dictionary, load_experimental_data,
                       exp_dict_to_numpy, synthetic_see_detector, synthetic_see_detector_full_los)
from density_comparison import add_thomson_to_axis
from electron_temperature_comparison import add_thomson_temperature_to_axis
from axuv import load_axuv_signal, pinholePositions
from axuv_comparison import add_halpha_signal_to_axis
from ela import load_ela_data, midplane_to_throat_density
from thomson import load_thomson_density
from warpx_simulation import (load_warpx_fields, midplane_profiles, field_interpolator,
                              throat_positions, throat_profiles, profile_at_points,
                              thomsonDensityField, ionDensityField)
from warpx_axuv_comparison import (warpx_halpha_emissivity, synthetic_warpx_halpha_signal,
                                   plasma_extent, warpx_time_label)
from warpx_lines_of_sight import plot_warpx_lines_of_sight

# Color of n_i + n_fast where it is drawn next to n_e
ionDensityColor = 'C9'

def plot_mirrored_line(ax, r, profile, **kwargs):
    """
    Plot a midplane profile mirrored about r = 0, with any line style (plot_mirrored_profile() in
    plotting.py only draws solid lines). Only the r > 0 half is labelled.
    """

    label = kwargs.pop('label', None)

    ax.plot(r, profile, label=label, **kwargs)
    ax.plot(-r, profile, **kwargs)

    return

def warpx_see_signal(shotnum, runFile='fields_rtz.npz', detectorsToDrop=None, fullLos=True):
    """
    The synthetic SEE signal of a WarpX simulation, and the measured one, on the detectors with
    data.

    The synthetic signal line-integrates n_i + n_fast (ionDensityField) along each detector's line
    of sight. The detectors are dropped and sorted the same way as in plot_radial_and_synthetic()
    in density_comparison.py, so the two simulations share the same points.

    Parameters
    ----------
    shotnum : int
        The shot number the experimental data is taken from.
    runFile : str
        The WarpX field dump, see load_warpx_fields().
        Default is 'fields_rtz.npz'.
    detectorsToDrop : list of int
        Indices (in detDictList) of the detectors with bad data, which are left out.
        Default is None, which drops defaultDetectorsToDrop.
    fullLos : bool
        Integrate along the whole line of sight out to the edge of the WarpX grid
        (synthetic_see_detector_full_los()) rather than over 'los_rz' alone.
        Default is True.

    Returns
    -------
    impactParams : np.array
        1D array of the impact parameter of each detector with data, sorted. [m]
    simSignal : np.array
        1D array of the synthetic signal of each detector. [m^-2]
    expTimeArr : np.array
        1D array of the experimental times. [s]
    expDataArr : np.array
        2D array [Detector x Time] of the measured signal. [m^-2]
    expDataSigmaArr : np.array
        2D array [Detector x Time] of the error on the measured signal. [m^-2]
    """

    if detectorsToDrop is None:
        detectorsToDrop = defaultDetectorsToDrop

    fields = load_warpx_fields(runFile)

    #### Synthetic SEE signal, on every detector

    detDictList = load_detector_dictionary()
    interpFunc = field_interpolator(fields, ionDensityField)

    if fullLos:
        simSignal = synthetic_see_detector_full_los(detDictList, interpFunc,
                                                    fields['r'][-1], fields['z'][-1])
    else:
        simSignal = synthetic_see_detector(detDictList, interpFunc)

    #### Measured SEE signal

    expDictList = load_experimental_data(shotnum, detectorsToDrop=detectorsToDrop)
    expTimeArr, expDataArr, expDataSigmaArr, dataPresent = exp_dict_to_numpy(expDictList)

    impactParams = np.array([d['impact_param_vertical'] for d in expDictList]) / 1e3

    # Remove the detectors without data (see load_experimental_data())
    impactParams = impactParams[dataPresent]
    simSignal = simSignal[dataPresent]

    sortIdx = np.argsort(impactParams)

    return (impactParams[sortIdx], simSignal[sortIdx], expTimeArr, expDataArr[sortIdx],
            expDataSigmaArr[sortIdx])

def plot_warpx_see_comparison(shotnum, expTime, runFile='fields_rtz.npz', detectorsToDrop=None,
                              fullLos=True, showPlot=True, saveplot=False, axes=None):
    """
    Plot the WarpX midplane density and its synthetic SEE signal against the measured SEE signal.

    The synthetic signal line-integrates n_i + n_fast (ionDensityField) along each detector's line
    of sight. The density panel shows it and n_e (thomsonDensityField), which is what the Thomson
    scattering data is compared against.

    Parameters
    ----------
    shotnum : int
        The shot number the experimental data is taken from.
    expTime : float
        The experimental time of the measured signal. [s]
        The closest available experimental timestep is used.
    runFile : str
        The WarpX field dump, see load_warpx_fields().
        Default is 'fields_rtz.npz'.
    detectorsToDrop : list of int
        Indices (in detDictList) of the detectors with bad data, which are left out.
        Default is None, which drops defaultDetectorsToDrop.
    fullLos : bool
        Integrate along the whole line of sight out to the edge of the WarpX grid
        (synthetic_see_detector_full_los()). Set to False to integrate over 'los_rz' alone, as
        the CQL3D comparison does. For this run the full line of sight adds 3-5% to the central
        detectors and 9-12% to the outer ones.
        Default is True.
    showPlot : bool
        Show the figure before returning.
        Default is True.
    saveplot : bool
        Save the plot in config.plotDir as f'{shotnum}_warpx_{runName}_see_comparison.png'.
        Default is False.
    axes : tuple of matplotlib.axes.Axes
        (radial profile axis, synthetic diagnostic axis) to draw into instead of making a new
        figure.
        Default is None, which makes a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axDens : matplotlib.axes.Axes
        The radial profile axis.
    axSee : matplotlib.axes.Axes
        The SEE signal axis.
    impactParams : np.array
        The impact parameter of each detector plotted, sorted. [m]
    """

    fields = load_warpx_fields(runFile)
    profiles = midplane_profiles(fields)

    impactParams, simSignal, expTimeArr, expDataArr, expDataSigmaArr = warpx_see_signal(
        shotnum, runFile=runFile, detectorsToDrop=detectorsToDrop, fullLos=fullLos)

    expTimeIdx = np.argmin(np.abs(expTimeArr - expTime))

    if axes is None:
        fig = plt.figure(figsize=(12, 10), tight_layout=True)
        axSee = fig.add_subplot(2, 1, 1)
        axDens = fig.add_subplot(2, 1, 2)
    else:
        axDens, axSee = axes
        fig = axDens.figure

    #### Midplane density

    r = profiles['r']

    plot_mirrored_line(axDens, r, profiles[thomsonDensityField], color=densityColor, linewidth=3,
                       label=r'WarpX n$_e$')
    plot_mirrored_line(axDens, r, profiles[ionDensityField], color=ionDensityColor, linewidth=3,
                       linestyle='dashed', label=r'WarpX n$_i$ + n$_{fast}$')

    axDens.set_ylabel(r'n [m$^{-3}$]')
    axDens.set_ylim(0, None)
    axDens.set_xlabel('Radius [m]')

    #### SEE signal

    axSee.plot(impactParams, simSignal,
               label=r'WarpX $\int$(n$_i$ + n$_{fast}$) dl' + (', full LOS' if fullLos else ''),
               color=densityColor,
               linewidth=3)

    axSee.plot(impactParams, expDataArr[:, expTimeIdx],
               color='k',
               linewidth=3,
               linestyle='dashed')

    axSee.errorbar(impactParams, expDataArr[:, expTimeIdx],
                   yerr=expDataSigmaArr[:, expTimeIdx],
                   color='k',
                   fmt='o',
                   ms=10,
                   elinewidth=7)

    axSee.errorbar(impactParams, expDataArr[:, expTimeIdx],
                   yerr=3*expDataSigmaArr[:, expTimeIdx],
                   label=f'SEE Array {expTimeArr[expTimeIdx]*1e3:.2f}ms',
                   color='k',
                   fmt='o',
                   ms=10,
                   elinewidth=3)

    axSee.set_ylabel(r'$\int n_i \cdot dl$ [m$^{-2}$]')
    axSee.set_ylim(0, None)
    axSee.tick_params(labelbottom=False)
    axSee.set_title(f'Shot #{shotnum}', loc='right')
    axSee.legend(loc='lower left', fontsize='small')

    xLim = 1.1 * max(np.max(np.abs(impactParams)), plasma_extent(profiles[thomsonDensityField], r))
    for ax in (axDens, axSee):
        ax.set_xlim(-xLim, xLim)

    if saveplot:
        save_plot(fig, f'{shotnum}_warpx_{fields["runName"]}_see_comparison.png')

    if showPlot:
        plt.show()

    return fig, axDens, axSee, impactParams

def plot_warpx_ela_comparison(elaShotDatePrefix, expTime, ipsSimulationName, ipsSimTime,
                              runFile='fields_rtz.npz', showPlot=True, saveplot=False, axes=None):
    """
    Plot the ELA density and electron temperature against the WarpX ones at the z > 0 mirror
    throat, at the ELA time point closest to expTime.

    As in plot_ela_comparison() in ela_comparison.py, the ELA density is taken from the midplane
    to the throat with GDT scaling (n_throat = n_mid / sqrt(e)), the temperature is left alone,
    and the radius of each ELA (and the ends of its error bar, drawn horizontally) is followed
    along its flux surface to the throat. WarpX has no
    equilibrium, so the flux surfaces, and the throat, come from the eqdsk of an IPS simulation
    (throat_positions() in warpx_simulation.py). The WarpX n_i + n_fast and T_e are then taken at
    the (r, z) of the throat on each surface.

    Parameters
    ----------
    elaShotDatePrefix : int
        Date prefix of the ELA files, e.g. 260709.
    expTime : float
        The experimental time. [s]
        Only the ELA time point closest to it is drawn.
    ipsSimulationName : str
        The IPS simulation whose eqdsk gives the throat.
    ipsSimTime : float
        The IPS simulation time. [s]
        The eqdsk of the IPS timestep that covers it is used.
    runFile : str
        The WarpX field dump, see load_warpx_fields().
        Default is 'fields_rtz.npz'.
    showPlot : bool
        Show the figure before returning.
        Default is True.
    saveplot : bool
        Save the plot in config.plotDir as
        f'{elaShotDatePrefix}_warpx_{runName}_ela_comparison.png'.
        Default is False.
    axes : tuple of matplotlib.axes.Axes
        (density axis, electron temperature axis) to draw into instead of making a new figure.
        Default is None, which makes a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axs : tuple of matplotlib.axes.Axes
        The density and electron temperature axes.
    """

    fields = load_warpx_fields(runFile)

    #### ELA data at the time point closest to expTime

    ela = load_ela_data(elaShotDatePrefix)
    times, windowHalfWidth = ela['times'], ela['windowHalfWidth']

    timeIdx = np.argmin(np.abs(times - expTime))

    dens = midplane_to_throat_density(ela['dens'][timeIdx])
    densSigma = midplane_to_throat_density(ela['densSigma'][timeIdx])
    Te, TeSigma = ela['Te'][timeIdx], ela['TeSigma'][timeIdx]

    # Throat position of each ELA, and of the ends of its error bar, along the IPS flux surfaces
    radiusMid = ela['radius'][timeIdx]
    radius, zThroat = throat_positions(ipsSimulationName, ipsSimTime, radiusMid)
    radiusLow = throat_positions(ipsSimulationName, ipsSimTime,
                                 radiusMid - ela['radiusErrMinus'][timeIdx])[0]
    radiusHigh = throat_positions(ipsSimulationName, ipsSimTime,
                                  radiusMid + ela['radiusErrPlus'][timeIdx])[0]
    radiusErr = np.array([radius - radiusLow, radiusHigh - radius])

    #### WarpX at the throat

    throat = throat_profiles(fields, ipsSimulationName, ipsSimTime)

    densSimAtEla = profile_at_points(fields, ionDensityField, radius, zThroat)
    TeSimAtEla = profile_at_points(fields, 'T_electrons', radius, zThroat)

    if axes is None:
        fig = plt.figure(figsize=(12, 16), tight_layout=True)
        axDens = fig.add_subplot(2, 1, 1)
        axTe = fig.add_subplot(2, 1, 2, sharex=axDens)
    else:
        axDens, axTe = axes
        fig = axDens.figure

    simLabel = f'WarpX, {warpx_time_label(fields)}'

    for ax, value, sigma, simProfile, simAtEla, color in (
            (axDens, dens, densSigma, throat[ionDensityField], densSimAtEla, densityColor),
            (axTe, Te, TeSigma, throat['T_electrons'], TeSimAtEla, temperatureColor)):

        ax.errorbar(radius, value,
                    xerr=radiusErr,
                    yerr=sigma,
                    label=f'ELA {times[timeIdx]*1e3:g} ms',
                    color='k',
                    fmt='-o',
                    linewidth=2,
                    ms=10,
                    elinewidth=2,
                    capsize=4)

        ax.plot(throat['r'], simProfile,
                color=color,
                linewidth=3,
                label=simLabel)
        ax.plot(radius, simAtEla,
                color=color,
                linestyle='none',
                marker='s',
                ms=12)

        # relim() ignores the error bars, so the top of the axis is set from them, and from the
        # simulated values at the ELAs, with headroom for the legend
        ax.set_ylim(0, 1.3 * max(np.max(value + sigma), np.max(simAtEla)))
        ax.grid(alpha=0.3)

    axDens.set_ylabel(r'n$_{throat}$ [m$^{-3}$]')
    axTe.set_ylabel(r'T$_e$ [eV]')
    axTe.set_xlabel(r'Radius at the mirror throat, r$_{throat}$ [m]')

    # WarpX runs well past the outermost ELA, so the axis is kept on the analyzers
    axDens.set_xlim(0, 1.1 * np.max(radiusHigh))

    axDens.legend(loc='lower center', fontsize='x-small', ncol=2)
    axDens.set_title(f'ELA {elaShotDatePrefix}, '
                     rf'$\pm${windowHalfWidth*1e3:g} ms, '
                     r'n$_{mid}$/$\sqrt{e}$, '
                     rf'z$_{{throat}}$ = {np.mean(zThroat):.3g} m (IPS eqdsk)',
                     loc='right', fontsize='x-small')

    if saveplot:
        save_plot(fig, f'{elaShotDatePrefix}_warpx_{fields["runName"]}_ela_comparison.png')

    if showPlot:
        plt.show()

    return fig, (axDens, axTe)

def plot_warpx_multi_diagnostic_comparison(shotnum, expTime, ipsSimulationName, ipsSimTime,
                                           runFile='fields_rtz.npz', elaShotDatePrefix=None,
                                           detectorsToDrop=None, axuvDetectorsToDrop=None,
                                           fullLos=True, signalDiodeArrayNum=2, rateKHz=10.0,
                                           showPlot=True, saveplot=False):
    """
    Plot the headline comparison of each diagnostic for one WarpX simulation against one shot.

    The four midplane panels in the left column, from top to bottom, are-
        1. The synthetic SEE signal against the measured one (plot_warpx_see_comparison()).
        2. The midplane n_e and n_i + n_fast, against the Thomson scattering n_e profile.
        3. The midplane electron temperature against the Thomson scattering profile.
        4. The synthetic H-alpha signal against the measured AXUV signal (see
           warpx_axuv_comparison.py).
    All four are on the same x axis.

    The right column has the ELA density and electron temperature at the mirror throat against
    the WarpX ones (plot_warpx_ela_comparison()), and below them the SEE and AXUV lines of sight
    on the WarpX (r, z) grid (plot_warpx_lines_of_sight() in warpx_lines_of_sight.py).

    Parameters
    ----------
    shotnum : int
        The shot number the experimental data is taken from.
    expTime : float
        The experimental time. [s]
    ipsSimulationName : str
        The IPS simulation whose eqdsk gives the mirror throat for the ELA comparison.
    ipsSimTime : float
        The IPS simulation time of that eqdsk. [s]
    runFile : str
        The WarpX field dump, see load_warpx_fields().
        Default is 'fields_rtz.npz'.
    elaShotDatePrefix : int
        Date prefix of the ELA files, e.g. 260709.
        Default is None, which takes the date of the shot (its first 6 digits).
    detectorsToDrop : list of int
        Indices of the SEE detectors with bad data, which are left out.
        Default is None, which drops defaultDetectorsToDrop.
    axuvDetectorsToDrop : list of int
        Indices of the AXUV chords with bad data, which are left out. These are in the order the
        chords are stored in the AXUV file, the index on each panel of plot_axuv_raw_signals().
        Default is None, which keeps every chord.
    fullLos : bool
        See plot_warpx_see_comparison().
        Default is True.
    signalDiodeArrayNum : int
        The diode array whose line-integrated signal is compared against the synthetic H-alpha
        signal.
        Default is 2, the H-alpha filtered array (DIODEARRAY2).
    rateKHz : float
        Sets the width of the window the AXUV data is averaged over, 1/rateKHz. [kHz]
        Default is 10.
    showPlot : bool
        Show the figure before returning.
        Default is True.
    saveplot : bool
        Save the plot in config.plotDir as
        f'{shotnum}_warpx_{runName}_multi_diagnostic_comparison.png'.
        Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axSee, axDens, axTe, axHalpha : matplotlib.axes.Axes
        The SEE signal, density profile, electron temperature and H-alpha signal axes.
    elaAxs : tuple of matplotlib.axes.Axes
        The ELA density and electron temperature axes.
    axLos : matplotlib.axes.Axes
        The lines of sight axis.
    """

    if elaShotDatePrefix is None:
        elaShotDatePrefix = shotnum // 1000

    fields = load_warpx_fields(runFile)
    profiles = midplane_profiles(fields)
    r = profiles['r']

    fig = plt.figure(figsize=(24, 24), tight_layout=True)

    grid = fig.add_gridspec(4, 2)

    axSee = fig.add_subplot(grid[0, 0])
    axDens = fig.add_subplot(grid[1, 0])
    axTe = fig.add_subplot(grid[2, 0])
    axHalpha = fig.add_subplot(grid[3, 0])

    axElaDens = fig.add_subplot(grid[0, 1])
    axElaTe = fig.add_subplot(grid[1, 1], sharex=axElaDens)
    elaAxs = (axElaDens, axElaTe)

    axLos = fig.add_subplot(grid[2:, 1])

    #### SEE and density profile (with Thomson)

    _, _, _, seeImpactParams = plot_warpx_see_comparison(shotnum, expTime, runFile=runFile,
                                                         detectorsToDrop=detectorsToDrop,
                                                         fullLos=fullLos, showPlot=False,
                                                         axes=(axDens, axSee))

    add_thomson_to_axis(axDens)
    axDens.legend(loc='upper left', fontsize='small')

    #### Electron temperature (with Thomson)

    plot_mirrored_line(axTe, r, profiles['T_electrons'], color=temperatureColor, linewidth=3,
                       label=f'WarpX, {warpx_time_label(fields)}')

    axTe.set_ylim(0, 1.25 * np.max(profiles['T_electrons']))
    add_thomson_temperature_to_axis(axTe)

    axTe.set_ylabel(r'T$_e$ [eV]')
    axTe.legend(loc='upper left', fontsize='small')

    #### H-alpha

    emissivity = warpx_halpha_emissivity(profiles)

    impactParams, signal, signalSigma, description = load_axuv_signal(shotnum, signalDiodeArrayNum,
                                                                      expTime, rateKHz=rateKHz,
                                                                      detectorsToDrop=axuvDetectorsToDrop)

    synthSignal = synthetic_warpx_halpha_signal(emissivity, impactParams,
                                                pinholePositions[signalDiodeArrayNum])

    add_halpha_signal_to_axis(axHalpha, impactParams, signal, signalSigma, synthSignal,
                              signalDiodeArrayNum, description, expTime)

    # Same x axis for every panel. The WarpX grid runs out to the wall, so the axis stops where
    # the plasma does, or at the outermost measurement.
    tsRadius = load_thomson_density()[0]
    xLim = 1.1 * max(np.max(np.abs(seeImpactParams)), np.max(np.abs(impactParams)),
                     np.max(tsRadius), plasma_extent(profiles[thomsonDensityField], r))
    for ax in (axSee, axDens, axTe, axHalpha):
        ax.set_xlim(-xLim, xLim)

    # Only the bottom panel keeps its x tick labels
    for ax in (axDens, axTe):
        ax.tick_params(labelbottom=False)
        ax.set_xlabel('')

    #### ELA

    plot_warpx_ela_comparison(elaShotDatePrefix, expTime, ipsSimulationName, ipsSimTime,
                              runFile=runFile, showPlot=False, axes=elaAxs)

    axElaDens.tick_params(labelbottom=False)

    #### Lines of sight

    plot_warpx_lines_of_sight(shotnum, runFile=runFile, signalDiodeArrayNum=signalDiodeArrayNum,
                              showPlot=False, ax=axLos)

    axSee.set_title(f'WarpX {fields["runName"]}, Shot #{shotnum}', loc='right', fontsize='small')
    axLos.set_title('Lines of sight', loc='right', fontsize='small')

    if saveplot:
        save_plot(fig, f'{shotnum}_warpx_{fields["runName"]}_multi_diagnostic_comparison.png')

    if showPlot:
        plt.show()

    return fig, axSee, axDens, axTe, axHalpha, elaAxs, axLos

if __name__ == "__main__":

    # WarpX run and shot to compare
    runFile = 'fields_rtz.npz'
    shotnum = 260709060

    # Experimental time (in seconds). The WarpX fields are averaged over their dumps.
    expTime = 7.83e-3

    # SEE detectors with bad data in this shot
    detectorsToDrop = [9, 13, 15]

    # AXUV chords with bad data in this shot (chord indices, see plot_axuv_raw_signals() in
    # axuv_comparison.py)
    axuvDetectorsToDrop = [8, 9, 10]

    # IPS simulation whose eqdsk gives the mirror throat for the ELA comparison
    ipsSimulationName = 'nneut_3e16_gb_1e18_NBI_800kW_ECH_0kW_ionDrrOn'
    ipsSimTime = 3e-3

    # Date prefix of the ELA data in config.elaDataDir
    elaShotDatePrefix = 260709

    plot_warpx_multi_diagnostic_comparison(shotnum, expTime, ipsSimulationName, ipsSimTime,
                                           runFile=runFile, elaShotDatePrefix=elaShotDatePrefix,
                                           detectorsToDrop=detectorsToDrop,
                                           axuvDetectorsToDrop=axuvDetectorsToDrop, saveplot=True)
