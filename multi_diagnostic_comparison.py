"""
This script compares one shot against an IPS (CQL3D + KN1D) simulation and a WarpX simulation, with
the headline comparison of each diagnostic in a single figure.

The layout is that of ips_multi_diagnostic_comparison.py. The left column has the midplane
diagnostics, one panel each, from top to bottom-
    1. The synthetic SEE signals against the measured one. IPS integrates n_i, WarpX n_i + n_fast.
    2. The simulated midplane density against the Thomson scattering profile. IPS shows n_i
       (n_e = n_i), WarpX n_e.
    3. The simulated midplane electron temperature against the Thomson scattering profile.
    4. The synthetic H-alpha signals against the one measured by the AXUV array, each normalized
       to its peak.
The right column has the end loss analyzer (ELA) density (IPS n_i, WarpX n_i + n_fast), electron
temperature and ion temperature at the mirror throat, at the ELA time point closest to the
experimental time, and below them the flux loops. WarpX gives no ion temperature or equilibrium
flux, so the ELA T_i and flux loop panels have the IPS simulation alone.

Each source keeps one color in every panel: the measurements are black with error bars, the IPS
simulation is ipsColor (blue) and the WarpX simulation warpxColor (red). The three are labelled
once, in the SEE panel.

WarpX has no equilibrium, so the mirror throat and the ELA radii there come from the eqdsk of the
IPS simulation at simTime, as in the IPS ELA panel. The WarpX fields are averaged over their dumps
and theta (see warpx_simulation.py), so simTime only applies to the IPS simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plotting import save_plot, ipsColor, warpxColor
from density_comparison import plot_radial_and_synthetic, add_thomson_to_axis
from electron_temperature_comparison import plot_electron_temperature_comparison
from ela_comparison import plot_ela_comparison
from fluxloops_comparison import plot_fluxloop_comparison
from axuv_comparison import (load_simulated_profiles, halpha_emissivity, synthetic_halpha_signal,
                             add_halpha_signal_to_axis)
from axuv import load_axuv_signal, pinholePositions
from thomson import load_thomson_density, load_thomson_temperature
from ela import load_ela_density
from warpx_simulation import (load_warpx_fields, midplane_profiles, throat_positions,
                              throat_profiles, profile_at_points, thomsonDensityField,
                              ionDensityField)
from warpx_axuv_comparison import (warpx_halpha_emissivity, synthetic_warpx_halpha_signal,
                                   plasma_extent, warpx_time_label)
from warpx_multi_diagnostic_comparison import warpx_see_signal, plot_mirrored_line

def plot_multi_diagnostic_comparison(simulationName, shotnum, simTime, expTime,
                                     elaShotDatePrefix=None, useExtension=False,
                                     detectorsToDrop=None, axuvDetectorsToDrop=None,
                                     radiusDiodeArrayNum=1, signalDiodeArrayNum=2, rateKHz=10.0,
                                     tanhSteepness=4.0, redoAnalysis=False,
                                     warpxRunFile='fields_rtz.npz', fullLos=True,
                                     showPlot=True, saveplot=False):
    """
    Plot the headline comparison of each diagnostic for one shot against an IPS and a WarpX
    simulation.

    The IPS panels are those of plot_ips_multi_diagnostic_comparison() in
    ips_multi_diagnostic_comparison.py, drawn by the same functions, with the WarpX simulation
    added to every panel it has data for (all but the ELA T_i and the flux loops).

    Parameters
    ----------
    simulationName : str
        The name of the IPS simulation. Its eqdsk at simTime also gives the mirror throat of the
        WarpX ELA comparison.
    shotnum : int
        The shot number the experimental data is taken from.
    simTime : float
        The IPS simulation time. [s]
        The closest available simulated timestep is used.
    expTime : float
        The experimental time. [s]
    elaShotDatePrefix : int
        Date prefix of the ELA files, e.g. 260709.
        Default is None, which takes the date of the shot (its first 6 digits).
    useExtension : bool
        Use the IPS simulation extended out to the AXUV plasma radius in the SEE and density
        panels. See plot_radial_and_synthetic(). The WarpX grid reaches the wall, so it needs none.
        Default is False.
    detectorsToDrop : list of int
        Indices of the SEE detectors with bad data. See plot_radial_and_synthetic().
        Default is None, which discards defaultDetectorsToDrop.
    axuvDetectorsToDrop : list of int
        Indices of the AXUV chords with bad data, which are left out. These are in the order the
        chords are stored in the AXUV file, the index on each panel of plot_axuv_raw_signals().
        Default is None, which keeps every chord.
    radiusDiodeArrayNum : int
        The diode array the AXUV plasma radius of the extension is taken from.
        Only used when useExtension is True.
        Default is 1.
    signalDiodeArrayNum : int
        The diode array whose line-integrated signal is compared against the synthetic H-alpha
        signals.
        Default is 2, the H-alpha filtered array (DIODEARRAY2).
    rateKHz : float
        Sets the width of the window the AXUV data is averaged over, 1/rateKHz. [kHz]
        Default is 10.
    tanhSteepness : float
        Steepness of the tanh fall-off of the extension.
        Only used when useExtension is True.
        Default is 4.
    redoAnalysis : bool
        Recompute the saved analysis of every IPS comparison instead of loading it.
        Default is False.
    warpxRunFile : str
        The WarpX field dump, see load_warpx_fields() in warpx_simulation.py.
        Default is 'fields_rtz.npz'.
    fullLos : bool
        Integrate the WarpX SEE signal along the whole line of sight out to the edge of the WarpX
        grid. See warpx_see_signal() in warpx_multi_diagnostic_comparison.py.
        Default is True.
    showPlot : bool
        Show the figure before returning.
        Default is True.
    saveplot : bool
        Save the plot in config.plotDir as
        f'{shotnum}_{simulationName}_warpx_{runName}_multi_diagnostic_comparison.png'.
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

    fields = load_warpx_fields(warpxRunFile)
    warpxProfiles = midplane_profiles(fields)
    rWarpx = warpxProfiles['r']

    fig = plt.figure(figsize=(24, 24), tight_layout=True)

    # 4 panels in each column, as in ips_multi_diagnostic_comparison.py
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

    axFluxLoop = fig.add_subplot(grid[3, 1])

    midplaneAxs = (axSee, axDens, axTe, axHalpha)

    #### IPS, drawn by the IPS comparison functions in ipsColor

    # plot_radial_and_synthetic() draws its first simulation in 'C0', which is ipsColor
    plot_radial_and_synthetic(simulationName, shotnum, np.atleast_1d(simTime),
                              np.atleast_1d(expTime),
                              useExtension=useExtension, diodeArrayNum=radiusDiodeArrayNum,
                              rateKHz=rateKHz, tanhSteepness=tanhSteepness,
                              redoAnalysis=redoAnalysis, showPlot=False,
                              detectorsToDrop=detectorsToDrop, axes=(axDens, axSee))

    add_thomson_to_axis(axDens)

    plot_electron_temperature_comparison(simulationName, simTime, redoAnalysis=redoAnalysis,
                                         showPlot=False, axes=axTe, color=ipsColor)

    ipsProfiles = load_simulated_profiles(simulationName, simTime, redoAnalysis=redoAnalysis)
    ipsEmissivity = halpha_emissivity(ipsProfiles)

    impactParams, signal, signalSigma, description = load_axuv_signal(shotnum, signalDiodeArrayNum,
                                                                      expTime, rateKHz=rateKHz,
                                                                      detectorsToDrop=axuvDetectorsToDrop)

    pinholePos = pinholePositions[signalDiodeArrayNum]
    ipsSynthSignal = synthetic_halpha_signal(ipsEmissivity, impactParams, pinholePos)

    add_halpha_signal_to_axis(axHalpha, impactParams, signal, signalSigma, ipsSynthSignal,
                              signalDiodeArrayNum, description, expTime, synthColor=ipsColor)

    plot_ela_comparison(simulationName, elaShotDatePrefix, simTime, expTime=expTime,
                        showPlot=False, axes=elaAxs, simColor=ipsColor)

    plot_fluxloop_comparison(simulationName, simTime, showPlot=False, axes=axFluxLoop,
                             color=ipsColor)

    # The x extent the IPS panels need, before the WarpX profiles, which run out to the wall
    xLimIps = max(np.max(np.abs(ax.get_xlim())) for ax in midplaneAxs)

    #### WarpX, in warpxColor

    # SEE, on the same detectors as the IPS signal
    seeImpactParams, warpxSee, _, _, _ = warpx_see_signal(shotnum, runFile=warpxRunFile,
                                                          detectorsToDrop=detectorsToDrop,
                                                          fullLos=fullLos)

    axSee.plot(seeImpactParams, warpxSee, color=warpxColor, linewidth=3)

    # Midplane n_e, which is what Thomson measures, and T_e
    plot_mirrored_line(axDens, rWarpx, warpxProfiles[thomsonDensityField], color=warpxColor,
                       linewidth=3)
    plot_mirrored_line(axTe, rWarpx, warpxProfiles['T_electrons'], color=warpxColor,
                       linewidth=3)

    # H-alpha, on the same chords as the IPS signal, normalized to its own peak
    warpxSynthSignal = synthetic_warpx_halpha_signal(warpx_halpha_emissivity(warpxProfiles),
                                                     impactParams, pinholePos)

    axHalpha.plot(impactParams, warpxSynthSignal / np.max(warpxSynthSignal),
                  color=warpxColor, linewidth=3, marker='o', ms=10)

    # ELA, at the throat of the IPS eqdsk, with the radius of each ELA there
    elaTimes, elaRadiusMid = load_ela_density(elaShotDatePrefix)[:2]
    elaTimeIdx = np.argmin(np.abs(elaTimes - expTime))

    rThroatEla, zThroatEla = throat_positions(simulationName, simTime, elaRadiusMid[elaTimeIdx])
    throat = throat_profiles(fields, simulationName, simTime)

    for ax, fieldName in ((axElaDens, ionDensityField), (axElaTe, 'T_electrons')):

        simAtEla = profile_at_points(fields, fieldName, rThroatEla, zThroatEla)

        ax.plot(throat['r'], throat[fieldName], color=warpxColor, linewidth=3)
        ax.plot(rThroatEla, simAtEla, color=warpxColor, linestyle='none', marker='s', ms=12)

        # The axis was set to fit the measurements and the IPS values at the ELAs
        ax.set_ylim(0, max(ax.get_ylim()[1], 1.3 * np.max(simAtEla)))

    #### Axes

    # Same x axis for every midplane panel, all of which are symmetric about 0
    xLim = max(xLimIps, 1.1 * plasma_extent(warpxProfiles[thomsonDensityField], rWarpx))
    for ax in midplaneAxs:
        ax.set_xlim(-xLim, xLim)

    # The y limits were set before the WarpX curves were added
    inView = rWarpx <= xLim
    axSee.set_ylim(0, max(axSee.get_ylim()[1], 1.1 * np.max(warpxSee)))
    axDens.set_ylim(0, max(axDens.get_ylim()[1],
                           1.1 * np.max(warpxProfiles[thomsonDensityField][inView])))
    axTe.set_ylim(0, max(axTe.get_ylim()[1], 1.25 * np.max(warpxProfiles['T_electrons'][inView])))

    # Only the bottom panel of each column keeps its x tick labels
    for ax in (axDens, axTe):
        ax.tick_params(labelbottom=False)
        ax.set_xlabel('')
    for ax in (axElaDens, axElaTe):
        ax.tick_params(labelbottom=False)

    axDens.set_ylabel(r'n [m$^{-3}$]')

    #### Labels

    # The three sources are labelled once, in the SEE panel, so every other legend goes
    for ax in midplaneAxs + elaAxs + (axFluxLoop,):
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        for loc in ('left', 'center', 'right'):
            ax.set_title('', loc=loc)

    # Kept short so the legend fits in a corner of the panel. The runs and their times are in
    # the figure title.
    handles = [Line2D([], [], color='k', marker='o', ms=10, linestyle='none', label='Experiment'),
               Line2D([], [], color=ipsColor, linewidth=3, label='IPS'),
               Line2D([], [], color=warpxColor, linewidth=3, label='WarpX')]

    axSee.legend(handles=handles, loc='best', fontsize='small')

    # The neutral densities plot_radial_and_synthetic() writes on the density panel are in the
    # IPS simulation name in the figure title, and the box would cover the WarpX profile
    for text in list(axDens.texts):
        text.remove()

    # What each panel shows, and when it was measured, in place of the legends
    tsDensTime = load_thomson_density()[3]
    tsTeTime = load_thomson_temperature()[3]

    panelTitles = {
        axSee: r'SEE array, $\int$n dl (IPS n$_i$, WarpX n$_i$ + n$_{fast}$)',
        axDens: rf'Thomson n$_e$ {tsDensTime*1e3:.2f} ms (IPS n$_i$, WarpX n$_e$)',
        axTe: rf'Thomson T$_e$ {tsTeTime*1e3:.2f} ms',
        axHalpha: rf'AXUV DA{signalDiodeArrayNum} H$_\alpha$ {expTime*1e3:.2f} ms, each normalized to its peak',
        axElaDens: (rf'ELA {elaShotDatePrefix} {elaTimes[elaTimeIdx]*1e3:g} ms, n$_{{mid}}$/$\sqrt{{e}}$ '
                    r'(IPS n$_i$, WarpX n$_i$ + n$_{fast}$)'),
        axElaTe: r'ELA T$_e$',
        axElaTi: r'ELA T$_i$ (none from WarpX)',
        axFluxLoop: 'Flux loops (no equilibrium from WarpX)',
    }
    for ax, title in panelTitles.items():
        ax.set_title(title, loc='right', fontsize='x-small')

    fig.suptitle(rf'Shot #{shotnum}, t$_{{exp}}$ = {expTime*1e3:.2f} ms' + '\n'
                 rf'IPS: {simulationName}, t$_{{Sim}}$ = {simTime*1e3:.4g} ms' + '    '
                 f'WarpX: {fields["runName"]}, {warpx_time_label(fields)}',
                 fontsize='medium')

    if saveplot:
        save_plot(fig, f'{shotnum}_{simulationName}_warpx_{fields["runName"]}_multi_diagnostic_comparison.png')

    if showPlot:
        plt.show()

    return fig, axSee, axDens, axTe, axHalpha, elaAxs, axFluxLoop

if __name__ == "__main__":

    # IPS simulation, WarpX run and shot to compare
    simName = 'nneut_3e16_gb_1e18_NBI_800kW_ECH_0kW_ionDrrOn'
    warpxRunFile = 'fields_rtz.npz'
    shotnum = 260709060

    # Times to plot (in seconds). The WarpX fields are averaged over their dumps, so simTime is
    # the IPS time alone.
    simTime = 3e-3
    expTime = 7.83e-3

    # SEE detectors with bad data in this shot
    detectorsToDrop = [9, 13, 15]

    # AXUV chords with bad data in this shot (chord indices, see plot_axuv_raw_signals() in
    # axuv_comparison.py)
    axuvDetectorsToDrop = [8, 9, 10]

    # Date prefix of the ELA data in config.elaDataDir
    elaShotDatePrefix = 260709

    plot_multi_diagnostic_comparison(simName, shotnum, simTime, expTime,
                                     elaShotDatePrefix=elaShotDatePrefix, useExtension=True,
                                     detectorsToDrop=detectorsToDrop,
                                     axuvDetectorsToDrop=axuvDetectorsToDrop,
                                     warpxRunFile=warpxRunFile, saveplot=True)
