"""
This script compares the H-alpha emission of a WarpX simulation against the AXUV array.

It is the WarpX version of axuv_comparison.py. The midplane profiles come from the WarpX fields,
averaged over the dumps and theta (see warpx_simulation.py), rather than from CQL3D and KN1D, so-
    - The electron and ion densities are separate (no n_e = n_i).
    - The neutral density is WarpX's own, on the same radial grid as the plasma, rather than one
      value per flux tube from KN1D.
    - The grid reaches the wall, so no AXUV extension is needed.
"""

import numpy as np
import matplotlib.pyplot as plt

from plotting import save_plot, plot_mirrored_profile, densityColor, temperatureColor
from axuv import load_axuv_signal, pinholePositions
from axuv_comparison import (line_integrate_emissivity, add_halpha_signal_to_axis,
                             plot_axuv_raw_signals)
from atomic_data import load_pec_table, evaluate_pec
from warpx_simulation import load_warpx_fields, midplane_profiles, thomsonDensityField, ionDensityField

# Color of the neutral density, as in axuv_comparison.py
neutralColor = 'C2'

def warpx_time_label(fields):
    """
    Label of the time window the WarpX fields were averaged over, e.g. 't$_{Sim}$ = 1.08-1.1 ms'.

    Parameters
    ----------
    fields : dict
        The averaged fields, from load_warpx_fields().

    Returns
    -------
    label : str
        The label.
    """

    times = fields['times']

    return rf't$_{{Sim}}$ = {times[0]*1e3:.4g}-{times[-1]*1e3:.4g} ms'

def warpx_halpha_emissivity(profiles):
    """
    Calculate the H-alpha emissivity on the WarpX midplane radial grid.

    As in halpha_emissivity() in axuv_comparison.py, the emissivity is electron impact excitation
    of the neutrals plus recombination of the ions-
        eps = n_e * n_0 * PEC_excit(n_e, T_e) + n_e * n_i * PEC_recom(n_e, T_e)
    with the PECs of the ADAS n-resolved Balmer data. Here n_e, n_0 and n_i are WarpX's own, with
    n_i the thermal plus fast ions (ionDensityField). Where there are no electrons the emissivity
    is zero.

    The caveats of halpha_emissivity() still apply: the PECs assume Maxwellian electrons, only
    atomic neutrals are included (no H2), and the PECs are for hydrogen.

    Parameters
    ----------
    profiles : dict
        The WarpX midplane profiles, from midplane_profiles() in warpx_simulation.py.

    Returns
    -------
    emissivity : dict
        'r'          : [r], midplane radius. [m]
        'excit'      : [r], emissivity from electron impact excitation. [photons m^-3 s^-1]
        'recom'      : [r], emissivity from recombination. [photons m^-3 s^-1]
        'total'      : [r], total emissivity. [photons m^-3 s^-1]
        'adf15File'  : name of the ADAS file the PECs are from.
    """

    pecTable = load_pec_table()

    ne = profiles['n_electrons']
    Te = profiles['T_electrons']
    nNeutral = profiles['n_neutrals']
    nIon = profiles[ionDensityField]

    # The PECs are interpolated in log10(n_e), log10(T_e), so the points without plasma are
    # left out and given zero emission
    hasPlasma = (ne > 0) & (Te > 0)

    emissExcit = np.zeros_like(ne)
    emissRecom = np.zeros_like(ne)

    emissExcit[hasPlasma] = (ne[hasPlasma] * nNeutral[hasPlasma]
                             * evaluate_pec(pecTable, 'excit', ne[hasPlasma], Te[hasPlasma]))
    emissRecom[hasPlasma] = (ne[hasPlasma] * nIon[hasPlasma]
                             * evaluate_pec(pecTable, 'recom', ne[hasPlasma], Te[hasPlasma]))

    emissivity = {'r': profiles['r'],
                  'excit': emissExcit,
                  'recom': emissRecom,
                  'total': emissExcit + emissRecom,
                  'adf15File': pecTable['adf15File']}

    return emissivity

def synthetic_warpx_halpha_signal(emissivity, impactParams, pinholePos, numPoints=1000):
    """
    Line-integrate the WarpX H-alpha emissivity along the line of sight of each chord.

    The emissivity is taken at the midplane and is linearly interpolated in radius along each
    chord (see line_integrate_emissivity() in axuv_comparison.py), out to the edge of the WarpX
    grid.

    Parameters
    ----------
    emissivity : dict
        The H-alpha emissivity, from warpx_halpha_emissivity().
    impactParams : np.array
        1D array of the impact parameter of each chord. [m]
    pinholePos : np.array
        (x, y) of the pinhole, in the frame of axuv.pinholePositions. [m]
    numPoints : int
        Number of points along each chord the emissivity is integrated over.
        Default is 1000.

    Returns
    -------
    synthSignal : np.array
        1D array of the line-integrated emissivity of each chord. [photons m^-2 s^-1]
    """

    r = emissivity['r']

    return line_integrate_emissivity(r, emissivity['total'], r[-1], impactParams, pinholePos,
                                     numPoints=numPoints)

def plasma_extent(profile, r, fraction=1e-2):
    """
    Radius out to which a midplane profile is above a fraction of its peak, to set the x limits
    of the plots without the empty part of the WarpX grid near the wall.

    Parameters
    ----------
    profile : np.array
        1D array of the profile on r.
    r : np.array
        1D array of the radius. [m]
    fraction : float
        Fraction of the peak.
        Default is 0.01.

    Returns
    -------
    rExtent : float
        The outermost radius at which the profile is above fraction * its peak. [m]
    """

    return r[np.nonzero(profile > fraction * np.max(profile))[0][-1]]

def plot_warpx_axuv_comparison(shotnum, expTime, runFile='fields_rtz.npz', signalDiodeArrayNum=2,
                               rateKHz=10.0, detectorsToDrop=None,
                               showPlot=True, saveplot=False, axes=None):
    """
    Plot the WarpX midplane profiles against the measured AXUV signal.

    The five panels, from top to bottom, are-
        1. The midplane electron density and ion density (thermal + fast).
        2. The midplane electron temperature.
        3. The midplane neutral density.
        4. The H-alpha emissivity, calculated from panels 1-3 by warpx_halpha_emissivity().
        5. The measured line-integrated signal of each chord of the AXUV array, averaged over a
           window of width 1/rateKHz centred on expTime, against the synthetic signal of each
           chord (synthetic_warpx_halpha_signal()). Each is normalized to its peak.
    The radial profiles are mirrored about r = 0 so that all five share the same x axis.

    Parameters
    ----------
    shotnum : int
        The shot number the experimental data is taken from.
    expTime : float
        The experimental time of the AXUV signal. [s]
    runFile : str
        The WarpX field dump, see load_warpx_fields().
        Default is 'fields_rtz.npz'.
    signalDiodeArrayNum : int
        The diode array whose line-integrated signal is plotted.
        Default is 2, the H-alpha filtered array (DIODEARRAY2).
    rateKHz : float
        Sets the width of the window the AXUV signal is averaged over, 1/rateKHz. [kHz]
        Default is 10.
    detectorsToDrop : list of int
        Indices of the AXUV chords with bad data, which are left out. These are in the order the
        chords are stored in the AXUV file, the index on each panel of plot_axuv_raw_signals().
        Default is None, which keeps every chord.
    showPlot : bool
        Show the figure before returning. Set to False to add to the figure afterwards.
        Default is True.
    saveplot : bool
        Save the plot in config.plotDir as
        f'{shotnum}_warpx_{runName}_axuv_DA{signalDiodeArrayNum}_comparison.png'.
        Default is False.
    axes : tuple of matplotlib.axes.Axes
        (density axis, electron temperature axis, neutral density axis, H-alpha emissivity axis,
        AXUV signal axis) to draw into instead of making a new figure.
        Default is None, which makes a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axDens, axTe, axNH, axHa, axSig : matplotlib.axes.Axes
        The density, electron temperature, neutral density, H-alpha emissivity and AXUV signal
        axes.
    """

    #### WarpX profiles and emissivity

    fields = load_warpx_fields(runFile)
    profiles = midplane_profiles(fields)
    emissivity = warpx_halpha_emissivity(profiles)

    #### Measured and synthetic AXUV signal

    impactParams, signal, signalSigma, description = load_axuv_signal(shotnum, signalDiodeArrayNum,
                                                                      expTime, rateKHz=rateKHz,
                                                                      detectorsToDrop=detectorsToDrop)

    synthSignal = synthetic_warpx_halpha_signal(emissivity, impactParams,
                                                pinholePositions[signalDiodeArrayNum])

    # The WarpX grid runs out to the wall, so the axis stops where the plasma does
    r = profiles['r']
    xLim = 1.1 * max(np.max(np.abs(impactParams)), plasma_extent(profiles['n_electrons'], r))

    if axes is None:

        fig = plt.figure(figsize=(12, 25), tight_layout=True)

        axDens = fig.add_subplot(5, 1, 1)
        axTe = fig.add_subplot(5, 1, 2, sharex=axDens)
        axNH = fig.add_subplot(5, 1, 3, sharex=axDens)
        axHa = fig.add_subplot(5, 1, 4, sharex=axDens)
        axSig = fig.add_subplot(5, 1, 5, sharex=axDens)

    else:

        axDens, axTe, axNH, axHa, axSig = axes
        fig = axDens.figure

    #### Midplane density

    plot_mirrored_profile(axDens, r, profiles[thomsonDensityField], color=densityColor,
                          label=r'n$_e$')
    plot_mirrored_profile(axDens, r, profiles[ionDensityField], color='C9',
                          label=r'n$_i$ + n$_{fast}$')

    axDens.set_ylabel(r'n [m$^{-3}$]')
    axDens.set_ylim(0, 1.6 * np.max(profiles[thomsonDensityField]))
    axDens.legend(loc='upper right', fontsize='small')

    axDens.set_title(f'WarpX {fields["runName"]}, Shot #{shotnum}', loc='right', fontsize='small')

    axDens.text(0.02, 0.95, warpx_time_label(fields),
                transform=axDens.transAxes,
                ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    #### Midplane electron temperature

    plot_mirrored_profile(axTe, r, profiles['T_electrons'], color=temperatureColor)

    axTe.set_ylabel(r'T$_e$ [eV]')
    axTe.set_ylim(0, 1.4 * np.max(profiles['T_electrons'][np.abs(r) <= xLim]))

    #### Midplane neutral density

    plot_mirrored_profile(axNH, r, profiles['n_neutrals'], color=neutralColor)

    axNH.set_ylabel(r'n$_0$ [m$^{-3}$]')

    # The neutrals are attenuated as they move in from the wall, so the profile spans decades.
    # The limits are whole decades, so that at least two are labelled.
    axNH.set_yscale('log')
    nNeutralShown = profiles['n_neutrals'][(r <= xLim) & (profiles['n_neutrals'] > 0)]
    axNH.set_ylim(10**np.floor(np.log10(np.min(nNeutralShown))),
                  10**np.ceil(np.log10(np.max(nNeutralShown))))

    #### H-alpha emissivity

    # Both parts are shown, as recombination can be comparable to excitation in the core. The
    # total is drawn last, so it stays on top where it lies on the larger of the two.
    plot_mirrored_profile(axHa, r, emissivity['excit'], color='C4', label='Excitation')
    plot_mirrored_profile(axHa, r, emissivity['recom'], color='C5', label='Recombination')
    plot_mirrored_profile(axHa, r, emissivity['total'], color='C3', label='Total')

    axHa.set_ylabel(r'$\epsilon_{H_\alpha}$ [ph m$^{-3}$ s$^{-1}$]')
    axHa.set_yscale('log')

    # Down to 1e-4 of the peak, below which the edge of the grid only adds empty decades, and
    # up to 1e3 above it, to keep the legend clear of the profile
    axHa.set_ylim(1e-4 * np.max(emissivity['total']), 1e3 * np.max(emissivity['total']))

    axHa.legend(loc='upper center', fontsize='x-small', ncol=3,
                title=f'ADAS {emissivity["adf15File"]}', title_fontsize='x-small')

    #### Line-integrated AXUV signal

    add_halpha_signal_to_axis(axSig, impactParams, signal, signalSigma, synthSignal,
                              signalDiodeArrayNum, description, expTime)

    axSig.set_xlim(-xLim, xLim)

    # The x axis is shared, so only the bottom panel keeps its tick labels
    for ax in (axDens, axTe, axNH, axHa):
        ax.tick_params(labelbottom=False)

    if saveplot:
        save_plot(fig, f'{shotnum}_warpx_{fields["runName"]}_axuv_DA{signalDiodeArrayNum}_comparison.png')

    if showPlot:
        plt.show()

    return fig, axDens, axTe, axNH, axHa, axSig

if __name__ == "__main__":

    # WarpX run and shot to compare
    runFile = 'fields_rtz.npz'
    shotnum = 260709060

    # Experimental time (in seconds). The WarpX fields are averaged over their dumps.
    expTime = 7.83e-3

    # AXUV chords with bad data in this shot (chord indices, see plot_axuv_raw_signals())
    axuvDetectorsToDrop = None

    # Raw signal of every chord against time, to check for bad chords
    plot_axuv_raw_signals(shotnum, expTime=expTime, detectorsToDrop=axuvDetectorsToDrop,
                          saveplot=True, showPlot=False)

    plot_warpx_axuv_comparison(shotnum, expTime, runFile=runFile,
                               detectorsToDrop=axuvDetectorsToDrop, saveplot=True)
