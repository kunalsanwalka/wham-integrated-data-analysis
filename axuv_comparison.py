"""
This script compares the CQL3D simulation against the H-alpha filtered AXUV array (DIODEARRAY2).

For now it plots, at one pair of simulated and experimental times-
    - The simulated midplane density.
    - The simulated midplane electron temperature.
    - The simulated midplane neutral density in each flux tube, from KN1D.
    - The simulated H-alpha emissivity in each flux tube, calculated from the three profiles
      above with the ADAS photon emissivity coefficients.
    - The measured line-integrated H-alpha signal, and the synthetic one from integrating the
      simulated emissivity along each chord. Both are normalized to their peak.
The simulated profiles are on the CQL3D grid alone, i.e. without the AXUV extension.
"""

import numpy as np
import matplotlib.pyplot as plt

from plotting import (save_plot, plot_mirrored_profile, densityColor, temperatureColor,
                      halphaColor)
from cql3d_simulation import (load_density_data, load_temperature_data, parse_simulation_name,
                              density_label, flux_tube_edges)
from kn1d_simulation import neutral_density_per_flux_tube
from axuv import (load_axuv_signal, load_axuv_raw_signal, load_axuv_impact_params,
                  pinholePositions, chord_directions, chord_circle_intersections, average_down,
                  axuv_keep_mask)
from atomic_data import load_pec_table, evaluate_pec

def load_simulated_profiles(simulationName, simTime, redoAnalysis=False):
    """
    Load the simulated midplane density, electron temperature and neutral density profiles at one
    time.

    The density and electron temperature are taken at the CQL3D timestep closest to simTime.
    The neutral density is calculated by KN1D once per IPS timestep, so it is taken from the end
    of the IPS timestep closest to simTime. It is the neutral density of the midplane KN1D run
    (the main vessel neutrals), averaged over the midplane cross-section of each flux tube (see
    neutral_density_per_flux_tube() in kn1d_simulation.py).

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    simTime : float
        The simulation time of the profiles. [s]
        The closest available simulated timestep is used.
    redoAnalysis : bool
        Force the temperature and neutral density profiles to be recomputed instead of loading
        the saved ones.
        Default is False.

    Returns
    -------
    profiles : dict
        'rMid'       : [r], midplane radius of the density and temperature. [m]
        'dens'       : [r], midplane ion density. [m^-3]
        'Te'         : [r], midplane electron temperature. [keV]
        'rTube'      : [Flux Tube], midplane radius of the centre of each flux tube. [m]
        'nHTube'     : [Flux Tube], midplane neutral density in each flux tube. [m^-3]
        'simTime'    : simulated time of the density and temperature actually used. [s]
        'nHTime'     : simulated time of the neutral density actually used. [s]
    """

    #### Density

    dens, solrz, _, densTimes = load_density_data(simulationName)

    simTimeIdx = np.argmin(np.abs(densTimes - simTime))

    #### Electron temperature

    # On the same timebase as the density
    Te, _, rTe, teTimes = load_temperature_data(simulationName, redoAnalysis=redoAnalysis)

    teTimeIdx = np.argmin(np.abs(teTimes - simTime))

    #### Neutral density

    nHTube, rTube, nHTimes = neutral_density_per_flux_tube(simulationName, 'midplane',
                                                           redoAnalysis=redoAnalysis)

    nHTimeIdx = np.argmin(np.abs(nHTimes - simTime))

    profiles = {'rMid': solrz[:, 0],
                'dens': dens[simTimeIdx, :, 0],
                'Te': Te[teTimeIdx],
                'rTube': rTube[nHTimeIdx],
                'nHTube': nHTube[nHTimeIdx],
                'simTime': densTimes[simTimeIdx],
                'nHTime': nHTimes[nHTimeIdx]}

    return profiles

def halpha_emissivity(profiles):
    """
    Calculate the H-alpha emissivity in each flux tube from the simulated midplane profiles.

    The emissivity is the sum of electron impact excitation of the neutrals and recombination of
    the ions-
        eps = n_e * n_H * PEC_excit(n_e, T_e) + n_e * n_i * PEC_recom(n_e, T_e)
    with the PECs of the ADAS n-resolved Balmer data (see load_pec_table() in atomic_data.py).
    The plasma is taken to be quasi-neutral hydrogen, so n_e = n_i.

    The neutral density is only known per flux tube, so the density and electron temperature are
    interpolated onto the midplane radius of the centre of each flux tube.

    Caveats-
        - The PECs assume Maxwellian electrons at T_e. The CQL3D electrons are a general species,
          and T_e here is (2/3)<E> of their distribution function.
        - Only atomic neutrals are included. KN1D's molecular density (and so the H-alpha from
          dissociative excitation of H2) is not used, and neither is the charge exchange
          emission from the fast ions.
        - The PECs are for hydrogen. For a deuterium plasma, D-alpha is still well inside the
          654-659 nm filter, and the change in the PECs is small.

    Parameters
    ----------
    profiles : dict
        The simulated midplane profiles, from load_simulated_profiles().

    Returns
    -------
    emissivity : dict
        'rTube'      : [Flux Tube], midplane radius of the centre of each flux tube. [m]
        'excit'      : [Flux Tube], emissivity from electron impact excitation. [photons m^-3 s^-1]
        'recom'      : [Flux Tube], emissivity from recombination. [photons m^-3 s^-1]
        'total'      : [Flux Tube], total emissivity. [photons m^-3 s^-1]
        'adf15File'  : name of the ADAS file the PECs are from.
    """

    pecTable = load_pec_table()

    rTube = profiles['rTube']

    # Onto the flux tubes. Quasi-neutral, so n_e = n_i.
    ne = np.interp(rTube, profiles['rMid'], profiles['dens'])
    # [keV] to [eV]
    Te = np.interp(rTube, profiles['rMid'], profiles['Te']) * 1e3

    emissExcit = ne * profiles['nHTube'] * evaluate_pec(pecTable, 'excit', ne, Te)
    emissRecom = ne * ne * evaluate_pec(pecTable, 'recom', ne, Te)

    emissivity = {'rTube': rTube,
                  'excit': emissExcit,
                  'recom': emissRecom,
                  'total': emissExcit + emissRecom,
                  'adf15File': pecTable['adf15File']}

    return emissivity

def synthetic_halpha_signal(emissivity, impactParams, pinholePos, numPoints=1000):
    """
    Line-integrate the simulated H-alpha emissivity along the line of sight of each chord.

    The emissivity is a function of the midplane radius alone. Along each chord it is-
        - Linearly interpolated between the flux tube centres.
        - Held at the innermost flux tube's value inside its centre, as the simulation does not
          reach r = 0.
        - Held at the outermost flux tube's value out to the outer edge of that tube (see
          flux_tube_edges() in cql3d_simulation.py), and zero beyond it.
    Chords that miss the simulated plasma entirely get zero signal.

    Parameters
    ----------
    emissivity : dict
        The H-alpha emissivity, from halpha_emissivity().
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

    rTube = emissivity['rTube']
    rEdge = flux_tube_edges(rTube)[-1]

    return line_integrate_emissivity(rTube, emissivity['total'], rEdge, impactParams, pinholePos,
                                     numPoints=numPoints)

def line_integrate_emissivity(rProfile, emissProfile, rEdge, impactParams, pinholePos,
                              numPoints=1000):
    """
    Line-integrate a midplane emissivity profile along the line of sight of each chord.

    The chords lie in the midplane cross-section, so the emissivity along each one only depends on
    the distance from the axis. Along each chord it is linearly interpolated in radius on
    rProfile, held at its first value inside rProfile[0] and at its last value out to rEdge, and
    zero beyond rEdge. Chords that miss the circle of radius rEdge get zero signal.

    This is the integral behind synthetic_halpha_signal() (CQL3D flux tubes), and it takes any
    radial grid, e.g. the WarpX one.

    Parameters
    ----------
    rProfile : np.array
        1D array of the radius of each point of the emissivity profile, increasing. [m]
    emissProfile : np.array
        1D array of the emissivity on rProfile. [photons m^-3 s^-1]
    rEdge : float
        Radius beyond which the emissivity is zero. [m]
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

    directions = chord_directions(impactParams, pinholePos)
    sIn, sOut = chord_circle_intersections(pinholePos, directions, rEdge)

    synthSignal = np.zeros(len(impactParams))

    for i in range(len(impactParams)):

        # Misses the simulated plasma
        if np.isnan(sIn[i]):
            continue

        # [Point x 2] along the chord, inside the simulated plasma
        s = np.linspace(sIn[i], sOut[i], numPoints)
        points = pinholePos + s[:, None] * directions[i]

        # np.interp holds the end values, which gives the flat core and outermost tube
        epsChord = np.interp(np.linalg.norm(points, axis=1), rProfile, emissProfile)

        synthSignal[i] = np.trapezoid(epsChord, s)

    return synthSignal

def add_halpha_signal_to_axis(ax, impactParams, signal, signalSigma, synthSignal,
                              signalDiodeArrayNum, description, expTime, synthColor=halphaColor):
    """
    Plot the measured line-integrated AXUV signal of each chord against the synthetic H-alpha
    signal.

    The two are in different units and differ in magnitude, so each is normalized to its peak.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw on.
    impactParams, signal, signalSigma, description :
        The measured signal, from load_axuv_signal().
    synthSignal : np.array
        1D array of the synthetic signal of each chord, from synthetic_halpha_signal().
    signalDiodeArrayNum : int
        The diode array the signal is from.
    expTime : float
        The experimental time of the signal. [s]
    synthColor : str
        Color of the synthetic signal.
        Default is halphaColor.
    """

    # The description holds the filter on a second line, e.g. '$H_{\alpha}$\n[654-659 nm]'
    signalLabel = ' '.join(description.split())

    signalNorm = np.max(signal)
    synthNorm = np.max(synthSignal)

    ax.plot(impactParams, signal / signalNorm,
            color='k',
            linewidth=3,
            linestyle='dashed')

    ax.errorbar(impactParams, signal / signalNorm,
                yerr=signalSigma / signalNorm,
                label=f'DA{signalDiodeArrayNum} {signalLabel} {expTime*1e3:.2f}ms',
                color='k',
                fmt='o',
                ms=10,
                elinewidth=3)

    ax.plot(impactParams, synthSignal / synthNorm,
            label='Synthetic',
            color=synthColor,
            linewidth=3,
            marker='o',
            ms=10)

    ax.set_ylabel(r'$\int \epsilon \cdot dl$ [Normalized]')

    # Headroom above the error bars for the legend
    ax.set_ylim(0, 1.6 * np.max((signal + signalSigma) / signalNorm))

    ax.set_xlabel('Radius / Impact Parameter [m]')

    ax.legend(loc='upper center')

    return

def plot_axuv_comparison(simulationName, shotnum, simTime, expTime, signalDiodeArrayNum=2,
                         rateKHz=10.0, detectorsToDrop=None, redoAnalysis=False,
                         showPlot=True, saveplot=False, axes=None):
    """
    Plot the simulated midplane profiles against the measured AXUV signal.

    The five panels, from top to bottom, are-
        1. The simulated midplane ion density.
        2. The simulated midplane electron temperature.
        3. The simulated midplane neutral density in each flux tube, from KN1D, plotted at the
           midplane radius of the centre of each flux tube.
        4. The simulated H-alpha emissivity in each flux tube, calculated from panels 1-3 by
           halpha_emissivity().
        5. The measured line-integrated signal of each chord of the AXUV array, averaged over a
           window of width 1/rateKHz centred on expTime. The error bars are the standard
           deviation of the signal within that window. Along with it, the synthetic signal of
           each chord, from synthetic_halpha_signal(). The two differ in magnitude (and units),
           so each is normalized to its peak.
    The radial profiles are mirrored about r = 0 so that all five share the same x axis.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    shotnum : int
        The shot number the experimental data is taken from.
    simTime : float
        The simulation time of the profiles. [s]
        The closest available simulated timestep is used.
    expTime : float
        The experimental time of the AXUV signal. [s]
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
    redoAnalysis :
        See load_simulated_profiles().
    showPlot : bool
        Show the figure before returning. Set to False to add to the figure afterwards.
        Default is True.
    saveplot : bool
        Save the plot in config.plotDir as
        f'{shotnum}_{simulationName}_axuv_DA{signalDiodeArrayNum}_comparison.png'.
        Default is False.
    axes : tuple of matplotlib.axes.Axes
        (density axis, electron temperature axis, neutral density axis, H-alpha emissivity axis,
        AXUV signal axis) to draw into instead of making a new figure, e.g. to place this plot in
        a column of a larger figure.
        Default is None, which makes a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axDens : matplotlib.axes.Axes
        The density axis.
    axTe : matplotlib.axes.Axes
        The electron temperature axis.
    axNH : matplotlib.axes.Axes
        The neutral density axis.
    axHa : matplotlib.axes.Axes
        The H-alpha emissivity axis.
    axSig : matplotlib.axes.Axes
        The AXUV signal axis.
    """

    # Simulated profiles
    profiles = load_simulated_profiles(simulationName, simTime, redoAnalysis=redoAnalysis)

    # Simulated H-alpha emissivity
    emissivity = halpha_emissivity(profiles)

    # Measured AXUV signal
    impactParams, signal, signalSigma, description = load_axuv_signal(shotnum, signalDiodeArrayNum,
                                                                      expTime, rateKHz=rateKHz,
                                                                      detectorsToDrop=detectorsToDrop)

    # Synthetic signal along the same chords
    synthSignal = synthetic_halpha_signal(emissivity, impactParams,
                                          pinholePositions[signalDiodeArrayNum])

    # The outermost chords reach past the simulation grid, so both are taken into account to keep
    # the panels on the same x axis
    xLim = 1.1 * max(np.max(np.abs(impactParams)), np.max(profiles['rMid']))

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

    rMid = profiles['rMid']

    #### Midplane density

    plot_mirrored_profile(axDens, rMid, profiles['dens'], color=densityColor)

    axDens.set_ylabel(r'n$_i$ [m$^{-3}$]')

    # The profile is flat topped, so it is given headroom to keep it clear of the labels
    axDens.set_ylim(0, 1.6 * np.max(profiles['dens']))

    axDens.set_title(f'Shot #{shotnum}', loc='right')

    # Label the panel with the neutral densities of the simulation, in the empty space under the
    # flat topped profile
    mainVesselDens, gasBoxDens = parse_simulation_name(simulationName)
    if mainVesselDens is not None:
        densText = (f'Gas Box '+r'$n_n$'+f': {density_label(gasBoxDens)}\n'
                    f'Main Vessel '+r'$n_n$'+f': {density_label(mainVesselDens)}')
        axDens.text(0.5, 0.05, densText,
                    transform=axDens.transAxes,
                    ha='center', va='bottom',
                    fontsize='small',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axDens.text(0.02, 0.95, rf't$_{{Sim}}$ = {profiles["simTime"]*1e3:.4g} ms',
                transform=axDens.transAxes,
                ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    #### Midplane electron temperature

    # [keV] to [eV]
    plot_mirrored_profile(axTe, rMid, profiles['Te']*1e3, color=temperatureColor)

    axTe.set_ylabel(r'T$_e$ [eV]')

    # The profile is close to flat, so it is given some headroom to keep it off the frame
    axTe.set_ylim(0, 1.4 * np.max(profiles['Te']) * 1e3)

    #### Midplane neutral density per flux tube

    plot_mirrored_profile(axNH, profiles['rTube'], profiles['nHTube'],
                          color='C2', label='Main vessel (midplane KN1D run)', marker='o')

    axNH.set_ylabel(r'n$_H$ per flux tube [m$^{-3}$]')

    # The neutrals are attenuated as they move in from the wall, so the profile spans decades
    axNH.set_yscale('log')

    # KN1D is run once per IPS timestep, so its time can differ from that of the profiles above.
    # It goes in the legend title, as the profile fills both lower corners of the panel.
    axNH.legend(loc='upper center', fontsize='small',
                title=rf't$_{{KN1D}}$ = {profiles["nHTime"]*1e3:.4g} ms', title_fontsize='small')

    #### H-alpha emissivity per flux tube

    plot_mirrored_profile(axHa, emissivity['rTube'], emissivity['total'],
                          color=halphaColor, label='Total', marker='o')

    # In the core the neutral density is low enough for recombination to be comparable to
    # excitation, so both parts are shown
    plot_mirrored_profile(axHa, emissivity['rTube'], emissivity['excit'],
                          color='C4', label='Excitation')
    plot_mirrored_profile(axHa, emissivity['rTube'], emissivity['recom'],
                          color='C5', label='Recombination')

    axHa.set_ylabel(r'$\epsilon_{H_\alpha}$ [ph m$^{-3}$ s$^{-1}$]')

    # It follows the neutral density, so it also spans decades
    axHa.set_yscale('log')

    # The PECs used go in the legend title
    axHa.legend(loc='upper center', fontsize='small',
                title=f'ADAS {emissivity["adf15File"]}', title_fontsize='small')

    #### Line-integrated AXUV signal

    add_halpha_signal_to_axis(axSig, impactParams, signal, signalSigma, synthSignal,
                              signalDiodeArrayNum, description, expTime)

    axSig.set_xlim(-xLim, xLim)

    # The x axis is shared, so only the bottom panel keeps its tick labels
    for ax in (axDens, axTe, axNH, axHa):
        ax.tick_params(labelbottom=False)

    if saveplot:
        save_plot(fig, f'{shotnum}_{simulationName}_axuv_DA{signalDiodeArrayNum}_comparison.png')

    if showPlot:
        plt.show()

    return fig, axDens, axTe, axNH, axHa, axSig

def plot_axuv_raw_signals(shotnum, diodeArrayNum=2, expTime=None, rateKHz=10.0,
                          detectorsToDrop=None, tMin=None, tMax=None,
                          showPlot=True, saveplot=False):
    """
    Plot the raw signal of every chord of an AXUV diode array against time, one panel per chord,
    to check each chord for saturation or other faults before choosing detectorsToDrop.

    Each panel shows the signal at the native (~1 MHz) rate in grey and averaged down to rateKHz
    in black. The panels are in order of impact parameter and share both axes, so the chords
    can be compared with each other directly. The y axis is set by the 99.9th percentile of the
    native samples over all chords, so isolated spikes above it are cut off rather than squashing
    every panel.

    Each panel is titled with its chord index, in file order, which is the index detectorsToDrop
    takes (see load_axuv_signal() in axuv.py), and its impact parameter.

    Parameters
    ----------
    shotnum : int
        The shot number.
    diodeArrayNum : int
        The diode array.
        Default is 2, the H-alpha filtered array (DIODEARRAY2).
    expTime : float
        The experimental time of the comparison. [s]
        If given, the averaging window of load_axuv_signal(), of width 1/rateKHz centred on it,
        is shaded on every panel.
        Default is None.
    rateKHz : float
        Rate the native data is averaged down to, and the inverse width of the averaging window
        at expTime. [kHz]
        Default is 10.
    detectorsToDrop : list of int
        Chord indices that are dropped from the comparison. Their panels are marked as dropped.
        Default is None.
    tMin : float
        Start of the time axis. [s]
        Default is None, which starts at the beginning of the data.
    tMax : float
        End of the time axis. [s]
        Default is None, which ends at the end of the data.
    showPlot : bool
        Show the figure before returning.
        Default is True.
    saveplot : bool
        Save the plot in config.plotDir as f'{shotnum}_axuv_DA{diodeArrayNum}_raw_signals.png'.
        Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axs : np.array of matplotlib.axes.Axes
        The axis of each panel, in order of impact parameter.
    """

    timeArr, impactParams, signalRaw, description = load_axuv_raw_signal(shotnum, diodeArrayNum)

    keep = axuv_keep_mask(len(impactParams), detectorsToDrop)

    inRange = np.ones(len(timeArr), dtype=bool)
    if tMin is not None:
        inRange &= timeArr >= tMin
    if tMax is not None:
        inRange &= timeArr <= tMax
    timeArr, signalRaw = timeArr[inRange], signalRaw[:, inRange]

    numChords = len(impactParams)
    numCols = 5
    numRows = int(np.ceil(numChords / numCols))

    fig, axs = plt.subplots(numRows, numCols, figsize=(6*numCols, 4.5*numRows),
                            sharex=True, sharey=True, tight_layout=True, squeeze=False)
    axs = axs.ravel()

    yMax = 1.1 * np.max(np.percentile(signalRaw, 99.9, axis=1))

    # [ms] for the axis
    timeMs = timeArr * 1e3

    for panelIdx, chordIdx in enumerate(np.argsort(impactParams)):

        ax = axs[panelIdx]

        ax.plot(timeMs, signalRaw[chordIdx],
                color='0.7',
                linewidth=0.5,
                label='1 MHz' if panelIdx == 0 else None)

        avgTimes, avgSignal = average_down(timeArr, signalRaw[chordIdx], rateKHz)
        ax.plot(avgTimes * 1e3, avgSignal,
                color='k',
                linewidth=2,
                label=f'{rateKHz:g} kHz' if panelIdx == 0 else None)

        if expTime is not None:
            halfWidth = 0.5 / (rateKHz * 1e3)
            ax.axvspan((expTime - halfWidth) * 1e3, (expTime + halfWidth) * 1e3,
                       color=halphaColor, alpha=0.3,
                       label=f'{expTime*1e3:.4g} ms window' if panelIdx == 0 else None)

        title = f'#{chordIdx}, b = {impactParams[chordIdx]*1e2:+.1f} cm'
        if not keep[chordIdx]:
            title += ' (dropped)'
            ax.set_facecolor('0.92')
        ax.set_title(title, fontsize='small')

        ax.grid(alpha=0.3)

    for ax in axs[numChords:]:
        ax.set_visible(False)

    axs[0].set_ylim(0, yMax)

    for ax in axs[(numRows - 1) * numCols:numChords]:
        ax.set_xlabel('Time [ms]')
    for ax in axs[::numCols]:
        ax.set_ylabel('Signal [nA]')

    axs[0].legend(loc='upper right', fontsize='x-small')

    fig.suptitle(f'Shot #{shotnum}, DA{diodeArrayNum} {" ".join(description.split())}, '
                 'raw signal (signal_raw_nA) by chord, in order of impact parameter')

    if saveplot:
        save_plot(fig, f'{shotnum}_axuv_DA{diodeArrayNum}_raw_signals.png')

    if showPlot:
        plt.show()

    return fig, axs[:numChords]

def plot_axuv_chords(shotnum, diodeArrayNum=2, plasmaRadius=0.15,
                     showPlot=True, saveplot=False, ax=None):
    """
    Plot the line of sight of every chord of an AXUV diode array in the midplane cross-section,
    to check the geometry before line-integrating along them.

    Each chord is drawn from the pinhole (axuv.pinholePositions) through the plasma and colored
    by its impact parameter. The point of closest approach to the axis is marked on each chord,
    so its distance from (0, 0) can be checked against |b|.

    Parameters
    ----------
    shotnum : int
        The shot number the impact parameters are taken from.
    diodeArrayNum : int
        The diode array.
        Default is 2, the H-alpha filtered array (DIODEARRAY2).
    plasmaRadius : float
        Radius of the circle drawn around the axis. [m]
        Default is 0.15.
    showPlot : bool
        Show the figure before returning. Set to False to add to the figure afterwards.
        Default is True.
    saveplot : bool
        Save the plot in config.plotDir as f'{shotnum}_axuv_DA{diodeArrayNum}_chords.png'.
        Default is False.
    ax : matplotlib.axes.Axes
        Axis to draw into instead of making a new figure.
        Default is None, which makes a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axis.
    """

    pinholePos = pinholePositions[diodeArrayNum]

    impactParams = load_axuv_impact_params(shotnum, diodeArrayNum)
    directions = chord_directions(impactParams, pinholePos)

    # Distance from the pinhole to the closest approach of each chord to the axis
    sClosest = -(directions @ pinholePos)
    closestPoints = pinholePos + sClosest[:, None] * directions

    # Draw every chord the same length past its closest approach, so that the chords that miss
    # the circle are drawn too
    sEnd = sClosest + 1.5 * plasmaRadius

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 12), tight_layout=True)
    else:
        fig = ax.figure

    bMax = np.max(np.abs(impactParams))
    norm = plt.Normalize(-bMax, bMax)
    cmap = plt.get_cmap('coolwarm')

    for b, d, sE in zip(impactParams, directions, sEnd):
        chord = np.stack((pinholePos, pinholePos + sE * d))
        ax.plot(chord[:, 0], chord[:, 1], color=cmap(norm(b)), linewidth=2)

    ax.scatter(closestPoints[:, 0], closestPoints[:, 1], c=impactParams, cmap=cmap, norm=norm,
               edgecolor='k', s=60, zorder=3, label='Closest approach')

    ax.add_patch(plt.Circle((0, 0), plasmaRadius, edgecolor='k', facecolor='none', linewidth=3,
                            label=f'r = {plasmaRadius*1e2:.3g} cm'))

    ax.plot(*pinholePos, marker='s', color='k', ms=15, linestyle='none',
            label=f'DA{diodeArrayNum} pinhole')
    ax.plot(0, 0, marker='+', color='k', ms=20, mew=3, linestyle='none')

    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8)
    colorbar.set_label('Impact Parameter [m]')

    # Chords that miss the circle
    sIn, _ = chord_circle_intersections(pinholePos, directions, plasmaRadius)
    numMissed = np.sum(np.isnan(sIn))
    if numMissed > 0:
        # Bottom right, clear of the pinhole in the bottom left
        ax.text(0.98, 0.02, f'{numMissed} chord(s) miss the circle',
                transform=ax.transAxes, ha='right', va='bottom', fontsize='small')

    ax.set_aspect('equal')
    ax.set_xlabel(r'x (East $\rightarrow$ West) [m]')
    ax.set_ylabel(r'y (Bottom $\rightarrow$ Top) [m]')
    ax.set_title(f'Shot #{shotnum}', loc='right')
    ax.legend(loc='upper left', fontsize='small')

    if saveplot:
        save_plot(fig, f'{shotnum}_axuv_DA{diodeArrayNum}_chords.png')

    if showPlot:
        plt.show()

    return fig, ax

if __name__ == "__main__":

    # Simulation and shot to compare
    simName = 'nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW_ionDrrOn'
    shotnum = 261012050

    # Times to plot (in seconds)
    simTime = 3e-3
    expTime = 7.83e-3

    # AXUV chords with bad data in this shot (chord indices, see plot_axuv_raw_signals())
    axuvDetectorsToDrop = None

    # Raw signal of every chord against time, to check for bad chords
    plot_axuv_raw_signals(shotnum, diodeArrayNum=2, expTime=expTime, rateKHz=10.0,
                          detectorsToDrop=axuvDetectorsToDrop, saveplot=True, showPlot=False)

    plot_axuv_comparison(simName, shotnum, simTime, expTime,
                         signalDiodeArrayNum=2, rateKHz=10.0,
                         detectorsToDrop=axuvDetectorsToDrop, saveplot=True)
