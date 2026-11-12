"""
This script compares the end loss analyzer (ELA) density, electron temperature and ion
temperature profiles against the simulation at the mirror throat, where the simulated plasma is
Maxwellian.

There are two analyses of the ELA data (elaSources in ela.py), the CSV files of Aaron H. and the
NPZ files of Tucker P. Either or both can be drawn (sources=). Run as a script, this compares the
simulation with both.

The ELA density and radii are given at the midplane, mapped there with GDT scaling
(load_ela_data() in ela.py). They are taken on to the mirror throat-
    - The density is multiplied by 1/sqrt(e), the GDT drop from the midplane to the throat
      (midplane_to_throat_density() in ela.py).
    - The radius of each analyzer, and the ends of its error bar, are followed along their flux
      surfaces in the eqdsk of the IPS timestep that covers simTime, out to where each surface is
      closest to the axis (map_midplane_radius_to_throat() in cql3d_simulation.py).
    - The temperatures are left alone, since GDT takes the plasma to be isothermal along the flux
      tube.
Everything is plotted against the radius at the mirror throat.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from plotting import save_plot, densityColor, temperatureColor
from cql3d_simulation import (load_throat_density, load_throat_temperature, load_eqdsk_flux,
                              ips_timestep_index, map_midplane_radius_to_throat)
from ela import load_ela_data, midplane_to_throat_density

# How each ELA source is drawn. The time points of a source go from light (earliest) to dark
# (latest) in its colormap. Greys keeps Aaron H.'s data black-ish like the measurements of the
# other diagnostics, and Greens stays clear of the simulated density (C0) and temperatures (C1).
elaSourceStyles = {'csv': {'cmap': plt.cm.Greys, 'marker': 'o', 'linestyle': '-'},
                   'npz': {'cmap': plt.cm.Greens, 'marker': '^', 'linestyle': '--'}}

def plot_ela_comparison(simulationName, shotDatePrefix, simTime, expTime=None, sources=('csv',),
                        showPlot=True, saveplot=False, axes=None, simColor=None):
    """
    Plot the ELA density, electron temperature and ion temperature against the radius of each
    analyzer at the mirror throat, one curve per time point and source, with the simulated
    density, electron temperature and ion temperature at the mirror throat.

    The ELA density and radii are at the midplane in the data files. The density is mapped to the
    throat with GDT scaling (n_throat = n_mid / sqrt(e)), and the radii (and the ends of their
    error bars, drawn horizontally) along the flux surfaces of the eqdsk of the IPS timestep that
    covers simTime. The temperatures are not changed.

    Each source has its own colormap, marker and line style (elaSourceStyles), and its time points
    go from light (earliest) to dark (latest). If expTime is given, only the time point of each
    source closest to it is drawn, in the darkest color of its colormap (black for the CSV data).
    The simulated profiles are drawn over their whole radial extent, with a marker at the throat
    radius of each ELA of each source. ELAs inside the innermost simulated flux tube take its value.

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
    sources : tuple of str
        The ELA data to draw, keys of elaSources in ela.py- 'csv' (Aaron H.) and/or 'npz'
        (Tucker P.).
        Default is ('csv',).
    showPlot : bool
        Show the figure before returning. Set to False to add to the figure afterwards.
        Default is True.
    saveplot : bool
        Save the plot in config.plotDir as f'{shotDatePrefix}_{simulationName}_ela_comparison.png',
        with the sources appended (e.g. '_ela_comparison_csv_npz.png') unless sources is ('csv',).
        Default is False.
    axes : tuple of matplotlib.axes.Axes
        The three axes (density, T_e, T_i) to draw into instead of making a new figure, e.g. to
        place this plot in a larger figure.
        Default is None, which makes a new figure.
    simColor : str
        Color of every simulated profile.
        Default is None, which draws the density in densityColor and the temperatures in
        temperatureColor.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axs : tuple of matplotlib.axes.Axes
        The density, electron temperature and ion temperature axes.
    """

    sources = tuple(sources)
    if not sources:
        raise ValueError('sources must name at least one ELA source')

    # Flux surfaces of the eqdsk that the simulation used at simTime, to take the ELA radii to the
    # throat
    timestepIdx = ips_timestep_index(simulationName, simTime)
    Rmesh, Zmesh, psi = load_eqdsk_flux(simulationName, timestepIdx)

    def to_throat(rMid):
        return map_midplane_radius_to_throat(rMid.ravel(), Rmesh, Zmesh, psi)[0].reshape(rMid.shape)

    #### ELA data of each source, at the mirror throat

    elaData = []
    for source in sources:

        ela = load_ela_data(shotDatePrefix, source)
        times = ela['times']

        # Every [time, analyzer] array, kept 2D when only one time is drawn
        timeIdx = (slice(None) if expTime is None
                   else [np.argmin(np.abs(times - expTime))])

        radiusMid = ela['radius'][timeIdx]
        radius = to_throat(radiusMid)
        radiusLow = to_throat(radiusMid - ela['radiusErrMinus'][timeIdx])
        radiusHigh = to_throat(radiusMid + ela['radiusErrPlus'][timeIdx])

        cmap = elaSourceStyles[source]['cmap']
        colors = (cmap(np.linspace(0.4, 1.0, len(times))) if expTime is None
                  else [cmap(1.0)])

        elaData.append({
            'source': source,
            'label': ela['label'],
            'times': times[timeIdx],
            'windowHalfWidth': ela['windowHalfWidth'],
            'radius': radius,
            'radiusErr': np.array([radius - radiusLow, radiusHigh - radius]),
            'radiusHigh': radiusHigh,
            # GDT scaling of the density from the midplane to the throat
            'dens': midplane_to_throat_density(ela['dens'][timeIdx]),
            'densSigma': midplane_to_throat_density(ela['densSigma'][timeIdx]),
            'Te': ela['Te'][timeIdx], 'TeSigma': ela['TeSigma'][timeIdx],
            'Ti': ela['Ti'][timeIdx], 'TiSigma': ela['TiSigma'][timeIdx],
            'colors': colors})

    #### Simulated density at the mirror throat

    densThroat, rThroat, _, simTimes = load_throat_density(simulationName)

    simTimeIdx = np.argmin(np.abs(simTimes - simTime))
    densSim = densThroat[simTimeIdx]

    #### Simulated temperatures at the mirror throat

    TeThroat, TiThroat, rThroatTemp, tempSimTimes = load_throat_temperature(simulationName)

    tempSimTimeIdx = np.argmin(np.abs(tempSimTimes - simTime))

    # [keV] to [eV]
    TeSim = TeThroat[tempSimTimeIdx] * 1e3
    TiSim = TiThroat[tempSimTimeIdx] * 1e3

    if axes is None:
        # Wider with more than one source, to fit the legend beside the density panel
        fig = plt.figure(figsize=(12 if len(sources) == 1 else 17, 24), tight_layout=True)
        axDens = fig.add_subplot(3, 1, 1)
        axTe = fig.add_subplot(3, 1, 2, sharex=axDens)
        axTi = fig.add_subplot(3, 1, 3, sharex=axDens)
    else:
        axDens, axTe, axTi = axes
        fig = axDens.figure

    simDensColor = densityColor if simColor is None else simColor
    simTempColor = temperatureColor if simColor is None else simColor

    # (axis, ELA quantity, simulated radii, simulated profile, simulated time, color)
    panels = ((axDens, 'dens', rThroat, densSim, simTimes[simTimeIdx], simDensColor),
              (axTe, 'Te', rThroatTemp, TeSim, tempSimTimes[tempSimTimeIdx], simTempColor),
              (axTi, 'Ti', rThroatTemp, TiSim, tempSimTimes[tempSimTimeIdx], simTempColor))

    for ax, key, rSim, profileSim, tSim, color in panels:

        ax.plot(rSim, profileSim,
                color=color,
                linewidth=3,
                label=rf'CQL3D, t$_{{Sim}}$ = {tSim*1e3:.4g} ms')

        # Top of the axis, from the error bars (relim() ignores them) and the simulated values at
        # the ELAs, which can be above the measured ones
        top = 0

        for data in elaData:
            style = elaSourceStyles[data['source']]
            value, sigma = data[key], data[key + 'Sigma']

            for i in range(len(data['times'])):
                ax.errorbar(data['radius'][i], value[i],
                            xerr=data['radiusErr'][:, i],
                            yerr=sigma[i],
                            label=f'{data["label"]}, {data["times"][i]*1e3:g} ms',
                            color=data['colors'][i],
                            marker=style['marker'],
                            linestyle=style['linestyle'],
                            linewidth=2,
                            ms=10,
                            elinewidth=2,
                            capsize=4)

            # The analyzers do not move, so the radius at the first time is used
            simAtEla = np.interp(data['radius'][0], rSim, profileSim)
            ax.plot(data['radius'][0], simAtEla,
                    color=color,
                    linestyle='none',
                    marker='s',
                    ms=12)

            top = max(top, np.max(value + sigma), np.max(simAtEla))

        # Headroom for the legend
        ax.set_ylim(0, 1.3 * top)
        ax.grid(alpha=0.3)

    axDens.set_ylabel(r'n$_{throat}$ [m$^{-3}$]')
    axTe.set_ylabel(r'T$_e$ [eV]')
    axTi.set_ylabel(r'T$_i$ [eV]')
    axTi.set_xlabel(r'Radius at the mirror throat, r$_{throat}$ [m]')

    # The simulation runs well past the outermost ELA, so the axis is kept on the analyzers
    axDens.set_xlim(0, 1.1 * max(np.max(data['radiusHigh']) for data in elaData))
    # Few enough ticks that their labels do not run into each other on a wide figure
    axDens.xaxis.set_major_locator(MaxNLocator(5))

    if len(sources) == 1:
        axDens.legend(loc='lower center', fontsize='x-small', ncol=3)
    else:
        # Too many curves to fit inside the panel
        axDens.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='x-small')

    # One time window in the title if every source has the same, otherwise one per source
    halfWidths = list(dict.fromkeys(data['windowHalfWidth'] for data in elaData))
    windows = ', '.join(rf'$\pm${w*1e3:g} ms' for w in halfWidths)
    axDens.set_title(f'ELA ({" vs. ".join(data["label"] for data in elaData)}), '
                     f'{shotDatePrefix}, {windows} window, '
                     r'n$_{throat}$ = n$_{mid}$/$\sqrt{e}$ (GDT)', fontsize='small')

    if saveplot:
        suffix = '' if sources == ('csv',) else '_' + '_'.join(sources)
        save_plot(fig, f'{shotDatePrefix}_{simulationName}_ela_comparison{suffix}.png')

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

    # ELA data to compare with, keys of elaSources in ela.py- 'csv' (Aaron H.), 'npz' (Tucker P.)
    sources = ('csv', 'npz')

    plot_ela_comparison(simName, shotDatePrefix, simTime, sources=sources, saveplot=True)
