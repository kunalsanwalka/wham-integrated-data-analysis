"""
This script draws the lines of sight of the SEE array and the AXUV array on the (r, z) grid of a
WarpX simulation, over its density.

Each line of sight is a straight line in 3D. In (r, z), with r = sqrt(x^2 + y^2), one that does
not pass through the axis traces a hyperbola, r^2 = b^2 + (z - z_b)^2 / tan^2(alpha), with its
vertex at the closest approach to the axis, r = |b|.
    - SEE: the lines of sight cross the plasma at an angle to the axis, so they are hyperbolas.
      The part in 'los_rz' (what the CQL3D comparison integrates over) is drawn solid, and the rest
      of the line out to the edge of the WarpX grid (what the WarpX comparison integrates over,
      see full_line_of_sight() in see_array.py) dashed.
    - AXUV: the chords lie in the midplane cross-section (z = 0), so the hyperbola is degenerate.
      Each chord runs along z = 0 from the wall in to r = |b| and back out, so they all fall on the
      z = 0 line. The closest approach of each chord is marked.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from plotting import save_plot, densityColor, halphaColor
from see_array import load_detector_dictionary, full_line_of_sight
from axuv import load_axuv_impact_params
from warpx_simulation import load_warpx_fields, profile_at_points, ionDensityField

def plot_warpx_lines_of_sight(shotnum, runFile='fields_rtz.npz', fieldName=ionDensityField,
                              signalDiodeArrayNum=2, decades=3,
                              showPlot=True, saveplot=False, ax=None):
    """
    Plot the SEE and AXUV lines of sight on the (r, z) grid of a WarpX simulation.

    Parameters
    ----------
    shotnum : int
        The shot number the AXUV impact parameters are taken from.
    runFile : str
        The WarpX field dump, see load_warpx_fields().
        Default is 'fields_rtz.npz'.
    fieldName : str
        The field drawn underneath, from load_warpx_fields().
        Default is ionDensityField, the density the SEE array is compared against.
    signalDiodeArrayNum : int
        The AXUV diode array whose chords are drawn.
        Default is 2, the H-alpha filtered array (DIODEARRAY2).
    decades : float
        Number of decades below its peak the field is drawn down to.
        Default is 3.
    showPlot : bool
        Show the figure before returning.
        Default is True.
    saveplot : bool
        Save the plot in config.plotDir as f'{shotnum}_warpx_{runName}_lines_of_sight.png'.
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

    fields = load_warpx_fields(runFile)
    r, z = fields['r'], fields['z']

    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 9), tight_layout=True)
    else:
        fig = ax.figure

    #### WarpX field

    field = fields[fieldName]
    norm = LogNorm(np.max(field) / 10**decades, np.max(field))
    mesh = ax.pcolormesh(z, r, np.ma.masked_less_equal(field, 0), shading='auto',
                         cmap='Greys', norm=norm)

    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    colorbar.set_label({'n_ions_total': r'n$_i$ + n$_{fast}$ [m$^{-3}$]',
                        'n_electrons': r'n$_e$ [m$^{-3}$]',
                        'n_neutrals': r'n$_0$ [m$^{-3}$]'}.get(fieldName, fieldName))

    #### SEE lines of sight

    zExtent = 0
    for i, detDict in enumerate(load_detector_dictionary()):

        losFull, _ = full_line_of_sight(detDict, r[-1], z[-1])
        losRZ = detDict['los_rz'] / 1e3

        ax.plot(losFull[:, 1], losFull[:, 0],
                color=densityColor, linewidth=1.5, linestyle='dashed',
                label='SEE, out to the WarpX grid edge' if i == 0 else None)
        ax.plot(losRZ[1], losRZ[0],
                color=densityColor, linewidth=2.5,
                label="SEE, 'los_rz' (CQL3D comparison)" if i == 0 else None)

        # Only the part of the line through the drawn plasma sets the axis limits
        inPlasma = profile_at_points(fields, fieldName, losFull[:, 0], losFull[:, 1]) > norm.vmin
        if np.any(inPlasma):
            zExtent = max(zExtent, np.max(np.abs(losFull[inPlasma, 1])))

    #### AXUV chords

    impactParams = load_axuv_impact_params(shotnum, signalDiodeArrayNum)

    # In the z = 0 plane, from the closest approach out to the edge of the grid
    ax.plot([0, 0], [np.min(np.abs(impactParams)), r[-1]],
            color=halphaColor, linewidth=2.5,
            label=f'AXUV DA{signalDiodeArrayNum} chords (all at z = 0)')
    ax.plot(np.zeros(len(impactParams)), np.abs(impactParams),
            color=halphaColor, linestyle='none', marker='o', ms=8, mec='k',
            label='AXUV closest approach, r = |b|')

    ax.set_xlim(-1.1 * zExtent, 1.1 * zExtent)
    ax.set_ylim(0, r[-1])
    ax.set_aspect('equal')

    ax.set_xlabel('z [m]')
    ax.set_ylabel('r [m]')
    ax.set_title(f'WarpX {fields["runName"]}, Shot #{shotnum}', loc='right', fontsize='small')
    ax.legend(loc='upper left', fontsize='x-small')

    if saveplot:
        save_plot(fig, f'{shotnum}_warpx_{fields["runName"]}_lines_of_sight.png')

    if showPlot:
        plt.show()

    return fig, ax

if __name__ == "__main__":

    # WarpX run and the shot the AXUV impact parameters are taken from
    runFile = 'fields_rtz.npz'
    shotnum = 260709060

    plot_warpx_lines_of_sight(shotnum, runFile=runFile, saveplot=True)
