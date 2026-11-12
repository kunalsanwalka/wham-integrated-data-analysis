"""
Loads a WarpX simulation of WHAM for comparison against the diagnostics.

WarpX dumps its fields on a cylindrical (r, theta, z) grid at a few times, as
    [config.warpxDataDir]/[runFile].npz
with one [t x r x theta x z] array per field (n_electrons, T_electrons, n_neutrals, n_ions,
n_fast_ions, phi) and the 1D coordinates r, theta, z [m, rad, m] and t [s].

The dumps only cover a few microseconds, much less than the time resolution of the diagnostics,
so every field is averaged over time and theta (load_warpx_fields()). This leaves (r, z) maps,
the same shape as the CQL3D density, so the synthetic diagnostics can take them through an
(r, z) interpolator (field_interpolator()).

Which density each diagnostic is compared against-
    - Thomson scattering:    n_electrons             (thomsonDensityField)
    - SEE array and ELA:     n_ions + n_fast_ions    (ionDensityField)

WarpX has no equilibrium, so the mirror throat is taken from the eqdsk of an IPS simulation of
the same machine (throat_positions()). The ELAs have not moved, so their throat radii are the ones
the IPS flux mapping gives.
"""

import functools
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import config
from cql3d_simulation import ips_timestep_index, load_eqdsk_flux, map_midplane_radius_to_throat

# Fields stored in the WarpX dumps
warpxFieldNames = ('n_electrons', 'T_electrons', 'n_neutrals', 'n_ions', 'n_fast_ions', 'phi')

# Thermal plus fast ions, added by load_warpx_fields()
totalIonField = 'n_ions_total'

# Density each diagnostic is compared against
thomsonDensityField = 'n_electrons'
ionDensityField = totalIonField

def warpx_file_path(runFile):
    """
    Path of a WarpX field dump.

    Parameters
    ----------
    runFile : str
        Name of the .npz file inside config.warpxDataDir, e.g. 'fields_rtz.npz', or an
        absolute path.

    Returns
    -------
    filePath : str
        Path of the .npz file.
    """

    if os.path.isabs(runFile):
        return runFile

    return config.warpxDataDir + runFile

@functools.lru_cache(maxsize=4)
def load_warpx_fields(runFile='fields_rtz.npz', weightTemperature=True):
    """
    Load a WarpX field dump and average every field over time and theta.

    The electron temperature is averaged weighted by the electron density,
        T_e(r, z) = sum(n_e T_e) / sum(n_e)
    over the dumps and theta. This is the temperature the electrons carry on average, and keeps
    cells that hold no electrons in some dumps (where WarpX gives T_e = 0) from pulling it down.
    At the midplane it is within 1% of the plain average inside r = 0.13 m, and within 5% out to
    r = 0.2 m. Every other field is a plain average.

    The result is cached for the session, so the dictionary returned is shared between callers
    and should not be changed in place.

    Parameters
    ----------
    runFile : str
        Name of the .npz file inside config.warpxDataDir, or an absolute path.
        Default is 'fields_rtz.npz'.
    weightTemperature : bool
        Weight the electron temperature by the electron density. Set to False for a plain
        average.
        Default is True.

    Returns
    -------
    fields : dict
        'r'            : [r], radius of the grid. [m]
        'z'            : [z], axial position of the grid. [m]
        'n_electrons'  : [r x z], electron density. [m^-3]
        'T_electrons'  : [r x z], electron temperature. [eV]
        'n_neutrals'   : [r x z], neutral density. [m^-3]
        'n_ions'       : [r x z], thermal ion density. [m^-3]
        'n_fast_ions'  : [r x z], fast ion density. [m^-3]
        'n_ions_total' : [r x z], n_ions + n_fast_ions. [m^-3]
        'phi'          : [r x z], electrostatic potential. [V]
        'times'        : [Dump], time of each dump averaged over. [s]
        'runName'      : name of the run, the file name without .npz, for labels and plot names.
    """

    filePath = warpx_file_path(runFile)

    fields = {}

    with np.load(filePath) as data:

        fields['r'] = data['r']
        fields['z'] = data['z']
        fields['times'] = data['t']

        # Summed in float64, as the dumps are float32
        ne = data['n_electrons'].astype(np.float64)

        for name in warpxFieldNames:

            if name == 'T_electrons' and weightTemperature:
                neSum = np.sum(ne, axis=(0, 2))
                neTeSum = np.sum(ne * data['T_electrons'], axis=(0, 2))
                # Where there are never any electrons T_e is 0 in every dump
                fields[name] = np.divide(neTeSum, neSum, out=np.zeros_like(neSum), where=neSum > 0)
            elif name == 'n_electrons':
                fields[name] = np.mean(ne, axis=(0, 2))
            else:
                fields[name] = np.mean(data[name], axis=(0, 2), dtype=np.float64)

    fields[totalIonField] = fields['n_ions'] + fields['n_fast_ions']

    fields['runName'] = os.path.splitext(os.path.basename(filePath))[0]

    return fields

def field_interpolator(fields, fieldName):
    """
    Make an (r, z) interpolation function of one averaged field.

    The function takes the same input as the CQL3D density interpolators, so it can be passed to
    the synthetic diagnostics (e.g. synthetic_see_detector() in see_array.py). Interpolation is
    linear, |r| is used so negative radii are mirrored about the axis, and points outside the
    WarpX grid get 0.

    Parameters
    ----------
    fields : dict
        The averaged fields, from load_warpx_fields().
    fieldName : str
        The field to interpolate, e.g. ionDensityField.

    Returns
    -------
    interpFunc : function
        Takes an [N x 2] array of (r, z) points [m] and returns the field at each point.
    """

    gridInterp = RegularGridInterpolator((fields['r'], fields['z']), fields[fieldName],
                                         method='linear', bounds_error=False, fill_value=0.0)

    def interpFunc(points):

        points = np.atleast_2d(np.asarray(points, dtype=float))

        return gridInterp(np.column_stack((np.abs(points[:, 0]), points[:, 1])))

    return interpFunc

def profile_at_points(fields, fieldName, r, z):
    """
    Evaluate one averaged field at a set of (r, z) points.

    Parameters
    ----------
    fields : dict
        The averaged fields, from load_warpx_fields().
    fieldName : str
        The field to evaluate.
    r : np.array
        Radius of each point. [m]
    z : np.array
        Axial position of each point. [m]

    Returns
    -------
    values : np.array
        The field at each point, with the shape of r.
    """

    r = np.asarray(r, dtype=float)
    z = np.broadcast_to(np.asarray(z, dtype=float), r.shape)

    values = field_interpolator(fields, fieldName)(np.column_stack((r.ravel(), z.ravel())))

    return values.reshape(r.shape)

def midplane_profiles(fields):
    """
    The averaged fields at the midplane (z = 0) against radius.

    The z grid has no point at 0, so the two points either side of it are interpolated between.

    Parameters
    ----------
    fields : dict
        The averaged fields, from load_warpx_fields().

    Returns
    -------
    profiles : dict
        'r' : [r], midplane radius. [m]
        One [r] array per field of load_warpx_fields() (e.g. 'n_electrons', 'T_electrons',
        'n_neutrals', 'n_ions_total'), at the midplane.
    """

    r = fields['r']

    profiles = {'r': r}

    for name in warpxFieldNames + (totalIonField,):
        profiles[name] = profile_at_points(fields, name, r, np.zeros_like(r))

    return profiles

def throat_positions(ipsSimulationName, ipsSimTime, rMid, end=1):
    """
    Map midplane radii to the mirror throat with the eqdsk of an IPS simulation.

    WarpX has no equilibrium, so the flux surfaces are taken from the eqdsk of the IPS timestep
    that covers ipsSimTime (see map_midplane_radius_to_throat() in cql3d_simulation.py). The
    throat is where each flux surface is closest to the axis, on the z > 0 half of the machine.

    The surface through r = 0 is the axis, which has no closest point. It is given the throat z
    of the nearest surface off the axis.

    Parameters
    ----------
    ipsSimulationName : str
        The name of the IPS simulation the eqdsk is taken from.
    ipsSimTime : float
        The IPS simulation time. [s]
        The eqdsk of the IPS timestep that covers it is used.
    rMid : np.array
        1D array of the midplane radii to map. [m]
    end : int
        +1 for the throat at z > 0, -1 for the one at z < 0 (the eqdsk z > 0 throat mirrored).
        The IPS mapping, and so the ELA comparison of the IPS runs, uses z > 0.
        Default is 1.

    Returns
    -------
    rThroat : np.array
        1D array of the radius of each flux surface at the mirror throat. [m]
        NaN for surfaces that leave the eqdsk grid.
    zThroat : np.array
        1D array of the axial position of the mirror throat on each flux surface. [m]
    """

    if end not in (1, -1):
        raise ValueError(f'end must be +1 or -1, not {end}')

    timestepIdx = ips_timestep_index(ipsSimulationName, ipsSimTime)
    Rmesh, Zmesh, psi = load_eqdsk_flux(ipsSimulationName, timestepIdx)

    rMid = np.atleast_1d(np.asarray(rMid, dtype=float))
    rThroat, zThroat = map_midplane_radius_to_throat(rMid, Rmesh, Zmesh, psi)

    onAxis = rMid == 0
    offAxis = ~onAxis & np.isfinite(zThroat)
    if np.any(onAxis) and np.any(offAxis):
        zThroat[onAxis] = zThroat[offAxis][np.argmin(rMid[offAxis])]

    return rThroat, end * zThroat

def throat_profiles(fields, ipsSimulationName, ipsSimTime, rMid=None, end=1):
    """
    The averaged fields at the mirror throat against the radius there.

    The throat is taken from the eqdsk of an IPS simulation (throat_positions()), and each field is
    interpolated onto the (r, z) of the throat on every flux surface.

    Parameters
    ----------
    fields : dict
        The averaged fields, from load_warpx_fields().
    ipsSimulationName : str
        The name of the IPS simulation the eqdsk is taken from.
    ipsSimTime : float
        The IPS simulation time. [s]
    rMid : np.array
        1D array of the midplane radii of the flux surfaces to follow. [m]
        Default is None, which uses 200 surfaces from the axis to the edge of the WarpX grid.
        Surfaces that leave the eqdsk grid are dropped.
    end : int
        +1 for the throat at z > 0, -1 for the one at z < 0.
        Default is 1.

    Returns
    -------
    profiles : dict
        'rMid'    : [Surface], midplane radius of each flux surface. [m]
        'r'       : [Surface], radius of each flux surface at the throat. [m]
        'z'       : [Surface], axial position of the throat on each flux surface. [m]
        One [Surface] array per field of load_warpx_fields(), at the throat.
    """

    if rMid is None:
        rMid = np.linspace(0, fields['r'][-1], 200)

    rThroat, zThroat = throat_positions(ipsSimulationName, ipsSimTime, rMid, end=end)

    keep = np.isfinite(rThroat) & np.isfinite(zThroat)

    profiles = {'rMid': np.asarray(rMid, dtype=float)[keep],
                'r': rThroat[keep],
                'z': zThroat[keep]}

    for name in warpxFieldNames + (totalIonField,):
        profiles[name] = profile_at_points(fields, name, profiles['r'], profiles['z'])

    return profiles
