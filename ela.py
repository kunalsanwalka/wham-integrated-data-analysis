"""
This script loads the end loss analyzer (ELA) profiles.

The data is read from two files in config.elaDataDir, one with the density and one with the
temperatures, each holding the profile across the analyzers at several times. The keys of each
time are prefixed f'surrogate_t_{time}ms_', e.g. 'surrogate_t_4ms_density_m3'.

The density and the radius of each analyzer in these files have already been mapped back to the
midplane with GDT scaling. midplane_to_throat_density() takes the density on to the mirror throat,
and cql3d_simulation.map_midplane_radius_to_throat() does the same for the radius. GDT takes the
plasma to be isothermal along the flux tube, so the temperatures are the same at the midplane and
the throat.
"""

import numpy as np

import config

# GDT scaling of the density from the midplane to the mirror throat, n_throat / n_mid. The
# ambipolar potential drop from the midplane to the throat, where the outflow reaches the sound
# speed, is T_e/2e, so the Boltzmann electrons drop by exp(-1/2).
gdtThroatDensityFactor = 1 / np.sqrt(np.e)

def midplane_to_throat_density(densMid, factor=gdtThroatDensityFactor):
    """
    Map a density at the midplane to the mirror throat with GDT scaling, n_throat = n_mid / sqrt(e).

    Parameters
    ----------
    densMid : np.array
        Density (or its error) at the midplane. [m^-3]
    factor : float
        n_throat / n_mid.
        Default is gdtThroatDensityFactor = 1/sqrt(e).

    Returns
    -------
    densThroat : np.array
        Density (or its error) at the mirror throat. [m^-3]
    """

    return densMid * factor

def _load_ela_file(filename, quantityKeys, dataDir=None):
    """
    Load the radius, analyzer number and the requested quantities at every time of one ELA file.

    Parameters
    ----------
    filename : str
        Name of the file inside dataDir.
    quantityKeys : list of str
        Keys of the quantities to load, without the f'surrogate_t_{time}ms_' prefix.
    dataDir : str
        Directory of the file.
        Default is None, which uses config.elaDataDir.

    Returns
    -------
    times : np.array
        1D array of the time points. [s]
    radius : np.array
        2D array of the radius of each analyzer mapped to the midplane, [time, analyzer]. [m]
    analyzerNum : np.array
        2D array of the number of each analyzer, [time, analyzer].
    quantities : dict
        2D array of each quantity in quantityKeys, [time, analyzer].
    windowHalfWidth : float
        Half width of the time window around each time point. [s]
    """

    if dataDir is None:
        dataDir = config.elaDataDir

    data = np.load(dataDir + filename)

    timesMs = data['requested_time_points_ms']
    # e.g. 4.0 ms -> 'surrogate_t_4ms_'
    prefixes = [f'surrogate_t_{t:g}ms_' for t in timesMs]

    radius = np.array([data[prefix + 'radius_cm'] for prefix in prefixes])
    analyzerNum = np.array([data[prefix + 'analyzer_number'] for prefix in prefixes])
    quantities = {key: np.array([data[prefix + key] for prefix in prefixes])
                  for key in quantityKeys}

    # [ms] to [s] and [cm] to [m]
    return (timesMs*1e-3, radius*1e-2, analyzerNum, quantities,
            float(data['window_half_width_ms'])*1e-3)

def load_ela_density(shotDatePrefix, dataDir=None):
    """
    Load the ELA density profile at every time in f'{shotDatePrefix}_ELA_Density.npz'.

    Parameters
    ----------
    shotDatePrefix : int
        Date prefix of the shots, e.g. 260709.
    dataDir : str
        Directory of the file.
        Default is None, which uses config.elaDataDir.

    Returns
    -------
    times : np.array
        1D array of the time points. [s]
    radius : np.array
        2D array of the radius of each analyzer mapped to the midplane, [time, analyzer]. [m]
    analyzerNum : np.array
        2D array of the number of each analyzer, [time, analyzer].
    dens : np.array
        2D array of the density mapped to the midplane, [time, analyzer]. [m^-3]
        midplane_to_throat_density() maps it on to the mirror throat.
    densSigma : np.array
        2D array of the 1 sigma error on the density, [time, analyzer]. [m^-3]
    windowHalfWidth : float
        Half width of the time window around each time point. [s]
    """

    times, radius, analyzerNum, quantities, windowHalfWidth = _load_ela_file(
        f'{shotDatePrefix}_ELA_Density.npz', ['density_m3', 'density_error_m3'], dataDir)

    return (times, radius, analyzerNum,
            quantities['density_m3'], quantities['density_error_m3'], windowHalfWidth)

def load_ela_temperature(shotDatePrefix, dataDir=None):
    """
    Load the ELA electron and ion temperature profiles at every time in
    f'{shotDatePrefix}_ELA_Temp.npz'.

    Parameters
    ----------
    shotDatePrefix : int
        Date prefix of the shots, e.g. 260709.
    dataDir : str
        Directory of the file.
        Default is None, which uses config.elaDataDir.

    Returns
    -------
    times : np.array
        1D array of the time points. [s]
    radius : np.array
        2D array of the radius of each analyzer mapped to the midplane, [time, analyzer]. [m]
    analyzerNum : np.array
        2D array of the number of each analyzer, [time, analyzer].
    Te : np.array
        2D array of the electron temperature, [time, analyzer]. [eV]
    TeSigma : np.array
        2D array of the 1 sigma error on the electron temperature, [time, analyzer]. [eV]
    Ti : np.array
        2D array of the ion temperature, [time, analyzer]. [eV]
    TiSigma : np.array
        2D array of the 1 sigma error on the ion temperature, [time, analyzer]. [eV]
    windowHalfWidth : float
        Half width of the time window around each time point. [s]
    """

    times, radius, analyzerNum, quantities, windowHalfWidth = _load_ela_file(
        f'{shotDatePrefix}_ELA_Temp.npz', ['Te_eV', 'Te_err_eV', 'Ti_eV', 'Ti_err_eV'], dataDir)

    return (times, radius, analyzerNum,
            quantities['Te_eV'], quantities['Te_err_eV'],
            quantities['Ti_eV'], quantities['Ti_err_eV'], windowHalfWidth)
