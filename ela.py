"""
This script loads the end loss analyzer (ELA) profiles.

The data is read from two files in config.elaDataDir, one with the density and one with the
temperatures, each holding the profile across the analyzers at several times. The keys of each
time are prefixed f'surrogate_t_{time}ms_', e.g. 'surrogate_t_4ms_density_m3'.
"""

import numpy as np

import config

# The ELAs sit in the expander where |B| is this many times lower than at the mirror throat
elaMirrorRatio = 40.0

def map_throat_to_ela(rThroat, densThroat, mirrorRatio=elaMirrorRatio):
    """
    Map a density profile at the mirror throat out to the ELAs in the expander.

    The flux through each flux tube is conserved, B * A = const, so going from the throat to the
    ELAs the cross-sectional area of the tube grows by the mirror ratio B_throat / B_ELA. Its
    radius grows by the square root of that and, with the particle flux along the tube
    conserved, the density drops by the same factor as the area grows.

    Parameters
    ----------
    rThroat : np.array
        1D array of the radius of each point at the mirror throat. [m]
    densThroat : np.array
        Density at each point of rThroat, with r as the last axis. [m^-3]
    mirrorRatio : float
        B_throat / B_ELA.
        Default is elaMirrorRatio.

    Returns
    -------
    rEla : np.array
        1D array of the radius of each point at the ELAs. [m]
    densEla : np.array
        Density at each point of rEla. [m^-3]
    """

    return rThroat * np.sqrt(mirrorRatio), densThroat / mirrorRatio

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
        2D array of the radius of each analyzer, [time, analyzer]. [m]
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
        2D array of the radius of each analyzer, [time, analyzer]. [m]
    analyzerNum : np.array
        2D array of the number of each analyzer, [time, analyzer].
    dens : np.array
        2D array of the density, [time, analyzer]. [m^-3]
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
        2D array of the radius of each analyzer, [time, analyzer]. [m]
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
