"""
This script loads the Thomson scattering electron density profile.

The data is read from a data.yaml file in the mirror_fit input format (config.thomsonDataPath).
"""

import yaml
import numpy as np

import config

def load_thomson_density(dataPath=None):
    """
    Load the Thomson scattering density profile from a mirror_fit data.yaml file.

    Parameters
    ----------
    dataPath : str
        Path to the data.yaml file.
        Default is None, which loads config.thomsonDataPath.

    Returns
    -------
    radius : np.array
        Radius of each measurement, sorted in ascending order. [m]
        The file stores R = |x|, so all the radii are positive.
    dens : np.array
        Measured electron density. [m^-3]
    densSigma : np.array
        1 sigma error on the measured electron density. [m^-3]
    time : float
        Time of the measurement. [s]
    """

    if dataPath is None:
        dataPath = config.thomsonDataPath

    with open(dataPath, 'r') as f:
        data = yaml.safe_load(f)

    thomson = data['diagnostics']['thomson_density']

    radius = np.array([rz[0] for rz in thomson['RZ']])
    # Only the first time is used
    dens = np.array(thomson['measurements'][0], dtype=float)
    densSigma = np.array(thomson['sigma_measurements'][0], dtype=float)
    time = float(data['times'][0])

    # Sort based on the radius
    sortIdx = np.argsort(radius)

    return radius[sortIdx], dens[sortIdx], densSigma[sortIdx], time

def load_thomson_temperature(dataPath=None):
    """
    Load the Thomson scattering temperature profile from a mirror_fit data.yaml file.

    Parameters
    ----------
    dataPath : str
        Path to the data.yaml file.
        Default is None, which loads config.thomsonDataPath.

    Returns
    -------
    radius : np.array
        Radius of each measurement, sorted in ascending order. [m]
        The file stores R = |x|, so all the radii are positive.
    dens : np.array
        Measured electron temperature. [eV]
    densSigma : np.array
        1 sigma error on the measured electron temperature. [eV]
    time : float
        Time of the measurement. [s]
    """

    if dataPath is None:
        dataPath = config.thomsonDataPath

    with open(dataPath, 'r') as f:
        data = yaml.safe_load(f)

    thomson = data['diagnostics']['thomson_temperature']

    radius = np.array([rz[0] for rz in thomson['RZ']])
    # Only the first time is used
    dens = np.array(thomson['measurements'][0], dtype=float)
    densSigma = np.array(thomson['sigma_measurements'][0], dtype=float)
    time = float(data['times'][0])

    # Sort based on the radius
    sortIdx = np.argsort(radius)

    return radius[sortIdx], dens[sortIdx], densSigma[sortIdx], time
