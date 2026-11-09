"""
This script loads the plasma radius measured by the AXUV diode arrays, the line-integrated
signal of each chord, and the line of sight of each chord.

The AXUV data is processed into NPZ files by axuv_data_output.py in the shinethru repository
(config.axuvScriptDir), which is run automatically for a shot that has not been processed yet.
"""

import sys
import subprocess
import numpy as np

from pathlib import Path
from scipy.stats import binned_statistic

import config

# Position of the pinhole of each diode array in the midplane cross-section. [m]
# x runs from east to west and y from bottom to top, with the machine axis at (0, 0).
pinholePositions = {2: np.array([-0.33, -0.19])}

def average_down(timeArr, values, rateKHz):
    """
    Average the values into windows of width 1/rateKHz.

    This is the same averaging as is done in plot_plasma_radius.py in config.axuvScriptDir.

    Parameters
    ----------
    timeArr : np.array
        1D array of the times of the values. [s]
    values : np.array
        1D array of the values being averaged down.
    rateKHz : float
        Width of the averaging window. [kHz]

    Returns
    -------
    centres : np.array
        1D array of the centres of the averaging windows. [s]
    means : np.array
        1D array of the values averaged within each window.
    """

    windowWidth = 1 / (rateKHz * 1e3)
    bins = np.arange(timeArr[0], timeArr[-1] + windowWidth, windowWidth)

    means, edges, _ = binned_statistic(timeArr, values, statistic='mean', bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Empty bins come back as NaN, which would break the interpolation
    keep = np.isfinite(means)

    return centres[keep], means[keep]

def axuv_data_files(shotnum, diodeArrayNum):
    """
    Find the AXUV NPZ files in the data directory for the given shot and diode array.

    Parameters
    ----------
    shotnum : int
        The shot number.
    diodeArrayNum : int
        The diode array, i.e. 1 for DIODEARRAY1.

    Returns
    -------
    filenameList : list of Path
        The matching files, with the original before any duplicates of it. Empty if the AXUV
        data for the shot has not been generated yet.
    """

    # The data directory can hold duplicates of the same diode (e.g. ' copy' files), which
    # sort after the original they copy
    return sorted(Path(config.axuvDataDir).glob(f'AXUV_results*{shotnum}*DA{diodeArrayNum}*.npz'),
                  key=lambda p: ('copy' in p.stem, p.name))

def load_axuv_radius(shotnum, diodeArrayNum, rateKHz=10.0, tMin=None, tMax=None):
    """
    Load the plasma radius vs. time measured by AXUV for the given shot.

    The radius is the RMS radius stored under the `radius_cm` key of the AXUV NPZ files
    (see plot_plasma_radius.py in config.axuvScriptDir). It is averaged down from the native (~1 MHz)
    rate to suppress the noise.

    Parameters
    ----------
    shotnum : int
        The shot number.
    diodeArrayNum : int
        The diode array the radius is taken from, i.e. 1 for DIODEARRAY1.
    rateKHz : float
        Rate the native data is averaged down to. [kHz]
        Default is 10.
    tMin : float
        Start of the experimental time window the data is trimmed to. [s]
        Default is None, which keeps the data from the start of the shot.
    tMax : float
        End of the experimental time window the data is trimmed to. [s]
        Default is None, which keeps the data to the end of the shot.

    Returns
    -------
    axuvTimes : np.array
        1D array of the times of the averaged down radius. [s]
    axuvRadius : np.array
        1D array of the plasma radius. [m]
    """

    filenameList = axuv_data_files(shotnum, diodeArrayNum)

    if len(filenameList) == 0:
        raise FileNotFoundError(f'No AXUV data for shot {shotnum}, DA{diodeArrayNum} in {config.axuvDataDir}')

    with np.load(filenameList[0], allow_pickle=True) as data:
        axuvTimes = data[f's{shotnum}_time_s']
        # [cm]
        axuvRadius = data[f's{shotnum}_radius_cm']

    axuvTimes, axuvRadius = average_down(axuvTimes, axuvRadius, rateKHz)

    # Convert to m to match the simulation
    axuvRadius = axuvRadius / 1e2

    # Trim to the part of the shot the radius is well behaved in
    keep = np.ones(len(axuvTimes), dtype=bool)
    if tMin is not None:
        keep &= axuvTimes >= tMin
    if tMax is not None:
        keep &= axuvTimes <= tMax

    if not np.any(keep):
        raise ValueError(f'No AXUV data between tMin={tMin} s and tMax={tMax} s for shot {shotnum}')

    axuvTimes = axuvTimes[keep]
    axuvRadius = axuvRadius[keep]

    return axuvTimes, axuvRadius

def generate_axuv_data(shotnum, diodeArrayNum):
    """
    Process the AXUV data of the given shot if it is not in config.axuvDataDir yet.

    This runs axuv_data_output.py from config.axuvScriptDir, which saves its results to
    config.axuvDataDir.

    The data is checked for explicitly rather than by catching a FileNotFoundError from
    load_axuv_radius(), as the callers also raise that for a missing simulation file (e.g. a
    misspelt simulation name), which would then rerun the AXUV script on every call.

    Parameters
    ----------
    shotnum : int
        The shot number.
    diodeArrayNum : int
        The diode array, i.e. 1 for DIODEARRAY1.
    """

    if len(axuv_data_files(shotnum, diodeArrayNum)) > 0:
        return

    print(f'No AXUV data present for shot {shotnum}, generating it.')

    subprocess.run([sys.executable, 'axuv_data_output.py', '-s', f'{shotnum}'],
                   cwd=config.axuvScriptDir, check=True)

    return

def load_axuv_signal(shotnum, diodeArrayNum, expTime, rateKHz=10.0):
    """
    Load the line-integrated signal of every chord of an AXUV diode array at one time.

    The raw chord signals (signal_raw_nA in the AXUV NPZ files) are averaged over a window of
    width 1/rateKHz centred on expTime to suppress the noise on the native (~1 MHz) data rate.
    This is the same window width the plasma radius is averaged down with in load_axuv_radius().

    Parameters
    ----------
    shotnum : int
        The shot number.
    diodeArrayNum : int
        The diode array, i.e. 2 for DIODEARRAY2 (the H-alpha filtered array).
    expTime : float
        The experimental time at the centre of the averaging window. [s]
    rateKHz : float
        Sets the width of the averaging window, 1/rateKHz. [kHz]
        Default is 10.

    Returns
    -------
    impactParams : np.array
        1D array of the impact parameter of each chord, sorted in ascending order. [m]
    signal : np.array
        1D array of the line-integrated signal of each chord, averaged over the window. [nA]
    signalSigma : np.array
        1D array of the standard deviation of each chord's signal within the window. [nA]
        This is the spread of the samples, i.e. the noise and fluctuation level, rather than
        the error on the mean.
    description : str
        Description of the diode array, e.g. its filter.
    """

    # Process the shot if it has not been processed yet
    generate_axuv_data(shotnum, diodeArrayNum)

    filenameList = axuv_data_files(shotnum, diodeArrayNum)

    with np.load(filenameList[0], allow_pickle=True) as data:
        timeArr = data[f's{shotnum}_time_s']
        impactParams = data[f's{shotnum}_impact_b_raw_m']
        # [Chord x Time]
        signalRaw = data[f's{shotnum}_signal_raw_nA']
        description = str(data[f's{shotnum}_diode_description'])

    # Averaging window centred on expTime
    windowWidth = 1 / (rateKHz * 1e3)
    inWindow = np.abs(timeArr - expTime) <= windowWidth / 2

    if not np.any(inWindow):
        raise ValueError(f'No AXUV data within {windowWidth*1e3:.3g} ms of {expTime*1e3:.4g} ms '
                         f'for shot {shotnum}, DA{diodeArrayNum}')

    signal = np.mean(signalRaw[:, inWindow], axis=1)
    signalSigma = np.std(signalRaw[:, inWindow], axis=1)

    # Sort based on the impact parameter
    sortIdx = np.argsort(impactParams)

    return impactParams[sortIdx], signal[sortIdx], signalSigma[sortIdx], description

def load_axuv_impact_params(shotnum, diodeArrayNum):
    """
    Load the impact parameter of every chord of an AXUV diode array.

    Parameters
    ----------
    shotnum : int
        The shot number.
    diodeArrayNum : int
        The diode array, i.e. 2 for DIODEARRAY2 (the H-alpha filtered array).

    Returns
    -------
    impactParams : np.array
        1D array of the impact parameter of each chord, sorted in ascending order, i.e. in the
        same order as load_axuv_signal(). [m]
    """

    # Process the shot if it has not been processed yet
    generate_axuv_data(shotnum, diodeArrayNum)

    filenameList = axuv_data_files(shotnum, diodeArrayNum)

    with np.load(filenameList[0], allow_pickle=True) as data:
        impactParams = data[f's{shotnum}_impact_b_raw_m']

    return np.sort(impactParams)

def chord_directions(impactParams, pinholePos):
    """
    Find the direction of the line of sight of each chord from the pinhole and its impact
    parameter.

    The chord at angle phi from the line joining the pinhole to the machine axis has the impact
    parameter
        b = -|P| sin(phi)
    as in impact_p() in the shinethru AXUV code (axuv_postprocessing/object_init.py). Written in
    terms of the direction d of the chord into the plasma, this is the cross product
        b = P_x d_y - P_y d_x
    so a chord with b > 0 passes the axis on the right as seen from the pinhole. |b| is the
    distance of closest approach of the chord to the axis.

    Parameters
    ----------
    impactParams : np.array
        1D array of the impact parameter of each chord. [m]
    pinholePos : np.array
        (x, y) of the pinhole. [m]

    Returns
    -------
    directions : np.array
        2D array [Chord x 2] of the unit vector along each chord, pointing from the pinhole into
        the plasma.
    """

    # Angle of the line from the pinhole to the axis
    centralAngle = np.arctan2(-pinholePos[1], -pinholePos[0])

    # Angle of each chord from that line
    phi = -np.arcsin(impactParams / np.linalg.norm(pinholePos))

    chordAngles = centralAngle + phi

    return np.stack((np.cos(chordAngles), np.sin(chordAngles)), axis=-1)

def chord_circle_intersections(pinholePos, directions, radius):
    """
    Find where each chord enters and leaves a circle centred on the machine axis.

    A point on chord i is pinholePos + s * directions[i], where s is the distance from the
    pinhole.

    Parameters
    ----------
    pinholePos : np.array
        (x, y) of the pinhole. [m]
    directions : np.array
        2D array [Chord x 2] of the unit vector along each chord, from chord_directions().
    radius : float
        Radius of the circle. [m]

    Returns
    -------
    sIn : np.array
        1D array of the distance from the pinhole at which each chord enters the circle. [m]
        NaN for a chord that misses the circle.
    sOut : np.array
        1D array of the distance from the pinhole at which each chord leaves the circle. [m]
        NaN for a chord that misses the circle.
    """

    # |P + s d|^2 = radius^2, with |d| = 1
    pDotD = directions @ pinholePos
    discriminant = pDotD**2 - (pinholePos @ pinholePos - radius**2)

    # Chords that miss the circle give NaN
    with np.errstate(invalid='ignore'):
        sqrtDisc = np.sqrt(discriminant)

    return -pDotD - sqrtDisc, -pDotD + sqrtDisc
