"""
This script holds the SEE (shine-through) detector array- the measured line-integrated density,
and the synthetic diagnostic that line integrates a CQL3D density profile along the same lines
of sight.

The measured data is processed from MDSplus by experimental_data.py in the shinethru repository
(config.seeProcessingScript), which is run automatically for a shot that has not been processed
yet.
"""

import os
import sys
import pickle
import subprocess
import numpy as np

import config
from cql3d_simulation import find_valid_time_slices, generate_single_interpolation, generate_times_and_functions
from axuv_extension import build_density_extension

# Detectors discarded from the analysis when no others are specified, as their index in detDictList.
# Index 15 (-13.68 cm, the most negative impact parameter) is railed/broken on every shot.
defaultDetectorsToDrop = [15]

def load_detector_dictionary(pickleFilePath=None):
    """
    Load the geometry (lines of sight, impact parameters) of the SEE detectors.

    Parameters
    ----------
    pickleFilePath : str
        Location of the detector dictionary.
        Default is None, which loads config.seeDetectorDictPath.

    Returns
    -------
    detDictList : list
        List of dictionaries, one per detector.
    """

    if pickleFilePath is None:
        pickleFilePath = config.seeDetectorDictPath

    with open(pickleFilePath, 'rb') as pickleFile:
        detDictList = pickle.load(pickleFile)

    return detDictList

def single_detector_see_density(detDict, interpFunc):
    """
    This function calculates the line-integrated density at the detector.
    
    Parameters
    ----------
    detDict : dict
        A dictionary containing the information about the SEE detector, including its line of sight points.
    interpFunc : function
        An interpolation function that takes in (r, z) coordinates (in m) and returns the density at that point. [m^-3]

    Returns
    -------
    predictedSeeDensity : float
        Line integrated density along the line-of-sight of the given detector.
    """

    # Get the line of sight points for the detector
    losPoints = detDict['los_rz'] / 1e3 # Convert from mm to m

    # Spacing between points along the line of sight
    point1 = losPoints[:, 0]
    point2 = losPoints[:, 1]
    spacing = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    # Get the density at each point along the line of sight
    # Define the points in the way that the interpolation function expects (r, z)
    points = np.array([losPoints[0, :], losPoints[1, :]]).T
    densityAlongLos = interpFunc(points)

    # Integrate the line-integrated density along the line of sight to get the predicted SEE density at the detector
    predictedSeeDensity = np.trapezoid(densityAlongLos, dx=spacing)

    return predictedSeeDensity

def synthetic_see_detector(detDictList, interpFunc):
    """
    Calculate the line-integrated density along the line-of-sight for each detector.

    Parameters
    ----------
    detDictList : list
        List of dictionaries defining the detector parameters.
    interpFunc : function
        An interpolation function that takes in (r, z) coordinates (in m) and returns the density at that point. [m^-3]

    Returns
    -------
    lineIntegratedDensArr : np.array
        Line-integrated plasma density for each detector [m^-2]
    """

    # Array to store the line-integrated densities
    lineIntegratedDensArr = np.zeros(shape=(len(detDictList)))

    # Go over each detector in the list and calculate the predicted SEE density at each detector
    for i in range(len(detDictList)):
        lineIntegratedDensArr[i] = single_detector_see_density(detDictList[i], interpFunc)

    return lineIntegratedDensArr

def full_line_of_sight(detDict, rMax, zMax, numPoints=2000):
    """
    The line of sight of a detector, followed in both directions out to a cylinder around the
    axis.

    'los_rz' in the detector dictionary only covers |x| < 200 mm, which is enough for the CQL3D
    grid but stops short of the edge of a simulation that reaches the wall (r ~ 0.21-0.25 m at
    its ends). The line of sight is a straight line in 3D, through the points of 'los_points'
    (x, y, z) [mm]. This follows that line until it leaves the cylinder r <= rMax, |z| <= zMax.

    Parameters
    ----------
    detDict : dict
        The detector, from load_detector_dictionary().
    rMax : float
        Radius of the cylinder. [m]
    zMax : float
        Half length of the cylinder. [m]
    numPoints : int
        Number of points along the line of sight.
        Default is 2000.

    Returns
    -------
    losRZ : np.array
        2D array [Point x 2] of the (r, z) of each point along the line of sight. [m]
    spacing : float
        Distance between the points. [m]
        Returns (an empty array, 0) if the line misses the cylinder.
    """

    losPoints = detDict['los_points'] / 1e3

    # Straight line p(s) = p0 + s u, with s the distance from the first point
    p0 = losPoints[:, 0]
    u = losPoints[:, -1] - p0
    u = u / np.linalg.norm(u)

    # Where x^2 + y^2 = rMax^2
    a = u[0]**2 + u[1]**2
    b = 2 * (p0[0]*u[0] + p0[1]*u[1])
    c = p0[0]**2 + p0[1]**2 - rMax**2
    disc = b**2 - 4*a*c

    if a == 0 or disc <= 0:
        return np.empty((0, 2)), 0.0

    sMin = (-b - np.sqrt(disc)) / (2*a)
    sMax = (-b + np.sqrt(disc)) / (2*a)

    # Where |z| = zMax
    if u[2] != 0:
        sZ = np.sort(((-zMax - p0[2]) / u[2], (zMax - p0[2]) / u[2]))
        sMin, sMax = max(sMin, sZ[0]), min(sMax, sZ[1])
    elif abs(p0[2]) > zMax:
        return np.empty((0, 2)), 0.0

    if sMax <= sMin:
        return np.empty((0, 2)), 0.0

    s = np.linspace(sMin, sMax, numPoints)
    points = p0[:, None] + u[:, None] * s

    losRZ = np.column_stack((np.hypot(points[0], points[1]), points[2]))

    return losRZ, s[1] - s[0]

def synthetic_see_detector_full_los(detDictList, interpFunc, rMax, zMax, numPoints=2000):
    """
    Calculate the line-integrated density of each detector along its whole line of sight inside
    the cylinder r <= rMax, |z| <= zMax (see full_line_of_sight()), instead of the |x| < 200 mm
    of 'los_rz' that synthetic_see_detector() integrates over.

    Parameters
    ----------
    detDictList : list
        List of dictionaries defining the detector parameters.
    interpFunc : function
        An interpolation function that takes in (r, z) coordinates (in m) and returns the density
        at that point. [m^-3]
    rMax : float
        Radius of the cylinder, e.g. the edge of the simulation grid. [m]
    zMax : float
        Half length of the cylinder. [m]
    numPoints : int
        Number of points along each line of sight.
        Default is 2000.

    Returns
    -------
    lineIntegratedDensArr : np.array
        Line-integrated plasma density for each detector [m^-2]
    """

    lineIntegratedDensArr = np.zeros(len(detDictList))

    for i, detDict in enumerate(detDictList):

        losRZ, spacing = full_line_of_sight(detDict, rMax, zMax, numPoints=numPoints)

        if len(losRZ) > 0:
            lineIntegratedDensArr[i] = np.trapezoid(interpFunc(losRZ), dx=spacing)

    return lineIntegratedDensArr

def time_dependent_see_detector(simulationName, detDictList):
    """
    Run a full synthetic SEE diagnostic on the simulation.

    The signals are saved to (and loaded from)-
    config.simulationScanDir + f'{simulationName}/synthetic_detector_data.npz'

    Parameters
    ----------
    simulationName : str
        Name of the simulation
    detDictList : list
        List of dictionaries with the detector parameters

    Returns
    -------
    detDictList : list
        List of dictionaries with the synthetic simulation data added to it.
        This function adds the following keys to each dictionary
        ['simulated_signal'] = Calculated line-integrated density associated with the given detector [m^-2]
        ['simulated_signal_time'] = Corresponding time array [s]
    """

    # All simulations are stored in the same directory
    simulationDir = config.simulationScanDir + simulationName + '/'

    # Load the saved data if it already exists
    try:

        print(f'Trying to load the synthetic detector data for {simulationName}')

        dataObj = np.load(simulationDir + 'synthetic_detector_data.npz')

        syntheticSignal = dataObj['syntheticSignal']
        times = dataObj['times']

    except FileNotFoundError:

        print('No saved synthetic detector data present, generating it.')

        # Load all the interpolation functions
        interpFuncs, times = generate_times_and_functions(simulationName)

        # Array to store the time dependent SEE signals
        # [Time x Detector Number]
        syntheticSignal = np.zeros(shape=(len(times), len(detDictList)))

        for i in range(len(times)):

            # print(f'Calculating the signal for {times[i]*1e3:.2f}ms')
            syntheticSignal[i] = synthetic_see_detector(detDictList, interpFuncs[i])

        #### Save the data

        np.savez(simulationDir + 'synthetic_detector_data.npz',
                 syntheticSignal = syntheticSignal,
                 times = times)

    # Drop the CQL3D restart slices (older saved data still contains them)
    keep = find_valid_time_slices(times)
    syntheticSignal = syntheticSignal[keep]
    times = times[keep]

    # Put the data into the dictionaries
    for i in range(len(detDictList)):

        detDictList[i]['simulated_signal'] = syntheticSignal[:, i]
        detDictList[i]['simulated_signal_time'] = times

    return detDictList

def axuv_window_mask(dataObj, expTime, axuvTMin, axuvTMax):
    """
    Find the experimental times of saved data that lie in the requested AXUV time window.

    The data is saved with the window it was built with, so a narrower window can be cut out
    of it rather than regenerated. Every comparison metric reduces over the detectors only, so
    the values at the times that are kept do not depend on the window.

    Parameters
    ----------
    dataObj : NpzFile
        The saved data. Holds 'axuvTMin' and 'axuvTMax' (NaN for no limit), apart from files
        saved before these were recorded.
    expTime : np.array
        1D array of the experimental times of the saved data. [s]
    axuvTMin : float
        Start of the requested window. [s]
        None for no limit.
    axuvTMax : float
        End of the requested window. [s]
        None for no limit.

    Returns
    -------
    keep : np.array
        1D boolean array of the times in expTime that are in the requested window.

    Raises
    ------
    Exception
        If the saved data does not cover the requested window, so that it is regenerated.
    """

    if 'axuvTMin' in dataObj:
        savedTMin = float(dataObj['axuvTMin'])
        savedTMax = float(dataObj['axuvTMax'])
    else:
        # Older files do not record the window, so it is taken from the times. These are the
        # centers of the bins the AXUV data was averaged down into, so are allowed to sit up
        # to one sample inside the window they were built with.
        dt = np.median(np.diff(expTime))
        savedTMin = expTime.min() - dt
        savedTMax = expTime.max() + dt

    # No limit is stored as NaN and passed in as None
    savedTMin = -np.inf if np.isnan(savedTMin) else savedTMin
    savedTMax = np.inf if np.isnan(savedTMax) else savedTMax
    tMin = -np.inf if axuvTMin is None else axuvTMin
    tMax = np.inf if axuvTMax is None else axuvTMax

    if tMin < savedTMin or tMax > savedTMax:
        raise Exception(f'The saved data covers {savedTMin} s to {savedTMax} s, which does not '
                        f'cover the requested window of {tMin} s to {tMax} s')

    return (expTime >= tMin) & (expTime <= tMax)

def time_dependent_see_detector_extended(simulationName, shotnum, detDictList, diodeArrayNum=1,
                                         rateKHz=10.0, tanhSteepness=4.0,
                                         axuvTMin=None, axuvTMax=None, redoAnalysis=False):
    """
    Run the synthetic SEE diagnostic on the simulation extended out to the AXUV plasma radius.

    This is the same diagnostic as time_dependent_see_detector(), but run on the density profile
    built by build_density_extension() in axuv_extension.py, which adds a tanh fall-off from the
    edge of the CQL3D grid out to the plasma radius measured by AXUV. That
    extension depends on the experimental time through the AXUV plasma radius, so the signal now
    depends on both the simulated and the experimental time rather than on the simulated time
    alone.

    The signals are saved to (and loaded from)-
    config.simulationScanDir + f'{simulationName}/synthetic_detector_data_{shotnum}_extension.npz'
    which is kept separate from the synthetic_detector_data.npz of the unextended simulation.

    Parameters
    ----------
    simulationName : str
        Name of the simulation.
    shotnum : int
        Shot number the AXUV plasma radius is taken from.
    detDictList : list
        List of dictionaries with the detector parameters.
    diodeArrayNum : int
        The diode array the AXUV plasma radius is taken from, i.e. 1 for DIODEARRAY1.
        Default is 1.
    rateKHz : float
        Rate the native AXUV data is averaged down to. [kHz]
        This sets the experimental timebase of the result.
        Default is 10.
    tanhSteepness : float
        Steepness of the tanh fall-off of the extension.
        Default is 4.
    axuvTMin : float
        Start of the experimental time window the AXUV data is trimmed to. [s]
        Default is None, which keeps the data from the start of the shot.
    axuvTMax : float
        End of the experimental time window the AXUV data is trimmed to. [s]
        Default is None, which keeps the data to the end of the shot.
    redoAnalysis : bool
        Force the signals to be recomputed instead of loading the saved ones.
        Default is False.

    Returns
    -------
    syntheticSignal : np.array
        3D array [simTime x expTime x detector] of the line-integrated density. [m^-2]
        The detectors are in the same order as detDictList.
    simTime : np.array
        1D array of the simulation times. [s]
    expTime : np.array
        1D array of the experimental times of the AXUV plasma radius. [s]
    """

    # All simulations are stored in the same directory
    simulationDir = config.simulationScanDir + simulationName + '/'

    # Kept separate from the unextended synthetic_detector_data.npz
    savePath = simulationDir + f'synthetic_detector_data_{shotnum}_extension.npz'

    # Load the saved data if it already exists
    try:

        if redoAnalysis:
            raise Exception('Forcing redo of the extended synthetic detector data')

        print(f'Trying to load the extended synthetic detector data for {simulationName}')

        dataObj = np.load(savePath)

        simTime = dataObj['simTime']
        expTime = dataObj['expTime']

        # Cut the requested window out of the saved one
        keep = axuv_window_mask(dataObj, expTime, axuvTMin, axuvTMax)

        syntheticSignal = dataObj['syntheticSignal'][:, keep]
        expTime = expTime[keep]

        return syntheticSignal, simTime, expTime

    except Exception as e:

        print(e)
        print('No saved extended synthetic detector data present, generating it.')

    # Everything the extension needs that does not depend on the simulated time. The full
    # [simTime x expTime x R x Z] profile (which runs to hundreds of MB) is not built, so that it
    # does not have to be held in memory just to be line-integrated.
    # redoAnalysis is passed on so that forcing a redo here also rebuilds the cached
    # extension, rather than recomputing the signals from a stale one
    extData = build_density_extension(simulationName, shotnum, diodeArrayNum,
                                      rateKHz=rateKHz, tanhSteepness=tanhSteepness,
                                      tMin=axuvTMin, tMax=axuvTMax,
                                      redoAnalysis=redoAnalysis)

    simTime = extData['times']
    expTime = extData['axuvTimes']
    densSim = extData['densSim']
    solrz = extData['solrz']
    solzz = extData['solzz']
    shapeExt = extData['shapeExt']
    rExtMapped = extData['rExtMapped']
    zGrid = extData['zGrid']

    numRSim = solrz.shape[0]
    numRExt = rExtMapped.shape[0]

    # The mesh the extended profile lives on, i.e. the simulation grid with the flux surfaces
    # of the extension stacked onto the outside of it
    solrzExt = np.concatenate([solrz, rExtMapped], axis=0)
    solzzExt = np.concatenate([solzz, np.tile(zGrid, (numRExt, 1))], axis=0)

    # The line integral and the interpolation onto the lines of sight are both linear in the
    # density at the mesh points, and the mesh is the same at every pair of times. The extended
    # profile is the simulated density (0 on the extension surfaces) plus the extension (0 on
    # the simulated surfaces), and the extension is the same shape at every simulated time
    # scaled by the density at the edge of the simulation grid. The signal is therefore
    #
    #   signal[simTime, expTime] = simSignal[simTime] + edgeDensity[simTime]*extSignal[expTime]
    #
    # which needs one line integration per simulated time and one per experimental time rather
    # than one per pair of them.

    # Signal of the simulated density alone
    simSignal = np.zeros(shape=(len(simTime), len(detDictList)))
    densField = np.zeros(solrzExt.shape)
    for i in range(len(simTime)):

        if i % 50 == 0:
            print(f'Line integrating simulated timestep {i+1} of {len(simTime)}')

        densField[:numRSim] = densSim[i]

        interpFunc = generate_single_interpolation(densField, solrzExt, solzzExt)
        simSignal[i] = synthetic_see_detector(detDictList, interpFunc)

    # Signal of the extension alone, with an edge density of 1. The density is constant along
    # each flux surface of the extension, so the shape is just repeated along z.
    extSignal = np.zeros(shape=(len(expTime), len(detDictList)))
    densField = np.zeros(solrzExt.shape)
    for i in range(len(expTime)):

        if i % 50 == 0:
            print(f'Line integrating experimental timestep {i+1} of {len(expTime)}')

        densField[numRSim:] = shapeExt[i][:, np.newaxis]

        interpFunc = generate_single_interpolation(densField, solrzExt, solzzExt)
        extSignal[i] = synthetic_see_detector(detDictList, interpFunc)

    # Midplane density at the edge of the simulation grid, which the extension is scaled by
    edgeDensity = densSim[:, -1, 0]

    # [simTime x expTime x detector]
    syntheticSignal = (simSignal[:, np.newaxis, :]
                       + edgeDensity[:, np.newaxis, np.newaxis] * extSignal[np.newaxis, :, :])

    #### Save the data

    # The window is saved so that a narrower one can be cut out of this data later. No limit
    # is saved as NaN.
    np.savez(savePath,
             syntheticSignal = syntheticSignal,
             simTime = simTime,
             expTime = expTime,
             axuvTMin = np.nan if axuvTMin is None else axuvTMin,
             axuvTMax = np.nan if axuvTMax is None else axuvTMax)

    print(f'Saved the extended synthetic detector data to- \n {savePath}')

    return syntheticSignal, simTime, expTime

def _branch_interpolate(srcRadii, srcDens, srcSigma, targetRadii, radiusTol=0.0):
    """
    Linearly interpolate one branch (upper or lower) of the detector array onto a set of target radii.

    The interpolation is done in radius (|impact parameter|) only, so the weights are the same at every
    timestep and the whole 2D [detector x time] array can be interpolated in one shot.

    Parameters
    ----------
    srcRadii : np.array
        Absolute impact parameters of the detectors on this branch. [m]
    srcDens : np.array
        Line-integrated density of the detectors on this branch. [detector x time] [m^-2]
    srcSigma : np.array
        Uncertainty on the line-integrated density of the detectors on this branch. [detector x time] [m^-2]
    targetRadii : np.array
        Absolute impact parameters at which we want the branch evaluated. [m]
    radiusTol : float
        Distance by which a target radius is allowed to sit outside the branch and still count as covered. [m]
        The branch is held flat over that distance. It is there to catch detectors whose partner on the
        opposite branch is nominally at the same radius but is a fraction of a mm away.
        Default is 0.0.

    Returns
    -------
    densInterp : np.array
        Branch line-integrated density evaluated at targetRadii. [target x time] [m^-2]
    sigmaInterp : np.array
        Branch uncertainty evaluated at targetRadii. [target x time] [m^-2]
    inRange : np.array
        Boolean array flagging the target radii that lie inside the radial coverage of this branch.
        Values outside the coverage are clamped to the edge of the branch and should not be trusted.
    """

    nTime = srcDens.shape[1]

    # A single detector cannot define a profile, so nothing on this branch is usable
    if len(srcRadii) < 2:
        return (np.zeros(shape=(len(targetRadii), nTime)),
                np.zeros(shape=(len(targetRadii), nTime)),
                np.zeros(len(targetRadii), dtype=bool))

    # Sort the branch by radius
    sortIdx = np.argsort(srcRadii)
    srcRadii = srcRadii[sortIdx]
    srcDens = srcDens[sortIdx]
    srcSigma = srcSigma[sortIdx]

    # Bracketing detectors for each target radius
    hiIdx = np.clip(np.searchsorted(srcRadii, targetRadii), 1, len(srcRadii) - 1)
    loIdx = hiIdx - 1

    # Linear interpolation weight of the outer bracketing detector
    weight = (targetRadii - srcRadii[loIdx]) / (srcRadii[hiIdx] - srcRadii[loIdx])
    weight = np.clip(weight, 0, 1)[:, np.newaxis]

    densInterp = (1 - weight) * srcDens[loIdx] + weight * srcDens[hiIdx]

    # Uncertainties of the two bracketing detectors are independent, so they add in quadrature
    sigmaInterp = np.sqrt(((1 - weight) * srcSigma[loIdx])**2 + (weight * srcSigma[hiIdx])**2)

    # Flag the target radii the branch actually covers
    inRange = (targetRadii >= srcRadii[0] - radiusTol) & (targetRadii <= srcRadii[-1] + radiusTol)

    return densInterp, sigmaInterp, inRange

def symmetrize_experimental_data(detDictList, nInnerSkip=4, radiusTol=2e-3):
    """
    Force the experimental line-integrated density to be up-down symmetric.

    The synthetic diagnostic is built on a purely radial CQL3D profile, so it cannot reproduce any
    up-down asymmetry in the data. This function removes that asymmetry from the measurement.

    The detectors are not placed symmetrically about the midplane, so a detector above the midplane
    generally has no partner at the same |impact parameter| below it. Instead of pairing detectors up,
    the upper and lower halves of the array are each treated as a profile in radius (|impact parameter|),
    and the symmetric profile is the average of the two branches interpolated onto a common radius. Each
    detector then takes the value of that symmetric profile at its own radius. The branch interpolation
    only uses radius, so the weights are time independent and the averaging is done for every timestep in
    one vectorised operation.

    Detectors with no data (including the ones discarded by load_experimental_data()) take no part in the
    symmetrization. Detectors are left untouched (and are excluded from the branch interpolation) when:
        - They are one of the nInnerSkip innermost detectors. Their up-down asymmetry comes from the
          viewing geometry, not from the plasma, so symmetrizing them would be wrong.
        - Their radius is outside the radial coverage of the opposite branch, so there is nothing to
          average against.
        - They are outside the time window covered by every detector in the symmetrization set.

    The raw data is kept under 'line_integrated_density_raw' and 'line_integrated_density_sigma_raw', and
    the part of the signal removed by the symmetrization (raw - symmetrized) is stored under
    'line_integrated_density_symmetrization_residual'.

    Parameters
    ----------
    detDictList : list
        List of detector dictionaries, as loaded from the cached experimental data.
    nInnerSkip : int
        Number of innermost detectors (smallest |impact parameter|) left unsymmetrized.
        Default is 4.
    radiusTol : float
        Distance by which a detector is allowed to sit outside the radial coverage of the opposite branch
        and still be symmetrized. [m]
        The opposite branch is held flat over that distance. The detectors are not mirror images of each
        other, so a pair that is meant to be at the same radius can be a fraction of a mm apart. Without
        this the outer detector of such a pair would be dropped.
        Default is 2e-3, which is well below the ~10 mm detector spacing.

    Returns
    -------
    detDictList : list
        The same list, with the symmetrized line-integrated density in 'line_integrated_density'.
    """

    # Detectors that actually have data
    validIdxList = [i for i in range(len(detDictList)) if detDictList[i]['line_integrated_density'] is not None]

    # Keep a copy of the raw data on every detector and default to 'not symmetrized'
    for detDict in detDictList:

        if detDict['line_integrated_density'] is None:
            detDict['line_integrated_density_raw'] = None
            detDict['line_integrated_density_sigma_raw'] = None
            detDict['line_integrated_density_symmetrization_residual'] = None
            detDict['symmetrized'] = False
            continue

        detDict['line_integrated_density'] = np.asarray(detDict['line_integrated_density'], dtype=float)
        detDict['line_integrated_density_sigma'] = np.asarray(detDict['line_integrated_density_sigma'], dtype=float)

        detDict['line_integrated_density_raw'] = detDict['line_integrated_density'].copy()
        detDict['line_integrated_density_sigma_raw'] = detDict['line_integrated_density_sigma'].copy()
        detDict['line_integrated_density_symmetrization_residual'] = np.zeros_like(detDict['line_integrated_density'])
        detDict['symmetrized'] = False

    if len(validIdxList) == 0:
        print('No experimental data to symmetrize')
        return detDictList

    # Detector impact parameters [m]
    impactParams = np.array([detDictList[i]['impact_param_vertical'] / 1e3 for i in validIdxList])

    # Detectors excluded from the symmetrization
    excludedIdxList = set()

    # The innermost detectors are asymmetric because of the geometry, not the plasma
    innerOrder = np.argsort(np.abs(impactParams))
    for k in range(min(nInnerSkip, len(innerOrder))):
        excludedIdxList.add(validIdxList[innerOrder[k]])

    # Detectors we are going to symmetrize
    symIdxList = [i for i in validIdxList if i not in excludedIdxList]

    if len(symIdxList) < 2:
        print('Not enough detectors left to symmetrize the experimental data')
        return detDictList

    symImpactParams = np.array([detDictList[i]['impact_param_vertical'] / 1e3 for i in symIdxList])

    #### Put the detectors we are symmetrizing on a common time axis

    timeArrList = [np.asarray(detDictList[i]['time_arr_slow']) for i in symIdxList]

    # Time window covered by every detector in the symmetrization set
    tStart = max([timeArr[0] for timeArr in timeArrList])
    tStop = min([timeArr[-1] for timeArr in timeArrList])

    if tStop <= tStart:
        print('Detector time axes do not overlap. Skipping the symmetrization.')
        return detDictList

    # Use the best resolved detector inside that window as the common time axis
    windowList = [timeArr[(timeArr >= tStart) & (timeArr <= tStop)] for timeArr in timeArrList]
    commonTimeArr = windowList[int(np.argmax([len(window) for window in windowList]))]

    densArr2D = np.zeros(shape=(len(symIdxList), len(commonTimeArr)))
    sigmaArr2D = np.zeros(shape=(len(symIdxList), len(commonTimeArr)))
    for k in range(len(symIdxList)):

        densArr2D[k] = np.interp(commonTimeArr, timeArrList[k], detDictList[symIdxList[k]]['line_integrated_density_raw'])
        sigmaArr2D[k] = np.interp(commonTimeArr, timeArrList[k], detDictList[symIdxList[k]]['line_integrated_density_sigma_raw'])

    #### Average the two branches of the array in |impact parameter|

    radii = np.abs(symImpactParams)

    upperMask = symImpactParams > 0
    lowerMask = symImpactParams < 0

    upperDens, upperSigma, upperInRange = _branch_interpolate(radii[upperMask], densArr2D[upperMask], sigmaArr2D[upperMask], radii, radiusTol=radiusTol)
    lowerDens, lowerSigma, lowerInRange = _branch_interpolate(radii[lowerMask], densArr2D[lowerMask], sigmaArr2D[lowerMask], radii, radiusTol=radiusTol)

    # A detector can only be symmetrized where both branches have coverage
    bothInRange = upperInRange & lowerInRange

    symDensArr2D = np.where(bothInRange[:, np.newaxis], 0.5 * (upperDens + lowerDens), densArr2D)

    # The two branches are independent measurements, so their uncertainties add in quadrature
    symSigmaArr2D = np.where(bothInRange[:, np.newaxis], 0.5 * np.sqrt(upperSigma**2 + lowerSigma**2), sigmaArr2D)

    #### Put the symmetrized data back on each detector's own time axis

    for k in range(len(symIdxList)):

        if not bothInRange[k]:
            print(f"Detector at {symImpactParams[k]*1e3:.1f} mm has no partner on the opposite branch. Leaving it unsymmetrized.")
            continue

        detDict = detDictList[symIdxList[k]]
        timeArr = timeArrList[k]

        # Outside the common time window there is nothing to average against, so keep the raw data
        inWindow = (timeArr >= commonTimeArr[0]) & (timeArr <= commonTimeArr[-1])

        newDens = detDict['line_integrated_density_raw'].copy()
        newSigma = detDict['line_integrated_density_sigma_raw'].copy()

        newDens[inWindow] = np.interp(timeArr[inWindow], commonTimeArr, symDensArr2D[k])
        newSigma[inWindow] = np.interp(timeArr[inWindow], commonTimeArr, symSigmaArr2D[k])

        detDict['line_integrated_density'] = newDens
        detDict['line_integrated_density_sigma'] = newSigma
        detDict['line_integrated_density_symmetrization_residual'] = detDict['line_integrated_density_raw'] - newDens
        detDict['symmetrized'] = True

    return detDictList

def load_experimental_data(shotnum, symmetrize=True, nInnerSkip=4, radiusTol=2e-3, detectorsToDrop=None):
    """
    Load the measured line-integrated density of the SEE detectors for a given shot number.

    The data is loaded from config.seeDataDir + f'{shotnum}.pkl'. If the shot has not been
    processed yet, config.seeProcessingScript is run to process it from MDSplus first.

    The raw data contains an up-down asymmetry that the synthetic diagnostic cannot reproduce, since it is
    built on a purely radial CQL3D profile. By default the line-integrated density is therefore made
    up-down symmetric before it is returned (see symmetrize_experimental_data). The cached file on disk is
    never modified, only the data held in memory.

    Detectors with bad data are discarded by setting their line-integrated density (and its uncertainty)
    to None, the same as a detector with no signal. This is done before the symmetrization so that a bad
    detector does not corrupt the detectors on the opposite half of the array.

    Parameters
    ----------
    shotnum : int
        Shot number for which we want to calculate the line-integrated plasma density.
    symmetrize : bool
        Force the line-integrated density to be up-down symmetric.
        Default is True.
    nInnerSkip : int
        Number of innermost detectors left unsymmetrized, since their asymmetry is geometric.
        Default is 4.
    radiusTol : float
        Distance by which a detector is allowed to sit outside the radial coverage of the opposite half of
        the array and still be symmetrized. [m]
        Default is 2e-3.
    detectorsToDrop : list of int
        Indices (in detDictList) of the detectors with bad data, which are discarded from the analysis.
        These are 0-indexed, i.e. detector 6 (no signal) is index 5.
        Default is None, which discards defaultDetectorsToDrop (the railed detector, index 15).

    Returns
    -------
    detDictList : list
        List of dictionaries that contains the detector information AND the experimentally calculated line-integrated plasma density. [m^-2]
    """

    if detectorsToDrop is None:
        detectorsToDrop = defaultDetectorsToDrop

    savename = config.seeDataDir + f'{shotnum}.pkl'

    # Process the shot if it has not been processed yet
    if not os.path.exists(savename):

        print(f'Experimental data for shot {shotnum} was not calculated. Computing now.')

        subprocess.run([sys.executable, config.seeProcessingScript, '-s', f'{shotnum}'],
                       cwd=os.path.dirname(config.seeProcessingScript), check=True)

    print(f'Loading the experimental data for shot {shotnum}')

    with open(savename, 'rb') as file:
        detDictList = pickle.load(file)

    # Discard the detectors with bad data
    for i in detectorsToDrop:

        print(f"Discarding detector index {i} ({detDictList[i]['impact_param_vertical']/10:.2f} cm)")

        detDictList[i]['line_integrated_density'] = None
        detDictList[i]['line_integrated_density_sigma'] = None

    # Remove the up-down asymmetry the synthetic diagnostic cannot capture
    if symmetrize:
        detDictList = symmetrize_experimental_data(detDictList, nInnerSkip=nInnerSkip, radiusTol=radiusTol)

    return detDictList

def exp_dict_to_numpy(detDictList):

    # Detector impact parameters [m]
    impactParams = np.zeros(len(detDictList))
    for i in range(len(impactParams)):

        beam_pos = detDictList[i]['impact_param_vertical']
        impactParams[i] = beam_pos / 1e3

    # Load the experimental data
    expTimeArr = []
    expDataArr = []
    expDataSigmaArr = []
    dataPresent = []
    for i in range(len(impactParams)):

        lineIntegratedDens = detDictList[i]['line_integrated_density']

        if lineIntegratedDens is not None:
            expDataArr.append(detDictList[i]['line_integrated_density'])
            expDataSigmaArr.append(detDictList[i]['line_integrated_density_sigma'])
            expTimeArr.append(detDictList[i]['time_arr_slow'])
            dataPresent.append(True)
        else:
            dataPresent.append(False)

    # Get the final time for each detector
    finalTime = 1e5
    shortestTimeIdx = 0
    for i in range(len(expTimeArr)):
        
        currFinalTime = expTimeArr[i].max()
        
        if finalTime > currFinalTime:
            finalTime = currFinalTime
            shortestTimeIdx = i

    # Common time array for the experimental data
    expTimeArrNew = expTimeArr[shortestTimeIdx]

    # Put all the experimental data on the same timebase
    expDataArrNew = []
    expDataSigmaArrNew = []
    for i in range(len(expDataArr)):

        expDataArrNew.append(np.interp(expTimeArrNew, expTimeArr[i], expDataArr[i]))
        expDataSigmaArrNew.append(np.interp(expTimeArrNew, expTimeArr[i], expDataSigmaArr[i]))

    # Convert to numpy as rename
    expTimeArr = np.array(expTimeArrNew)
    expDataArr = np.array(expDataArrNew)
    expDataSigmaArr = np.array(expDataSigmaArrNew)

    return expTimeArr, expDataArr, expDataSigmaArr, dataPresent

def sim_dict_to_numpy(detDictList):

    simTimeArr = detDictList[0]['simulated_signal_time']
            
    simDataArr = np.zeros(shape=(len(detDictList), len(simTimeArr)))
    for i in range(len(detDictList)):

        simDataArr[i] = detDictList[i]['simulated_signal']

    return simTimeArr, simDataArr

def drop_suffix(detectorsToDrop):
    """
    Filename suffix that records which detectors were discarded from the analysis.

    Parameters
    ----------
    detectorsToDrop : list of int
        Indices (in detDictList) of the discarded detectors.

    Returns
    -------
    suffix : str
        e.g. '_drop9-13-15', or '_dropnone' if no detectors were discarded.
    """

    if len(detectorsToDrop) == 0:
        return '_dropnone'

    return '_drop' + '-'.join(str(i) for i in sorted(set(detectorsToDrop)))

