"""
This script loads the CQL3D + KN1D simulation outputs- the 2D ion density, the equilibrium
(eqdsk) and the parameters encoded in the simulation name.

Every synthetic diagnostic is built on top of the functions in here.
"""

import os
import re
import pickle
import numpy as np
import xarray as xr
import matplotlib.tri as mtri
from scipy.interpolate import NearestNDInterpolator

import config

# Upper limit on the CQL3D ion density [m^-3]
# A few time slices have unphysical spikes (up to ~1e299 m^-3) at a handful of mesh points,
# far above the ~2e20 m^-3 peak of the densest simulations.
maxIonDensity = 1e21

def find_valid_time_slices(times):
    """
    Find the time slices to keep, dropping the CQL3D restart slices.

    Each CQL3D output file starts with a restart slice at the same time as the last slice of
    the previous output file. The restart slice sometimes contains garbage (up to ~1e299 m^-3),
    so only the first slice at each time is kept.

    Parameters
    ----------
    times : np.array
        Time array, which may have repeated values. [s]

    Returns
    -------
    keep : np.array
        Boolean mask of the time slices to keep.
    """

    return np.concatenate(([True], np.diff(times) > 0))

def load_density_data(simulationName):
    """
    Load the cached 2D ion density of the simulation from density_interp_data.pkl.

    The cache is made by generate_times_and_functions(). The CQL3D restart slices are dropped
    and the unphysical density spikes clipped, so the result is on the same timebase as the
    synthetic diagnostics.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.

    Returns
    -------
    dens : np.array
        3D array [Time x r x z] of the ion density. [m^-3]
    solrz : np.array
        2D array [r x z] of the r values of the simulation grid. [m]
    solzz : np.array
        2D array [r x z] of the z values of the simulation grid. [m]
    times : np.array
        1D array of the simulation times. [s]
    """

    # All simulations are stored in the same directory
    simulationDir = config.simulationScanDir + simulationName + '/'

    # Load the saved data
    with open(simulationDir + 'density_interp_data.pkl', 'rb') as loadFile:
        saveData = pickle.load(loadFile)

    solrz = saveData['solrz']
    solzz = saveData['solzz']
    times = saveData['times']

    # [Time x r x z]
    dens = saveData['dens']

    # Drop the CQL3D restart slices and clip the unphysical density spikes
    keep = find_valid_time_slices(times)
    times = times[keep]
    dens = np.clip(dens[keep], a_min=0, a_max=maxIonDensity)

    return dens, solrz, solzz, times

def load_throat_density(simulationName):
    """
    Load the ion density profile at the mirror throat vs. time.

    The CQL3D grid follows each flux surface out past the mirror throat into the expander. The
    throat is taken as the point where the flux surface is closest to the axis, which by flux
    conservation is where |B| peaks.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.

    Returns
    -------
    densThroat : np.array
        2D array [Time x r] of the ion density at the mirror throat. [m^-3]
    rThroat : np.array
        1D array of the radius of each flux surface at the mirror throat. [m]
    zThroat : np.array
        1D array of the axial position of the mirror throat on each flux surface. [m]
    times : np.array
        1D array of the simulation times. [s]
    """

    dens, solrz, solzz, times = load_density_data(simulationName)

    # Index along z of the throat on each flux surface
    surfIdx = np.arange(solrz.shape[0])
    throatIdx = np.argmin(solrz, axis=1)

    return dens[:, surfIdx, throatIdx], solrz[surfIdx, throatIdx], solzz[surfIdx, throatIdx], times

def ips_timestep_dirs(simulationDir):
    """
    Find the directories of every IPS timestep of a simulation, in time order.

    For each simulation, the output of each IPS component is stored under-
    [SIMULATION DIR.]/simulation_results/[TIMES]/components/[COMPONENT]/
    Here, the times go from 1.000 to X.000 where X is the final timestep. This changes by simulation.

    Parameters
    ----------
    simulationDir : str
        Directory of the simulation, ending in '/'.

    Returns
    -------
    stepDirList : list of str
        Directory of each IPS timestep, ending in '/', sorted by time.
    """

    resultsDir = simulationDir + 'simulation_results/'

    # The IPS timestep directories are named after the time
    simTimes = []
    for dirName in os.listdir(resultsDir):
        try:
            float(dirName)
        except ValueError:
            # Skips 'plasma_state' and anything else that is not a timestep
            continue
        simTimes.append(dirName)

    simTimes = sorted(simTimes, key=float)

    return [resultsDir + simTime + '/' for simTime in simTimes]

def generate_filename_list(simulationDir):
    """
    Location of the CQL3D output (WHAM.nc) of every IPS timestep of a simulation, in time order.

    Parameters
    ----------
    simulationDir : str
        Directory of the simulation, ending in '/'.

    Returns
    -------
    filenameList : list of str
        Location of each WHAM.nc file.
    """

    return [stepDir + 'components/fp__cql3dm_4/WHAM.nc' for stepDir in ips_timestep_dirs(simulationDir)]

def ion_dens(filename):
    """
    This function loads the (r,z) ion density profile from a given cql3d+kn1d .nc file.

    Parameters
    ----------
    filename : str
        Location of the CQL3D output file.

    Returns
    -------
    dens : np.array
        3D array of the ion density profile [Time x R x Z] [m^-3]
    solrz : np.array
        2D array of the r values [m]
    solzz : np.array
        2D array of the z values [m]
    time : np.array
        1D array of the time values [s]
    """

    # Open the file
    ds = xr.open_dataset(filename,
                         decode_timedelta=False)
    
    # Density [Time x R x Z x Species]
    dens = ds['densz1'].values
    # [cm^-3] to [m^-3]
    dens *= 1e6

    # 1st one is ions, the 2nd is electrons.
    dens = dens[:, :, :, 0]

    # Clip the unphysical density spikes
    dens = np.clip(dens, a_min=None, a_max=maxIonDensity)

    # Major radius of z points (=r)
    # [cm] to [m]
    solrz = ds['solrz'].values * 1e-2
    
    # Height of z points (=z)
    # [cm] to [m]
    solzz = ds['solzz'].values * 1e-2

    # Time array
    time = ds['time'].values

    return dens, solrz, solzz, time

def generate_single_interpolation(plasmaDens, solrz, solzz):
    """
    This function creates an interpolation function out of a 2D density profile.

    Parameters
    ----------
    plasmaDens : np.array
        2D density profile
    solrz : np.array
        r-values
    solzz : np.array
        z-values

    Returns
    -------
    interpFunc : function
        An interpolation function that takes in (r, z) coordinates (in m) and returns the density at that point. [m^-3]    
    """

    # Extend the data to +-z
    solrz_full = np.concatenate((solrz[:, :0:-1], solrz), axis=1)
    solzz_full = np.concatenate((-solzz[:, :0:-1], solzz), axis=1)
    plasmaDens_full = np.concatenate((plasmaDens[:, :0:-1], plasmaDens), axis=1)

    nr, nz = solrz_full.shape

    def grid_triangles(nr, nz):
        ir, iz = np.meshgrid(np.arange(nr - 1), np.arange(nz - 1), indexing='ij')
        i0 = (ir * nz + iz).ravel()
        i1 = i0 + 1
        i2 = i0 + nz
        i3 = i2 + 1
        return np.concatenate((np.column_stack((i0, i1, i2)),
                                np.column_stack((i1, i3, i2))))

    rVals = solrz_full.ravel()
    zVals = solzz_full.ravel()
    densVals = plasmaDens_full.ravel()

    triangles = grid_triangles(nr, nz)

    triangulation = mtri.Triangulation(zVals, rVals, triangles)
    triInterp = mtri.LinearTriInterpolator(triangulation, densVals)

    # The SOL flux-surface mesh doesn't extend to the magnetic axis, so
    # small-r queries near the axis fall outside every triangle even though
    # they are physically "inside" the plasma. Flux surfaces are nested, so
    # whichever radial row has the smaller mean r is the innermost boundary
    # of the mesh at every z; use it to tell that near-axis hole apart from
    # points that are genuinely outside the mesh (e.g. beyond the outer
    # wall), where returning 0 is correct.
    innerRow = 0 if np.mean(solrz_full[0, :]) < np.mean(solrz_full[-1, :]) else -1
    sortIdx = np.argsort(solzz_full[innerRow, :])
    zBoundary = solzz_full[innerRow, :][sortIdx]
    rBoundary = solrz_full[innerRow, :][sortIdx]

    nearestInterp = NearestNDInterpolator(np.column_stack((rVals, zVals)), densVals)

    def interpFunc(points):
        points = np.atleast_2d(points)
        rQ = points[:, 0]
        zQ = points[:, 1]

        densQ = triInterp(zQ, rQ)
        invalid = np.ma.getmaskarray(densQ)
        densQ = np.ma.filled(densQ, 0.0)

        if np.any(invalid):

            # Of the invalid points, find the ones inside the near-axis hole
            # (r below the innermost meshed flux surface at that z) and fill
            # them in with the nearest available data instead of 0.
            rInnerBoundary = np.interp(zQ[invalid], zBoundary, rBoundary)
            insideHole = invalid.copy()
            insideHole[invalid] = rQ[invalid] < rInnerBoundary

            if np.any(insideHole):
                densQ[insideHole] = nearestInterp(rQ[insideHole], zQ[insideHole])

        return densQ

    return interpFunc

def generate_interpolation_functions(filename):

    # Load the data
    dens, solrz, solzz, time = ion_dens(filename)

    interpFuncList = []

    # Go over each timestep and generate an interpolation function
    for i in range(len(time)):

        interpFuncList.append(generate_single_interpolation(dens[i], solrz, solzz))

    return interpFuncList, time, dens, solrz, solzz

def generate_times_and_functions(simulationName):
    """
    Generate interpolation functions for the plasma density for every timestep of the simulation.

    The density is loaded from the WHAM.nc file of every IPS timestep the first time this is run,
    and cached in- config.simulationScanDir + f'{simulationName}/density_interp_data.pkl'

    Parameters
    ----------
    simulationName : str
        Name of the simulation.

    Returns
    -------
    interpFuncs : list of functions
        Interpolation functions, one per timestep, parallel to `times`.
    times : np.array
        Time array [s].
    """

    # All simulations are stored in the same directory
    simulationDir = config.simulationScanDir + simulationName + '/'

    # Try to load the data if it has already been stored
    try:

        print('Trying to load the saved 2D density profile data')

        with open(simulationDir + 'density_interp_data.pkl', 'rb') as loadFile:
            saveData = pickle.load(loadFile)

            # Clip the unphysical density spikes (older saved data is not clipped)
            savedDens = np.clip(saveData['dens'], a_min=None, a_max=maxIonDensity)

            interpFuncs = [generate_single_interpolation(dens, saveData['solrz'], saveData['solzz']) for dens in savedDens]

            times = saveData['times']

    except FileNotFoundError:

        print('No saved 2D density profile data, loading it from the .nc files')

        # Generate a list of the directories that store the data
        filenameList = generate_filename_list(simulationDir)

        # Go over each filename and generate the interpolation functions and get the time array
        interpFuncs = []
        densList = []
        solrz = None
        solzz = None
        times = np.array([])

        endTime = 0

        for i in range(len(filenameList)):

            print(f'Loading {i+1} of {len(filenameList)}')

            try:

                currFuncs, currTimes, currDens, solrz, solzz = generate_interpolation_functions(filenameList[i])
                interpFuncs.extend(currFuncs)

                densList.append(currDens)

                times = np.concatenate([times, currTimes+endTime])
                endTime = times[-1]

            except Exception:

                print(f'Error when loading data from the following directory- \n {filenameList[i]}')
                continue

        # Save the times and the data needed to rebuild the interpolation functions.
        saveData = {
            'dens': np.concatenate(densList, axis=0),
            'solrz': solrz,
            'solzz': solzz,
            'times': times,
        }

        savePath = simulationDir + 'density_interp_data.pkl'
        with open(savePath, 'wb') as saveFile:
            pickle.dump(saveData, saveFile)

        print(f'Saved interpolation data to- \n {savePath}')

    # Drop the CQL3D restart slices
    keep = find_valid_time_slices(times)
    interpFuncs = [interpFuncs[i] for i in np.where(keep)[0]]
    times = times[keep]

    return interpFuncs, times

def load_temperature_data(simulationName, redoAnalysis=False):
    """
    Load the midplane radial temperature profiles of the electrons and ions vs. time.

    The temp variable in the CQL3D output is only updated for the Maxwellian species, so for the
    general species (here both the ions and the electrons) it stays at its input value. The
    temperature is instead taken as the effective temperature of the distribution function at
    the midplane, T = (2/3) * <E>, from the energym variable.

    The WHAM.nc files are stitched together in the same way as generate_times_and_functions()
    does for the density, so the result is on the same timebase as load_density_data().

    The profiles are cached in- config.simulationScanDir + f'{simulationName}/temperature_data.pkl'

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    redoAnalysis : bool
        Force the profiles to be reloaded from the WHAM.nc files instead of the cache.
        Default is False.

    Returns
    -------
    Te : np.array
        2D array [Time x r] of the midplane electron temperature. [keV]
    Ti : np.array
        2D array [Time x r] of the midplane ion temperature. [keV]
    rMid : np.array
        1D array of the midplane radius of each flux surface. [m]
    times : np.array
        1D array of the simulation times. [s]
    """

    # All simulations are stored in the same directory
    simulationDir = config.simulationScanDir + simulationName + '/'

    savePath = simulationDir + 'temperature_data.pkl'

    # Try to load the data if it has already been stored
    try:

        if redoAnalysis:
            raise FileNotFoundError('Forcing redo of the temperature profiles')

        print(f'Trying to load the saved temperature profiles for {simulationName}')

        with open(savePath, 'rb') as loadFile:
            saveData = pickle.load(loadFile)

    except FileNotFoundError as e:

        print(e)
        print('No saved temperature profiles, loading them from the .nc files')

        filenameList = generate_filename_list(simulationDir)

        TeList = []
        TiList = []
        rMid = None
        times = np.array([])

        endTime = 0

        for i in range(len(filenameList)):

            print(f'Loading {i+1} of {len(filenameList)}')

            try:

                ds = xr.open_dataset(filenameList[i],
                                     decode_timedelta=False)

                # [Time x General Species x r], the 1st general species is ions, the 2nd is electrons
                energym = ds['energym'].values

                # Major radius of the midplane point of each flux surface
                # [cm] to [m]
                rMid = ds['solrz'].values[:, 0] * 1e-2

                currTimes = ds['time'].values

                ds.close()

            except Exception:

                print(f'Error when loading data from the following file- \n {filenameList[i]}')
                continue

            TiList.append(2/3 * energym[:, 0, :])
            TeList.append(2/3 * energym[:, 1, :])

            # Each file starts at t=0, so offset it by the end of the previous one
            times = np.concatenate([times, currTimes+endTime])
            endTime = times[-1]

        saveData = {'Te': np.concatenate(TeList, axis=0),
                    'Ti': np.concatenate(TiList, axis=0),
                    'rMid': rMid,
                    'times': times}

        with open(savePath, 'wb') as saveFile:
            pickle.dump(saveData, saveFile)

        print(f'Saved the temperature profiles to- \n {savePath}')

    # Drop the CQL3D restart slices
    times = saveData['times']
    keep = find_valid_time_slices(times)

    return saveData['Te'][keep], saveData['Ti'][keep], saveData['rMid'], times[keep]

def load_throat_temperature(simulationName, redoAnalysis=False):
    """
    Load the radial temperature profiles of the electrons and ions at the mirror throat vs. time.

    The temperature is the effective temperature of the distribution function, T = (2/3) * <E>,
    as in load_temperature_data(), but taken from the energyz variable, which holds <E> at every
    point along each flux surface. The throat is taken as the point where |B| peaks on each flux
    surface (the maximum of bbpsi), the same point load_throat_density() picks.

    At the throat the ions are streaming out through the loss cone, so <E> includes the energy of
    that flow as well as the thermal energy.

    The profiles are cached in-
    config.simulationScanDir + f'{simulationName}/throat_temperature_data.pkl'

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    redoAnalysis : bool
        Force the profiles to be reloaded from the WHAM.nc files instead of the cache.
        Default is False.

    Returns
    -------
    Te : np.array
        2D array [Time x r] of the electron temperature at the mirror throat. [keV]
    Ti : np.array
        2D array [Time x r] of the ion temperature at the mirror throat. [keV]
    rThroat : np.array
        1D array of the radius of each flux surface at the mirror throat. [m]
    times : np.array
        1D array of the simulation times. [s]
    """

    # All simulations are stored in the same directory
    simulationDir = config.simulationScanDir + simulationName + '/'

    savePath = simulationDir + 'throat_temperature_data.pkl'

    # Try to load the data if it has already been stored
    try:

        if redoAnalysis:
            raise FileNotFoundError('Forcing redo of the throat temperature profiles')

        print(f'Trying to load the saved throat temperature profiles for {simulationName}')

        with open(savePath, 'rb') as loadFile:
            saveData = pickle.load(loadFile)

    except FileNotFoundError as e:

        print(e)
        print('No saved throat temperature profiles, loading them from the .nc files')

        filenameList = generate_filename_list(simulationDir)

        TeList = []
        TiList = []
        rThroat = None
        times = np.array([])

        endTime = 0

        for i in range(len(filenameList)):

            print(f'Loading {i+1} of {len(filenameList)}')

            try:

                ds = xr.open_dataset(filenameList[i],
                                     decode_timedelta=False)

                # [Time x r x General Species x z], the 1st general species is ions, the 2nd is electrons
                energyz = ds['energyz'].values

                # Index along z of the throat on each flux surface
                surfIdx = np.arange(energyz.shape[1])
                throatIdx = np.argmax(ds['bbpsi'].values, axis=1)

                # [cm] to [m]
                rThroat = ds['solrz'].values[surfIdx, throatIdx] * 1e-2

                currTimes = ds['time'].values

                ds.close()

            except Exception:

                print(f'Error when loading data from the following file- \n {filenameList[i]}')
                continue

            TiList.append(2/3 * energyz[:, surfIdx, 0, throatIdx])
            TeList.append(2/3 * energyz[:, surfIdx, 1, throatIdx])

            # Each file starts at t=0, so offset it by the end of the previous one
            times = np.concatenate([times, currTimes+endTime])
            endTime = times[-1]

        saveData = {'Te': np.concatenate(TeList, axis=0),
                    'Ti': np.concatenate(TiList, axis=0),
                    'rThroat': rThroat,
                    'times': times}

        with open(savePath, 'wb') as saveFile:
            pickle.dump(saveData, saveFile)

        print(f'Saved the throat temperature profiles to- \n {savePath}')

    # Drop the CQL3D restart slices
    times = saveData['times']
    keep = find_valid_time_slices(times)

    return saveData['Te'][keep], saveData['Ti'][keep], saveData['rThroat'], times[keep]

def load_flux_tube_data(simulationName, redoAnalysis=False):
    """
    Load the flux tubes of the CQL3D radial grid at the end of every IPS timestep.

    The equilibrium is updated every IPS timestep, so the midplane radius and volume of each flux
    tube change slightly from one timestep to the next. KN1D is also run once per IPS timestep,
    after CQL3D, so these are on the same timebase as its source rates and neutral densities.

    The times are stitched together in the same way as the CQL3D outputs in
    generate_times_and_functions(), so they are on the same timebase as the density.

    The data is cached in- config.simulationScanDir + f'{simulationName}/flux_tube_data.pkl'

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    redoAnalysis : bool
        Force the data to be reloaded from the WHAM.nc files instead of the cache.
        Default is False.

    Returns
    -------
    timeArr : np.array
        1D array of the time at the end of each IPS timestep. [s]
    rMid : np.array
        2D array [Time x Flux Tube] of the midplane radius of the centre of each flux tube. [m]
    dvol : np.array
        2D array [Time x Flux Tube] of the volume of each flux tube (dvol in the CQL3D output),
        over the whole length of the field line. [m^3]
    """

    # All simulations are stored in the same directory
    simulationDir = config.simulationScanDir + simulationName + '/'

    savePath = simulationDir + 'flux_tube_data.pkl'

    # Try to load the data if it has already been stored
    try:

        if redoAnalysis:
            raise FileNotFoundError('Forcing redo of the flux tube data')

        print(f'Trying to load the saved flux tube data for {simulationName}')

        with open(savePath, 'rb') as loadFile:
            saveData = pickle.load(loadFile)

    except FileNotFoundError as e:

        print(e)
        print('No saved flux tube data, loading it from the .nc files')

        timeArr = np.array([])
        rMid = []
        dvol = []
        currTime = 0

        for filename in generate_filename_list(simulationDir):

            # Open the file
            ds = xr.open_dataset(filename,
                                 decode_timedelta=False)

            # Get the times
            time = ds['time'].values

            # [cm] to [m]
            rMid.append(ds['solrz'].values[:, 0] * 1e-2)

            # [cm^3] to [m^3]
            dvol.append(ds['dvol'].values * 1e-6)

            ds.close()

            # Append the last timestep to the array
            timeArr = np.concatenate([timeArr, [time[-1]+currTime]])
            currTime = timeArr[-1]

        saveData = {'timeArr': timeArr,
                    'rMid': np.array(rMid),
                    'dvol': np.array(dvol)}

        with open(savePath, 'wb') as saveFile:
            pickle.dump(saveData, saveFile)

        print(f'Saved the flux tube data to- \n {savePath}')

    return saveData['timeArr'], saveData['rMid'], saveData['dvol']

def flux_tube_edges(rMid):
    """
    Midplane radii of the boundaries between the flux tubes.

    The boundaries are taken halfway between the centres of neighbouring flux tubes, the same as
    run_KN1DC.jl does when it assigns the KN1D radial points to the CQL3D flux tubes. The
    innermost tube starts on axis, and the outermost tube is as wide on the outside of its centre
    as it is on the inside.

    Parameters
    ----------
    rMid : np.array
        1D array of the midplane radius of the centre of each flux tube. [m]

    Returns
    -------
    edges : np.array
        1D array of the midplane radius of the boundaries, one longer than rMid. [m]
    """

    return np.concatenate([[0], 0.5*(rMid[1:] + rMid[:-1]), [rMid[-1] + 0.5*(rMid[-1] - rMid[-2])]])

def load_eqdsk_flux(simulationName):
    """
    Load the poloidal flux from the eqdsk of the last IPS timestep of the given simulation.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.

    Returns
    -------
    Rmesh : np.array
        2D array of the radial coordinates of the eqdsk grid. [m]
    Zmesh : np.array
        2D array of the axial coordinates of the eqdsk grid. [m]
    psi : np.array
        2D array of the poloidal flux on the eqdsk grid. [Wb]
    """

    # All simulations are stored in the same directory
    simulationDir = config.simulationScanDir + simulationName + '/'

    # The last IPS timestep
    lastStepDir = ips_timestep_dirs(simulationDir)[-1]

    filenameEQDSK = lastStepDir + 'components/eq__pleiades_3/eqdsk'

    return read_eqdsk_flux(filenameEQDSK)

def read_eqdsk_flux(filenameEQDSK):
    """
    Read the poloidal flux from the given eqdsk file.

    Parameters
    ----------
    filenameEQDSK : str
        Location of the eqdsk file.

    Returns
    -------
    Rmesh : np.array
        2D array of the radial coordinates of the eqdsk grid. [m]
    Zmesh : np.array
        2D array of the axial coordinates of the eqdsk grid. [m]
    psi : np.array
        2D array of the poloidal flux on the eqdsk grid. [Wb]
    """

    with open(filenameEQDSK, 'r') as eqdskFile:
        lines = eqdskFile.read().splitlines()

    # Number of grid points in the radial (nw) and axial (nh) directions
    nw, nh = [int(val) for val in lines[0].split()[-2:]]

    # Values before the flux grid- 20 header values and the 4 1D profiles (fpol, pres, ffprim, pprime)
    numValues = 20 + 4*nw + nw*nh

    # The eqdsk is written in fixed width (16 character) fields with no guaranteed
    # whitespace between them, so it cannot be parsed by splitting on whitespace
    values = []
    for line in lines[1:]:

        line = line.rstrip()
        values.extend([float(line[i:i+16]) for i in range(0, len(line), 16)])

        if len(values) >= numValues:
            break

    values = np.array(values[:numValues])

    # Size of the grid [m] and the location of its lower left corner
    rdim, zdim = values[0], values[1]
    rleft, zmid = values[3], values[4]

    # [z x r]
    psi = values[20 + 4*nw:].reshape(nh, nw)

    r1D = np.linspace(rleft, rleft + rdim, nw)
    z1D = np.linspace(zmid - zdim/2, zmid + zdim/2, nh)

    Zmesh, Rmesh = np.meshgrid(z1D, r1D, indexing='ij')

    return Rmesh, Zmesh, psi

def parse_simulation_name(simulationName):
    """
    Extract the main vessel and gas box neutral densities from a simulation name.

    The simulation names follow the format
    nneut_XXX_gb_YYY_NBI_800kW_ECH_0kW_ionDrrOn
    where XXX is the main vessel neutral density and YYY is the gas box neutral
    density.

    Parameters
    ----------
    simulationName : str
        Name of the simulation.

    Returns
    -------
    mainVesselDens : float
        Main vessel neutral density. [m^-3]
        None if the name could not be parsed.
    gasBoxDens : float
        Gas box neutral density. [m^-3]
        None if the name could not be parsed.
    """

    match = re.search(r'nneut_([0-9.]+e[+-]?[0-9]+)_gb_([0-9.]+e[+-]?[0-9]+)', simulationName)

    if match is None:
        return None, None

    mainVesselDens = float(match.group(1))
    gasBoxDens = float(match.group(2))

    return mainVesselDens, gasBoxDens

def density_label(dens, units=True):
    """
    Make a nice LaTeX label out of a neutral density value.

    Parameters
    ----------
    dens : float
        Neutral density. [m^-3]
    units : bool
        Append the units to the label.
        Default is True.

    Returns
    -------
    label : str
        Label of the density.
    """

    exponent = int(np.floor(np.log10(dens)))
    mantissa = dens / 10**exponent

    # Drop the trailing '.0' for integer mantissas
    if np.isclose(mantissa, round(mantissa)):
        mantissaStr = f'{int(round(mantissa))}'
    else:
        mantissaStr = f'{mantissa:.1f}'

    if units:
        label = rf'${mantissaStr}\times10^{{{exponent}}}\,\mathrm{{m^{{-3}}}}$'
    else:
        label = rf'${mantissaStr}\times10^{{{exponent}}}$'

    return label

