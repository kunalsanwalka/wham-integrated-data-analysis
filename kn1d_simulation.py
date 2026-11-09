"""
This script loads the neutral particle (ion) source rates and neutral densities calculated by KN1D
for the CQL3D simulation.

KN1D is run once per IPS timestep, after CQL3D, in two places-
    - At the midplane (z = 0), with the main vessel neutral density.
    - At the gas box (z = 0.8 m), with the gas box neutral density.
Each run is saved to its own file, and the two are then added together on the midplane grid
into the file CQL3D reads its particle source from. The files are-

    'total'    : kn1dc_lite.nc, the main vessel source rate before disambiguation, i.e. the
                 midplane and gas box runs added together. This is what CQL3D uses. Its
                 neutral distribution is that of the midplane run alone.
    'midplane' : kn1dc_midplane_lite.nc, the midplane run alone.
    'gasbox'   : kn1dc_gasbox_lite.nc, the gas box run alone, on the gas box radial grid.

These are the 'lite' files, in which KN1DC's convert_to_lite_netcdf() has rescaled everything
the neutrals drive (fH, CX and ionization) so that the neutral density at the wall is the
nH_wall_force of the run's KN1DC_*_inputs.jl, i.e. the main vessel and gas box neutral densities
in the simulation name. The full outputs (kn1dc.nc, kn1dc_gasbox.nc) are not rescaled, so their
nH, Sion etc. are not what CQL3D used.

The source rate is in [m^-3 s^-1] and the neutral density in [m^-3]. The *_per_flux_tube()
functions put them onto the CQL3D flux tubes, with the source rate in [s^-1] per flux tube.
"""

import pickle
import numpy as np
import scipy as sc
import xarray as xr

import config
from cql3d_simulation import ips_timestep_dirs, load_flux_tube_data, flux_tube_edges, load_eqdsk_flux

# KN1D output file of each component
kn1dFiles = {'total': 'kn1dc_lite.nc',
             'midplane': 'kn1dc_midplane_lite.nc',
             'gasbox': 'kn1dc_gasbox_lite.nc'}

def generate_kn1d_filename_list(simulationName, component='total'):
    """
    Location of the KN1D output of every IPS timestep of a simulation, in time order.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    component : str
        Which KN1D output to load, one of the keys of kn1dFiles.
        Default is 'total', the main vessel before disambiguation.

    Returns
    -------
    filenameList : list of str
        Location of the KN1D output file of each IPS timestep.
    """

    # All simulations are stored in the same directory
    simulationDir = config.simulationScanDir + simulationName + '/'

    return [stepDir + 'components/neut__kn1dc_5/' + kn1dFiles[component]
            for stepDir in ips_timestep_dirs(simulationDir)]

def kn1d_profiles(filename):
    """
    Calculate the radial source rate and neutral density profiles from a KN1D lite output file.

    Both are velocity integrals over the 2D (vperp, vpar) grid of the file.

    Parameters
    ----------
    filename : str
        Location of the KN1D lite output file.

    Returns
    -------
    Sion : np.array
        1D array of the net ion source rate at each radius. [m^-3 s^-1]
        This is CX_H_to_i - CX_i_to_H + Ionization - Recombination, so it can be negative.
    nH : np.array
        1D array of the neutral density at each radius. [m^-3]
    rhoH : np.array
        1D array of the radius of each point. [m]
    """

    # Open the file
    ds = xr.open_dataset(filename,
                         decode_timedelta=False)

    # Particle source rate for each velocity point
    # [radius x vpar x vperp] [1/(m^3 * sec * vol_2v)]
    net_ion_source_vperpvpar = ds['net_ion_source_vperpvpar'].values

    # Neutral distribution function
    # [radius x vpar x vperp] [1/(m^3 * vol_2v)]
    fH_vperpvpar = ds['fH_vperpvpar'].values

    # Radial positon [m]
    rhoH = ds['rhoH_lowres'].values

    # Velocity volume elements, normalized by Vnorm^3
    vol_2v = ds['vol_2v'].values

    ds.close()

    # Integrate over velocity at each radial position
    Sion = np.sum(net_ion_source_vperpvpar * vol_2v[np.newaxis, :, :], axis=(1, 2))
    nH = np.sum(fH_vperpvpar * vol_2v[np.newaxis, :, :], axis=(1, 2))

    return Sion, nH, rhoH

def load_kn1d_data(simulationName, component='total', redoAnalysis=False):
    """
    Load the time dependent radial source rate and neutral density profiles of the simulation.

    The profiles are cached in-
    config.simulationScanDir + f'{simulationName}/kn1d_data_{component}.pkl'

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    component : str
        Which KN1D output to load, one of the keys of kn1dFiles.
        Default is 'total', the main vessel before disambiguation.
    redoAnalysis : bool
        Force the profiles to be reloaded from the KN1D files instead of the cache.
        Default is False.

    Returns
    -------
    kn1dData : dict
        'Sion2D'  : [Time x r], net ion source rate. [m^-3 s^-1]
        'nH2D'    : [Time x r], neutral density. [m^-3]
        'rhoH'    : [r], radius of each point. [m]
                    For 'total' and 'midplane' this is the midplane radius, for 'gasbox' it is
                    the radius at the gas box.
        'timeArr' : [Time], time at the end of the IPS timestep each profile was calculated at. [s]
    """

    # All simulations are stored in the same directory
    simulationDir = config.simulationScanDir + simulationName + '/'

    savePath = simulationDir + f'kn1d_data_{component}.pkl'

    # Try to load the data if it has already been stored
    try:

        if redoAnalysis:
            raise FileNotFoundError('Forcing redo of the KN1D profiles')

        print(f'Trying to load the saved {component} KN1D profiles for {simulationName}')

        with open(savePath, 'rb') as loadFile:
            kn1dData = pickle.load(loadFile)

    except FileNotFoundError as e:

        print(e)
        print(f'No saved {component} KN1D profiles, loading them from the KN1D files')

        Sion2D = []
        nH2D = []
        rhoH = None
        for filename in generate_kn1d_filename_list(simulationName, component):

            Sion, nH, rhoH = kn1d_profiles(filename)
            Sion2D.append(Sion)
            nH2D.append(nH)

        timeArr, _, _ = load_flux_tube_data(simulationName, redoAnalysis=redoAnalysis)

        kn1dData = {'Sion2D': np.array(Sion2D),
                    'nH2D': np.array(nH2D),
                    'rhoH': rhoH,
                    'timeArr': timeArr}

        with open(savePath, 'wb') as saveFile:
            pickle.dump(kn1dData, saveFile)

        print(f'Saved the KN1D profiles to- \n {savePath}')

    return kn1dData

def load_source_rate_data(simulationName, component='total', redoAnalysis=False):
    """
    Load the time dependent radial source rate profile of the simulation.

    See load_kn1d_data() for the parameters.

    Returns
    -------
    Sion2D : np.array
        2D array [Time x r] of the net ion source rate. [m^-3 s^-1]
    rhoH : np.array
        1D array of the radius of each point. [m]
    timeArr : np.array
        1D array of the time at the end of the IPS timestep each profile was calculated at. [s]
    """

    kn1dData = load_kn1d_data(simulationName, component, redoAnalysis=redoAnalysis)

    return kn1dData['Sion2D'], kn1dData['rhoH'], kn1dData['timeArr']

def load_neutral_density_data(simulationName, component='midplane', redoAnalysis=False):
    """
    Load the time dependent radial neutral density profile of the simulation.

    The neutral density is the velocity integral of the neutral distribution function fH. It is
    the neutral density CQL3D's source rate was calculated with, i.e. scaled so that it is the
    nH_wall_force of the run at the wall.

    See load_kn1d_data() for the parameters. The default component is 'midplane', the main
    vessel neutrals at z = 0. The neutral distribution of 'total' is that of the midplane run as
    well, while 'gasbox' gives the neutrals at the gas box (z = 0.8 m).

    Returns
    -------
    nH2D : np.array
        2D array [Time x r] of the neutral density. [m^-3]
    rhoH : np.array
        1D array of the radius of each point. [m]
    timeArr : np.array
        1D array of the time at the end of the IPS timestep each profile was calculated at. [s]
    """

    kn1dData = load_kn1d_data(simulationName, component, redoAnalysis=redoAnalysis)

    return kn1dData['nH2D'], kn1dData['rhoH'], kn1dData['timeArr']

def flux_tube_average(rhoH, profile, rMid, numPoints=50):
    """
    Average a midplane radial profile over the cross-section of each flux tube.

    The flux tubes span the midplane radii between flux_tube_edges(rMid). The profile is linearly
    interpolated between its points (and held flat outside them) and averaged over the area of
    each tube, i.e. weighted by r.

    Parameters
    ----------
    rhoH : np.array
        1D array of the midplane radius of each point of the profile. [m]
    profile : np.array
        1D array of the profile.
    rMid : np.array
        1D array of the midplane radius of the centre of each flux tube. [m]
    numPoints : int
        Number of points each flux tube is sampled at.
        Default is 50, which is well below the KN1D grid spacing for the CQL3D flux tubes.

    Returns
    -------
    tubeAverage : np.array
        1D array of the area averaged profile in each flux tube.
    """

    edges = flux_tube_edges(rMid)

    tubeAverage = np.zeros(len(rMid))
    for j in range(len(rMid)):

        rFine = np.linspace(edges[j], edges[j+1], numPoints)
        profileFine = np.interp(rFine, rhoH, profile)

        tubeAverage[j] = np.trapezoid(profileFine * rFine, rFine) / np.trapezoid(rFine, rFine)

    return tubeAverage

def source_rate_per_flux_tube(simulationName, component='total', redoAnalysis=False):
    """
    Number of ions sourced per second in each CQL3D flux tube.

    The source rate density is averaged over the midplane cross-section of each flux tube (see
    flux_tube_average()) and multiplied by the volume of the flux tube (dvol), using the dvol of
    the same IPS timestep the source rate was calculated at.

    This treats the midplane source rate density as uniform along the whole length of the flux
    tube, as dvol is the volume over the whole field line.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    component : str
        'total' or 'midplane', see kn1dFiles. The gas box source rate is on the gas box radial
        grid, so it cannot be put on the flux tubes by its midplane radius.
        Default is 'total', the main vessel source rate before disambiguation.
    redoAnalysis : bool
        Force the profiles and flux tube data to be reloaded instead of using the caches.
        Default is False.

    Returns
    -------
    Stube : np.array
        2D array [Time x Flux Tube] of the net ion source rate in each flux tube. [s^-1]
    rMid : np.array
        2D array [Time x Flux Tube] of the midplane radius of the centre of each flux tube. [m]
    timeArr : np.array
        1D array of the time at the end of each IPS timestep. [s]
    """

    if component == 'gasbox':
        raise ValueError('The gas box source rate is on the gas box radial grid, so it cannot be '
                         'put on the flux tubes by its midplane radius.')

    Sion2D, rhoH, timeArr = load_source_rate_data(simulationName, component, redoAnalysis=redoAnalysis)
    _, rMid, dvol = load_flux_tube_data(simulationName, redoAnalysis=redoAnalysis)

    Stube = np.array([flux_tube_average(rhoH, Sion2D[i], rMid[i]) * dvol[i]
                      for i in range(len(timeArr))])

    return Stube, rMid, timeArr

def neutral_density_per_flux_tube(simulationName, component='midplane', redoAnalysis=False):
    """
    Midplane neutral density averaged over the cross-section of each CQL3D flux tube.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    component : str
        'midplane' or 'total', see kn1dFiles. Both hold the neutrals of the midplane run. The gas
        box neutrals are on the gas box radial grid, so they cannot be put on the flux tubes by
        their midplane radius.
        Default is 'midplane'.
    redoAnalysis : bool
        Force the profiles and flux tube data to be reloaded instead of using the caches.
        Default is False.

    Returns
    -------
    nTube : np.array
        2D array [Time x Flux Tube] of the midplane neutral density in each flux tube. [m^-3]
    rMid : np.array
        2D array [Time x Flux Tube] of the midplane radius of the centre of each flux tube. [m]
    timeArr : np.array
        1D array of the time at the end of each IPS timestep. [s]
    """

    if component == 'gasbox':
        raise ValueError('The gas box neutral density is on the gas box radial grid, so it cannot '
                         'be put on the flux tubes by its midplane radius.')

    nH2D, rhoH, timeArr = load_neutral_density_data(simulationName, component, redoAnalysis=redoAnalysis)
    _, rMid, _ = load_flux_tube_data(simulationName, redoAnalysis=redoAnalysis)

    nTube = np.array([flux_tube_average(rhoH, nH2D[i], rMid[i]) for i in range(len(timeArr))])

    return nTube, rMid, timeArr

def disambiguate_source_rate(simulationName, timeToPlot, zGasBox=0.8):
    """
    Separate the gas box and main vessel contributions to the main vessel source rate.

    The gas box source rate is calculated at z = zGasBox and the main vessel source rate
    at the midplane (z = 0). The main vessel source rate includes the contribution from
    the gas box, so the gas box source rate is mapped to the midplane along lines of
    constant flux and subtracted from the main vessel source rate.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    timeToPlot : float
        The time at which to disambiguate the source rates. [s]
        The closest available simulation timestep is used.
    zGasBox : float
        Axial location of the gas box source rate. [m]
        Default is 0.8.

    Returns
    -------
    rhoH : np.array
        Midplane radius of the main vessel source rate. [m]
    SionMV : np.array
        Main vessel source rate with the gas box contribution removed. [m^-3 s^-1]
    SionGBMid : np.array
        Gas box source rate mapped to the midplane on the rhoH grid. [m^-3 s^-1]
    timeVal : float
        The simulation time actually used. [s]
    """

    # rhoH, rhoHGB are 1D [m]
    # Sion2D, Sion2DGB have shape [Time x radius]
    Sion2D, rhoH, timeArr = load_source_rate_data(simulationName, 'total')
    Sion2DGB, rhoHGB, _ = load_source_rate_data(simulationName, 'gasbox')

    # Find the index of the closest time in the simulation data
    timeIdx = np.argmin(np.abs(timeArr - timeToPlot))
    timeVal = timeArr[timeIdx]

    # Clip the unphysical negative source rates before remapping and subtracting
    SionMain = np.clip(Sion2D[timeIdx], a_min=0, a_max=None)
    SionGB = np.clip(Sion2DGB[timeIdx], a_min=0, a_max=None)

    # Poloidal flux from the eqdsk of the last IPS timestep [z x r]
    Rmesh, Zmesh, psi = load_eqdsk_flux(simulationName)
    fluxInterpFunc = sc.interpolate.RectBivariateSpline(Zmesh[:, 0], Rmesh[0], psi)

    # Flux of each gas box radial point
    psiGB = fluxInterpFunc(zGasBox, rhoHGB).ravel()

    # Invert psi(r) at the midplane to find the radius of each gas box flux surface.
    # psi increases monotonically with r, so it can be used directly as the interpolation abscissa.
    rMidFine = np.linspace(0, Rmesh.max(), 2000)
    psiMidFine = fluxInterpFunc(0, rMidFine).ravel()
    rhoHGBMid = np.interp(psiGB, psiMidFine, rMidFine, right=np.nan)

    # Put the remapped gas box source rate on the main vessel grid.
    # Inside the innermost remapped point the profile is flat, outside the gas box grid there is no source.
    valid = ~np.isnan(rhoHGBMid)
    SionGBMid = np.interp(rhoH, rhoHGBMid[valid], SionGB[valid], left=SionGB[valid][0], right=0)

    # Remove the gas box contribution from the main vessel
    SionMV = SionMain - SionGBMid

    return rhoH, SionMV, SionGBMid, timeVal
