"""
This script extends the CQL3D density profile out to the plasma radius measured by AXUV.

CQL3D only solves out to the edge of its own radial grid, which sits well inside the plasma
radius seen by AXUV. The density between the two is filled in with a tanh fall-off that goes
from the simulated edge density to 0 at the AXUV radius, and is mapped along z by following
the eqdsk flux surfaces.
"""

import pickle
import numpy as np

import config
from axuv import generate_axuv_data, load_axuv_radius
from cql3d_simulation import load_density_data, load_eqdsk_flux

def extend_profile(rSim, densSim, plasmaRadius, rEQDSK, tanhSteepness=4.0):
    """
    Extend a midplane density profile from the edge of the simulation grid out to the plasma radius.

    The density falls from the edge density of the simulation to 0 at the plasma radius as a
    tanh, rescaled so that it hits both end points exactly.

    Parameters
    ----------
    rSim : np.array
        1D array of the midplane radial positions of the simulation grid. [m]
    densSim : np.array
        1D array of the midplane density of the simulation. [m^-3]
    plasmaRadius : float
        Radius at which the density goes to 0. [m]
    rEQDSK : np.array
        1D array of the radial positions of the eqdsk grid, used as the grid of the extension. [m]
    tanhSteepness : float
        Steepness of the tanh fall-off. Larger values keep the density flat for longer before
        dropping sharply at the plasma radius.
        Default is 4.

    Returns
    -------
    rExt : np.array
        1D array of the radial positions of the extension. [m]
        This starts at the edge of the simulation grid so the extension joins onto the profile.
    densExt : np.array
        1D array of the density of the extension. [m^-3]
    """

    rEdge = rSim[-1]
    densEdge = densSim[-1]

    # Nothing to extend if AXUV sees a plasma smaller than the simulation grid
    if plasmaRadius <= rEdge:
        return np.array([]), np.array([])

    # The eqdsk points between the two radii, with the end points added so the extension
    # joins the simulation exactly and reaches 0 exactly at the plasma radius
    rExt = rEQDSK[(rEQDSK > rEdge) & (rEQDSK < plasmaRadius)]
    rExt = np.concatenate([[rEdge], rExt, [plasmaRadius]])

    # Normalized coordinate, 0 at the edge of the simulation and 1 at the plasma radius
    rNorm = (rExt - rEdge) / (plasmaRadius - rEdge)

    tanhProfile = 0.5 * (1 - np.tanh(tanhSteepness * (rNorm - 0.5)))

    # Rescale so the fall-off goes from exactly densEdge to exactly 0
    tanhStart = 0.5 * (1 + np.tanh(tanhSteepness / 2))
    tanhEnd = 0.5 * (1 - np.tanh(tanhSteepness / 2))

    densExt = densEdge * (tanhProfile - tanhEnd) / (tanhStart - tanhEnd)

    return rExt, densExt

def map_along_flux_surfaces(rMidplane, zGrid, Rmesh, Zmesh, psi, rSimOuter):
    """
    Map a set of midplane radii along z by following the eqdsk flux surface each one sits on.

    The flux surface through a midplane radius is the contour of constant poloidal flux through
    that point, so the radius of that surface at an axial position z is the radius at which the
    flux at z equals the flux at the midplane.

    Parameters
    ----------
    rMidplane : np.array
        1D array of the midplane radii of the flux surfaces being mapped. [m]
    zGrid : np.array
        1D array of the axial positions the surfaces are mapped onto. [m]
    Rmesh : np.array
        2D array [z x r] of the radial coordinates of the eqdsk grid. [m]
    Zmesh : np.array
        2D array [z x r] of the axial coordinates of the eqdsk grid. [m]
    psi : np.array
        2D array [z x r] of the poloidal flux on the eqdsk grid. [Wb]
    rSimOuter : np.array
        1D array of the radius of the outermost simulated flux surface at each point of zGrid. [m]
        This is used to continue the mapping where the eqdsk cannot (see the note below).

    Returns
    -------
    rMapped : np.array
        2D array [rMidplane x zGrid] of the radius of each flux surface at each axial position. [m]
    """

    # The eqdsk grid is regular, so the same radial (axial) positions are used at every axial
    # (radial) position
    r1D = Rmesh[0]
    z1D = Zmesh[:, 0]

    def monotonic_prefix(psiSlice):
        """
        Number of points of an axial slice of the flux over which it increases with radius.

        The flux only rises monotonically from the machine axis out to the point where the
        eqdsk grid runs past the mirror coils, beyond which it turns over. Only the rising
        part can be inverted to get a radius from a flux.
        """

        turnover = np.nonzero(np.diff(psiSlice) <= 0)[0]

        if len(turnover) == 0:
            return len(psiSlice)

        return turnover[0] + 1

    # Flux on each of the surfaces being mapped
    izMid = np.argmin(np.abs(z1D))
    nMid = monotonic_prefix(psi[izMid])
    psiTarget = np.interp(rMidplane, r1D[:nMid], psi[izMid, :nMid])

    # Radius of each surface at every axial position of the eqdsk
    rTrace = np.full((len(rMidplane), len(z1D)), np.nan)
    for i in range(len(z1D)):

        nRise = monotonic_prefix(psi[i])

        if nRise < 2:
            continue

        # Surfaces whose flux is above the largest invertible flux of this slice have left
        # the eqdsk grid and are filled in further down instead
        onGrid = psiTarget <= psi[i, nRise-1]

        rTrace[onGrid, i] = np.interp(psiTarget[onGrid], psi[i, :nRise], r1D[:nRise])

    # Put the traced surfaces onto the axial grid of the simulation
    rMapped = np.full((len(rMidplane), len(zGrid)), np.nan)
    for k in range(len(rMidplane)):

        onGrid = np.isfinite(rTrace[k])

        # A surface that never lands on the eqdsk grid is left as NaN for the fill in below
        if np.sum(onGrid) < 2:
            continue

        rMapped[k] = np.interp(zGrid, z1D[onGrid], rTrace[k, onGrid],
                               left=np.nan, right=np.nan)

    # Past the mirror throat the surfaces expand off the edge of the eqdsk grid. There they are
    # continued with the flux expansion of the outermost simulated surface, anchored to the last
    # radius that could be traced so that the mapping stays continuous.
    for k in range(len(rMidplane)):

        onGrid = np.isfinite(rMapped[k])

        if np.all(onGrid):
            continue

        if not np.any(onGrid):
            rMapped[k] = rMidplane[k] * rSimOuter / rSimOuter[0]
            continue

        idxOnGrid = np.nonzero(onGrid)[0]

        offGridLow = np.arange(0, idxOnGrid[0])
        offGridHigh = np.arange(idxOnGrid[-1] + 1, len(zGrid))

        for edgeIdx, offGrid in ((idxOnGrid[0], offGridLow), (idxOnGrid[-1], offGridHigh)):

            if len(offGrid) == 0:
                continue

            rMapped[k, offGrid] = rMapped[k, edgeIdx] * rSimOuter[offGrid] / rSimOuter[edgeIdx]

    return rMapped

def build_density_extension(simulationName, shotnum, diodeArrayNum,
                            rateKHz=10.0, tanhSteepness=4.0, tMin=None, tMax=None,
                            redoAnalysis=False):
    """
    Build everything the density extension needs that does not depend on the simulation time.

    The radial grid of the extension, the flux surfaces it is mapped along and the shape of its
    tanh fall-off are all the same at every simulated timestep, so they are worked out once here
    and scaled by the edge density of whichever timestep is being extended.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    shotnum : int
        The shot number the AXUV plasma radius is taken from.
    diodeArrayNum : int
        The diode array the radius is taken from, i.e. 1 for DIODEARRAY1.
    rateKHz : float
        Rate the native AXUV data is averaged down to. [kHz]
        Default is 10.
    tanhSteepness : float
        Steepness of the tanh fall-off of the extension.
        Default is 4.
    tMin : float
        Start of the experimental time window the AXUV data is trimmed to. [s]
        Default is None, which keeps the data from the start of the shot.
    tMax : float
        End of the experimental time window the AXUV data is trimmed to. [s]
        Default is None, which keeps the data to the end of the shot.
    redoAnalysis : bool
        Force the extension to be rebuilt instead of loading the saved one.
        Default is False.

    Returns
    -------
    extData : dict
        Dictionary holding-
            'times' : 1D array of the simulation times. [s]
            'densSim' : 3D array [simTime x r x z] of the simulated density. [m^-3]
            'solrz' : 2D array [r x z] of the r values of the simulation grid. [m]
            'solzz' : 2D array [r x z] of the z values of the simulation grid. [m]
            'axuvTimes' : 1D array of the experimental times of the AXUV plasma radius. [s]
            'axuvRadius' : 1D array of the AXUV plasma radius. [m]
            'rExtGrid' : 1D array of the midplane radii of the extension. [m]
            'rExtMapped' : 2D array [rExt x z] of the radii of the flux surfaces of the
                           extension at every axial position. [m]
            'zGrid' : 1D array of the axial positions of the extension. [m]
            'shapeExt' : 2D array [expTime x rExt] of the shape of the extension, normalized
                         to 1 at the edge of the simulation grid.
            'Rmesh', 'Zmesh', 'psi' : the eqdsk grid and poloidal flux on it. [m, m, Wb]

    Notes
    -----
    The parts of the extension that do not come out of density_interp_data.pkl are cached in-
    config.simulationScanDir + f'{simulationName}/density_interp_data_{shotnum}_extension.pkl'
    which is kept separate from the density_interp_data.pkl of the unextended simulation. The
    simulated density is not written to it, so it stays small and cannot go out of sync with
    the unextended data. The cache records the settings it was built with and is rebuilt
    automatically if any of them, or the simulation grid, has changed since.
    """

    # All simulations are stored in the same directory
    simulationDir = config.simulationScanDir + simulationName + '/'

    # Kept separate from the unextended density_interp_data.pkl
    savePath = simulationDir + f'density_interp_data_{shotnum}_extension.pkl'

    # [Time x r x z], with the restart slices dropped and the density clipped
    dens, solrz, solzz, times = load_density_data(simulationName)

    # The settings the cached extension has to have been built with for it to be reused. The
    # shot number is in the filename, so it does not need to be checked here.
    settings = {'diodeArrayNum': diodeArrayNum,
                'rateKHz': rateKHz,
                'tanhSteepness': tanhSteepness,
                'tMin': tMin,
                'tMax': tMax}

    # The extension is mapped along the flux surfaces starting from the outermost surface of
    # the simulation grid, so a cache built on a different grid cannot be reused
    gridFingerprint = {'rSimOuter': solrz[-1], 'zGrid': solzz[-1]}

    # Load the saved extension if it already exists
    try:

        if redoAnalysis:
            raise Exception('Forcing redo of the density extension')

        print(f'Trying to load the saved density extension for {simulationName}')

        with open(savePath, 'rb') as loadFile:
            extData = pickle.load(loadFile)

        if extData['settings'] != settings:
            raise Exception('The saved density extension was built with different settings, '
                            'rebuilding it.')

        if not all(np.array_equal(extData['gridFingerprint'][key], gridFingerprint[key])
                   for key in gridFingerprint):
            raise Exception('The saved density extension was built on a different simulation '
                            'grid, rebuilding it.')

        # The simulated density is not cached, so that it cannot go out of sync with
        # density_interp_data.pkl, and is put back on here
        extData['times'] = times
        extData['densSim'] = dens
        extData['solrz'] = solrz
        extData['solzz'] = solzz

        return extData

    except Exception as e:

        print(e)
        print('No saved density extension present, generating it.')

    # Plasma radius measured by AXUV vs. time, processing the shot first if it has not been yet
    generate_axuv_data(shotnum, diodeArrayNum)
    axuvTimes, axuvRadius = load_axuv_radius(shotnum, diodeArrayNum, rateKHz=rateKHz,
                                             tMin=tMin, tMax=tMax)

    # Equilibrium the extension is mapped along
    Rmesh, Zmesh, psi = load_eqdsk_flux(simulationName)

    # The eqdsk grid is regular, so the radial positions are the same at every axial position
    rEQDSK = Rmesh[0]

    # Midplane radii of the simulation grid, and the edge it is extended from
    rSim = solrz[:, 0]
    rEdge = rSim[-1]

    # The extension has to be on one radial grid for every experimental time, so it is put on
    # the eqdsk points out to the largest plasma radius of the shot. The density at each time
    # then falls to 0 at that time's plasma radius and stays 0 beyond it.
    # The largest plasma radius is added as the outermost point so the extension ends exactly
    # there, and the eqdsk points within half a cell of it are dropped so that the last two
    # surfaces do not sit on top of each other
    rMaxExt = np.max(axuvRadius)
    halfCell = 0.5 * np.mean(np.diff(rEQDSK))

    rExtGrid = rEQDSK[(rEQDSK > rEdge) & (rEQDSK < rMaxExt - halfCell)]
    rExtGrid = np.concatenate([rExtGrid, [rMaxExt]])

    # Axial grid of the extension, taken from the outermost simulated surface as that is the
    # one the extension joins onto
    zGrid = solzz[-1]
    rSimOuter = solrz[-1]

    # Radius of each point of the extension at every axial position of its flux surface
    rExtMapped = map_along_flux_surfaces(rExtGrid, zGrid, Rmesh, Zmesh, psi, rSimOuter)

    # The tanh fall-off is proportional to the density at the edge of the simulation grid, so
    # its shape is worked out once with an edge density of 1 and scaled by the edge density of
    # whichever timestep is being extended. This is the same extension extend_profile() makes,
    # without having to call it once per simulated timestep.
    shapeExt = np.zeros((len(axuvTimes), len(rExtGrid)))
    for i in range(len(axuvTimes)):

        rExt, shapeProfile = extend_profile(rSim, np.ones_like(rSim), axuvRadius[i],
                                            rEQDSK, tanhSteepness=tanhSteepness)

        # AXUV can see a plasma smaller than the simulation grid, in which case there is
        # nothing to extend and the shape stays 0
        if len(rExt) == 0:
            continue

        # Onto the common radial grid, with nothing outside this time's plasma radius
        shapeExt[i] = np.interp(rExtGrid, rExt, shapeProfile,
                                left=shapeProfile[0], right=0.0)

    extData = {'times': times,
               'densSim': dens,
               'solrz': solrz,
               'solzz': solzz,
               'axuvTimes': axuvTimes,
               'axuvRadius': axuvRadius,
               'rExtGrid': rExtGrid,
               'rExtMapped': rExtMapped,
               'zGrid': zGrid,
               'shapeExt': shapeExt,
               'Rmesh': Rmesh,
               'Zmesh': Zmesh,
               'psi': psi,
               'settings': settings,
               'gridFingerprint': gridFingerprint}

    #### Save the data

    # Everything except the simulated density, which is left in density_interp_data.pkl so
    # that the two cannot go out of sync and so that this file stays small
    saveData = {key: value for key, value in extData.items()
                if key not in ('times', 'densSim', 'solrz', 'solzz')}

    with open(savePath, 'wb') as saveFile:
        pickle.dump(saveData, saveFile)

    print(f'Saved the density extension to- \n {savePath}')

    return extData

def extend_midplane_profile(extData, profileSim, expTimeIdx):
    """
    Extend any midplane profile on the simulation grid out to the AXUV plasma radius.

    This is the same extension as the density's- the tanh fall-off of extend_profile(), scaled
    by the value of the profile at the edge of the simulation grid and stacked onto the outside
    of the simulated profile. It can be applied to any quantity that should fall to 0 at the
    AXUV plasma radius, e.g. the electron temperature.

    Parameters
    ----------
    extData : dict
        The output of build_density_extension().
    profileSim : np.array
        1D array of the profile at the midplane radii of the simulation grid,
        extData['solrz'][:, 0].
    expTimeIdx : int
        Index into extData['axuvTimes'] of the AXUV plasma radius the extension is shaped by.

    Returns
    -------
    rMid : np.array
        1D array of the midplane radii of the simulation grid, followed by those of the
        extension. [m]
    profileMid : np.array
        1D array of the extended profile on rMid.
    """

    rMid = np.concatenate([extData['solrz'][:, 0], extData['rExtGrid']])

    profileMid = np.concatenate([profileSim, profileSim[-1] * extData['shapeExt'][expTimeIdx]])

    return rMid, profileMid

def extended_midplane_density(extData, simTimeIdx, expTimeIdx):
    """
    Assemble the extended midplane density profile at one pair of times.

    The extension is its shape scaled by the midplane density at the edge of the simulation
    grid, stacked onto the outside of the simulated profile.

    Parameters
    ----------
    extData : dict
        The output of build_density_extension().
    simTimeIdx : int
        Index into extData['times'] of the simulated profile.
    expTimeIdx : int
        Index into extData['axuvTimes'] of the AXUV plasma radius the extension is shaped by.

    Returns
    -------
    rMid : np.array
        1D array of the midplane radii of the simulation grid, followed by those of the
        extension. [m]
    densMid : np.array
        1D array of the midplane density on rMid. [m^-3]
    """

    return extend_midplane_profile(extData, extData['densSim'][simTimeIdx, :, 0], expTimeIdx)

