"""
This script loads the flux loop measurements and calculates the synthetic flux loop signal of a
simulation.

WHAM has two flux loops. Their measured signal, its uncertainty and the (R, Z) location of each
loop are stored in the same mirror_fit data.yaml file as the Thomson scattering data
(config.fluxLoopDataPath).

The measured signal is the change in the poloidal flux linked by each loop caused by the plasma
(its diamagnetism). The synthetic signal is the same change in the simulation: the flux through
each loop in the eqdsk of the initialization (the first IPS timestep) minus the flux in the eqdsk
of the IPS timestep being compared, as in pleiades (FluxSynth = 2*pi*|psi_vac - psi|).
"""

import yaml
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import config
from cql3d_simulation import load_eqdsk_flux

def load_flux_loops(dataPath=None):
    """
    Load the flux loop measurements from a mirror_fit data.yaml file.

    Parameters
    ----------
    dataPath : str
        Path to the data.yaml file.
        Default is None, which loads config.fluxLoopDataPath.

    Returns
    -------
    R : np.array
        Radius of each flux loop. [m]
    Z : np.array
        Axial position of each flux loop. [m]
    flux : np.array
        Measured flux of each loop. [Wb]
    fluxSigma : np.array
        1 sigma error on the measured flux. [Wb]
    time : float
        Time of the measurement. [s]
    """

    if dataPath is None:
        dataPath = config.fluxLoopDataPath

    with open(dataPath, 'r') as f:
        data = yaml.safe_load(f)

    fluxLoops = data['diagnostics']['flux_loops']

    R = np.array([rz[0] for rz in fluxLoops['RZ']], dtype=float)
    Z = np.array([rz[1] for rz in fluxLoops['RZ']], dtype=float)
    # Only the first time is used
    # dtype=float is needed because YAML 1.1 reads numbers like 5e-6 (no decimal point) as strings
    flux = np.array(fluxLoops['measurements'][0], dtype=float)
    fluxSigma = np.array(fluxLoops['sigma_measurements'][0], dtype=float)
    time = float(data['times'][0])

    return R, Z, flux, fluxSigma, time

def eqdsk_flux_at_points(Rmesh, Zmesh, psi, R, Z):
    """
    Interpolate the eqdsk poloidal flux onto a set of (R, Z) points.

    Parameters
    ----------
    Rmesh : np.array
        2D array [z x r] of the radial coordinates of the eqdsk grid. [m]
    Zmesh : np.array
        2D array [z x r] of the axial coordinates of the eqdsk grid. [m]
    psi : np.array
        2D array [z x r] of the poloidal flux on the eqdsk grid. [Wb/rad]
    R : np.array
        Radius of each point. [m]
    Z : np.array
        Axial position of each point. [m]

    Returns
    -------
    psiPoints : np.array
        Poloidal flux at each point. [Wb/rad]
    """

    interp = RegularGridInterpolator((Zmesh[:, 0], Rmesh[0]), psi, method='cubic')

    return interp(np.column_stack([Z, R]))

def synthetic_flux_loop_signal(simulationName, R, Z, timestepIdx, refTimestepIdx=0):
    """
    Calculate the synthetic flux loop signal of the simulation at a set of (R, Z) points.

    The eqdsk (written by pleiades) stores the poloidal flux per radian, psi, so the flux linked
    by a loop at (R, Z) is 2*pi*psi(R, Z). The signal is the change in that flux from the
    reference (initialization) eqdsk to the eqdsk of the given IPS timestep,

        flux = 2*pi * (psi_ref(R, Z) - psi(R, Z))

    This is positive when the plasma is diamagnetic, i.e. when it pushes flux out of the loop. It
    is the same as pleiades' FluxSynth = 2*pi*|psi_vac - psi|, but keeps the sign.

    Parameters
    ----------
    simulationName : str
        The name of the simulation.
    R : np.array
        Radius of each flux loop. [m]
    Z : np.array
        Axial position of each flux loop. [m]
    timestepIdx : int
        Index of the IPS timestep (in the order of cql3d_simulation.ips_timestep_dirs()) to compare.
    refTimestepIdx : int
        Index of the IPS timestep of the reference eqdsk.
        Default is 0, the first IPS timestep, whose eqdsk is the one the simulation is
        initialized with.

    Returns
    -------
    flux : np.array
        Synthetic flux of each loop. [Wb]
    """

    RmeshRef, ZmeshRef, psiRef = load_eqdsk_flux(simulationName, refTimestepIdx)
    Rmesh, Zmesh, psi = load_eqdsk_flux(simulationName, timestepIdx)

    psiRefPoints = eqdsk_flux_at_points(RmeshRef, ZmeshRef, psiRef, R, Z)
    psiPoints = eqdsk_flux_at_points(Rmesh, Zmesh, psi, R, Z)

    return 2*np.pi * (psiRefPoints - psiPoints)
