"""
This script loads the ADAS photon emissivity coefficients (PECs) the synthetic line emission is
calculated from, e.g. the H-alpha emission seen by the AXUV array behind the H-alpha filter.

The ADF15 files are read with the aurora package, which lives in its own environment
(config.auroraPython). The main pipeline does not need aurora: the first time a PEC is asked
for, this file is run as a script in the aurora environment, which downloads the ADF15 file from
OPEN-ADAS into config.atomicDataDir and saves the PEC of the requested line there as an NPZ
table. After that the table is read with numpy alone.
"""

import os
import sys
import subprocess
import numpy as np

from scipy.interpolate import RegularGridInterpolator

import config

# H-alpha (n = 3 -> 2) of neutral hydrogen, n-resolved Balmer series of the 2012 ADAS data
halphaADF15File = 'pec12#h_balmer#h0.dat'
halphaWavelengthA = 6561.9

def pec_table_path(adf15File, wavelengthA):
    """
    Path of the saved PEC table of one line of an ADF15 file.

    Parameters
    ----------
    adf15File : str
        Name of the ADF15 file, e.g. 'pec12#h_balmer#h0.dat'.
    wavelengthA : float
        Wavelength of the line. [Angstrom]

    Returns
    -------
    tablePath : str
        Path of the NPZ table inside config.atomicDataDir.
    """

    # '#' is part of the ADAS naming but awkward in a filename
    fileStem = os.path.splitext(adf15File)[0].replace('#', '_')

    return config.atomicDataDir + f'{fileStem}_{wavelengthA:.1f}A.npz'

def write_pec_table(adf15File, wavelengthA, tablePath, toleranceA=1.0):
    """
    Read the excitation and recombination PECs of one line from an ADF15 file with aurora, and
    save them in SI units.

    This needs aurora, so it is only run in the aurora environment, by generate_pec_table().

    Parameters
    ----------
    adf15File : str
        Name of the ADF15 file, e.g. 'pec12#h_balmer#h0.dat'.
        It is downloaded from OPEN-ADAS into config.atomicDataDir if it is not there yet.
    wavelengthA : float
        Wavelength of the line. [Angstrom]
        The closest line of each type within toleranceA is used.
    tablePath : str
        Path of the NPZ table to save.
    toleranceA : float
        Largest difference between wavelengthA and the wavelength of the line in the file.
        [Angstrom]
        Default is 1.
    """

    import aurora

    # Keep the downloaded ADAS files next to the tables, rather than inside the shared aurora
    # environment
    os.environ['AURORA_ADAS_DIR'] = config.atomicDataDir
    adf15Path = aurora.get_adas_file_loc(adf15File, filetype='adf15')

    trs = aurora.read_adf15(adf15Path)

    saveData = {}
    for pecType in ('excit', 'recom'):

        lines = trs[trs['type'] == pecType]
        lineIdx = np.argmin(np.abs(lines['lambda [A]'].values - wavelengthA))
        line = lines.iloc[lineIdx]

        if np.abs(line['lambda [A]'] - wavelengthA) > toleranceA:
            raise ValueError(f'No {pecType} line within {toleranceA} A of {wavelengthA} A in '
                             f'{adf15File}. The closest is at {line["lambda [A]"]} A.')

        # [cm^-3] to [m^-3], [cm^3 s^-1] to [m^3 s^-1]
        saveData[f'dens_{pecType}'] = line['dens pnts'] * 1e6
        saveData[f'temp_{pecType}'] = line['temp pnts']
        saveData[f'pec_{pecType}'] = line['PEC pnts'] * 1e-6
        saveData[f'lambda_{pecType}'] = line['lambda [A]']

    np.savez(tablePath, adf15File=adf15File, **saveData)

    return

def generate_pec_table(adf15File, wavelengthA):
    """
    Make the PEC table of one line if it is not in config.atomicDataDir yet.

    This runs this file in the aurora environment (config.auroraPython), which calls
    write_pec_table().

    Parameters
    ----------
    adf15File : str
        Name of the ADF15 file, e.g. 'pec12#h_balmer#h0.dat'.
    wavelengthA : float
        Wavelength of the line. [Angstrom]
    """

    tablePath = pec_table_path(adf15File, wavelengthA)

    if os.path.exists(tablePath):
        return

    print(f'No PEC table present for the {wavelengthA} A line of {adf15File}, generating it.')

    os.makedirs(config.atomicDataDir, exist_ok=True)

    subprocess.run([config.auroraPython, os.path.abspath(__file__),
                    adf15File, f'{wavelengthA}', tablePath],
                   cwd=config.repoDir, check=True)

    return

def load_pec_table(adf15File=halphaADF15File, wavelengthA=halphaWavelengthA):
    """
    Load the excitation and recombination PECs of one line, making the table if needed.

    Parameters
    ----------
    adf15File : str
        Name of the ADF15 file.
        Default is the n-resolved Balmer series of neutral hydrogen.
    wavelengthA : float
        Wavelength of the line. [Angstrom]
        Default is H-alpha.

    Returns
    -------
    pecTable : dict
        For each pecType in ('excit', 'recom')-
        f'dens_{pecType}'   : [n], electron density grid. [m^-3]
        f'temp_{pecType}'   : [T], electron temperature grid. [eV]
        f'pec_{pecType}'    : [n x T], PEC. [photons m^3 s^-1]
        f'lambda_{pecType}' : wavelength of the line in the file. [Angstrom]
        and 'adf15File', the name of the ADF15 file.
    """

    generate_pec_table(adf15File, wavelengthA)

    with np.load(pec_table_path(adf15File, wavelengthA)) as loadFile:
        pecTable = {key: loadFile[key] for key in loadFile.files}

    pecTable['adf15File'] = str(pecTable['adf15File'])

    return pecTable

def evaluate_pec(pecTable, pecType, ne, Te):
    """
    Interpolate a PEC onto the given electron densities and temperatures.

    The interpolation is linear in log10(PEC) vs. (log10(n_e), log10(T_e)), as in aurora. Points
    outside the ADAS grid are moved onto its edge rather than extrapolated.

    Parameters
    ----------
    pecTable : dict
        The PEC table, from load_pec_table().
    pecType : str
        'excit' for electron impact excitation (emission scales with n_e * n_neutral) or 'recom'
        for recombination (emission scales with n_e * n_ion).
    ne : np.array
        Electron density. [m^-3]
    Te : np.array
        Electron temperature. [eV]

    Returns
    -------
    pec : np.array
        The PEC at each point, the same shape as ne. [photons m^3 s^-1]
    """

    logDens = np.log10(pecTable[f'dens_{pecType}'])
    logTemp = np.log10(pecTable[f'temp_{pecType}'])

    interpolator = RegularGridInterpolator((logDens, logTemp),
                                           np.log10(pecTable[f'pec_{pecType}']))

    points = np.stack((np.clip(np.log10(ne), logDens[0], logDens[-1]),
                       np.clip(np.log10(Te), logTemp[0], logTemp[-1])), axis=-1)

    return 10**interpolator(points)

if __name__ == "__main__":

    # Run in the aurora environment by generate_pec_table()
    adf15File, wavelengthA, tablePath = sys.argv[1], float(sys.argv[2]), sys.argv[3]

    write_pec_table(adf15File, wavelengthA, tablePath)
