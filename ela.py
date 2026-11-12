"""
This script loads the end loss analyzer (ELA) profiles.

There are two analyses of the same ELA data, each kept in config.elaDataDir (see elaSources)-
    'csv' (Aaron H.)  : one CSV file per time point, with errors on the ELA positions.
    'npz' (Tucker P.) : one density and one temperature file holding every time point, with the
                        number of each analyzer.
load_ela_data() reads either one into the same dict of [time, analyzer] arrays.

The CSV data is read from one CSV file per time point, named
f'Scan_{shotDatePrefix}_t{time}ms_*.csv' with a 'p' for the decimal point of the time, e.g.
'Scan_260709_t7p5ms_wp25ms_ne_Te_Ti.csv' is at 7.5 ms. Each file has a header of
'name,value' lines (the shots that were averaged, the half width of the time window, the ion
species, ...) followed by a table with one row per analyzer- the ELA position at the midplane and
its errors, the density, T_e and T_i, each with its 1 sigma error.

The NPZ data is read from f'{shotDatePrefix}_ELA_Density.npz' and
f'{shotDatePrefix}_ELA_Temp.npz', each holding the profile across the analyzers at several times,
with the keys of each time prefixed f'surrogate_t_{time}ms_', e.g. 'surrogate_t_4ms_density_m3'.
The T_e and T_i are taken from the temperature file.

In both, the density and the position of each analyzer have already been mapped back to the
midplane with GDT scaling. midplane_to_throat_density() takes the density on to the mirror throat,
and cql3d_simulation.map_midplane_radius_to_throat() does the same for the radius. GDT takes the
plasma to be isothermal along the flux tube, so the temperatures are the same at the midplane and
the throat.
"""

import csv
import glob
import os
import re

import numpy as np

import config

# GDT scaling of the density from the midplane to the mirror throat, n_throat / n_mid. The
# ambipolar potential drop from the midplane to the throat, where the outflow reaches the sound
# speed, is T_e/2e, so the Boltzmann electrons drop by exp(-1/2).
gdtThroatDensityFactor = 1 / np.sqrt(np.e)

# Who made each analysis of the ELA data, used to label the plots
elaSources = {'csv': 'Aaron H.',
              'npz': 'Tucker P.'}

# Columns of the table in each ELA CSV file, and the key each one is returned under
_columnKeys = {'ELA position at midplane [cm]': 'radius',
               'ELA position error minus [cm]': 'radiusErrMinus',
               'ELA position error plus [cm]': 'radiusErrPlus',
               'Density [m^-3]': 'dens',
               'Density error [m^-3]': 'densSigma',
               'Te [eV]': 'Te',
               'Te error [eV]': 'TeSigma',
               'Ti [eV]': 'Ti',
               'Ti error [eV]': 'TiSigma'}

# Columns given in [cm], returned in [m]
_cmColumns = ('radius', 'radiusErrMinus', 'radiusErrPlus')

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

def ela_files(shotDatePrefix, dataDir=None):
    """
    Find the ELA file of each time point of one shot day, in order of time.

    Parameters
    ----------
    shotDatePrefix : int
        Date prefix of the shots, e.g. 260709.
    dataDir : str
        Directory of the files.
        Default is None, which uses config.elaDataDir.

    Returns
    -------
    times : np.array
        1D array of the time point of each file, from its name. [s]
    paths : list of str
        Path of each file.
    """

    if dataDir is None:
        dataDir = config.elaDataDir

    found = []
    for path in glob.glob(os.path.join(dataDir, f'Scan_{shotDatePrefix}_t*ms_*.csv')):
        # e.g. '_t7p5ms_' -> 7.5 ms
        match = re.search(r'_t(\d+(?:p\d+)?)ms_', os.path.basename(path))
        if match is not None:
            found.append((float(match.group(1).replace('p', '.')), path))

    if not found:
        raise FileNotFoundError(f'No ELA files Scan_{shotDatePrefix}_t*ms_*.csv in {dataDir}')

    found.sort()
    timesMs, paths = zip(*found)

    # [ms] to [s]
    return np.array(timesMs)*1e-3, list(paths)

def _read_ela_csv(path):
    """
    Read the header and the table of one ELA file.

    Parameters
    ----------
    path : str
        Path of the file.

    Returns
    -------
    header : dict
        The 'name,value' header lines, {name: list of the values as str}.
    table : dict
        1D array of each column, under its key in _columnKeys, one entry per analyzer. Lengths
        are in [m].
    """

    with open(path, newline='') as f:
        rows = [row for row in csv.reader(f) if row]

    # The table starts at the row of column names
    tableStart = next(i for i, row in enumerate(rows) if row[0] in _columnKeys)

    header = {row[0]: row[1:] for row in rows[:tableStart]}

    columnNames = rows[tableStart]
    missing = set(_columnKeys) - set(columnNames)
    if missing:
        raise ValueError(f'{path} has no column(s) {sorted(missing)}')

    values = np.array(rows[tableStart + 1:], dtype=float)

    table = {}
    for i, name in enumerate(columnNames):
        if name in _columnKeys:
            key = _columnKeys[name]
            table[key] = values[:, i] * (1e-2 if key in _cmColumns else 1)

    return header, table

def load_ela_data(shotDatePrefix, source='csv', dataDir=None):
    """
    Load the ELA position, density, electron temperature and ion temperature at every time point
    of one shot day.

    Parameters
    ----------
    shotDatePrefix : int
        Date prefix of the shots, e.g. 260709.
    source : str
        Which analysis of the ELA data to load, a key of elaSources-
            'csv' : the CSV files of Aaron H. (_load_ela_csv_data()).
            'npz' : the NPZ files of Tucker P. (_load_ela_npz_data()).
        Default is 'csv'.
    dataDir : str
        Directory of the files.
        Default is None, which uses config.elaDataDir.

    Returns
    -------
    ela : dict
        'times'           : 1D array of the time points. [s]
        'radius'          : 2D array of the position of each analyzer at the midplane,
                            [time, analyzer]. [m]
        'radiusErrMinus'  : 2D array of the error on the position towards the axis,
                            [time, analyzer]. [m]
        'radiusErrPlus'   : 2D array of the error on the position away from the axis,
                            [time, analyzer]. [m]
        'dens'            : 2D array of the density at the midplane, [time, analyzer]. [m^-3]
                            midplane_to_throat_density() maps it on to the mirror throat.
        'densSigma'       : 2D array of the 1 sigma error on the density, [time, analyzer]. [m^-3]
        'Te', 'TeSigma'   : 2D arrays of the electron temperature and its 1 sigma error,
                            [time, analyzer]. [eV]
        'Ti', 'TiSigma'   : 2D arrays of the ion temperature and its 1 sigma error,
                            [time, analyzer]. [eV]
        'windowHalfWidth' : Half width of the time window around each time point. [s]
        'shots'           : List of the shot numbers averaged over (CSV only, empty for NPZ).
        'analyzerNum'     : 2D array of the number of each analyzer, [time, analyzer]
                            (NPZ only, None for CSV).
        'source'          : The source, as passed in.
        'label'           : Who made the analysis, elaSources[source].
        The NPZ files give no error on the positions, so its radiusErrMinus and radiusErrPlus
        are zero.
    """

    loaders = {'csv': _load_ela_csv_data, 'npz': _load_ela_npz_data}
    if source not in loaders:
        raise ValueError(f'source must be one of {list(loaders)}, not {source!r}')

    ela = loaders[source](shotDatePrefix, dataDir)
    ela['source'] = source
    ela['label'] = elaSources[source]

    return ela

def _load_ela_csv_data(shotDatePrefix, dataDir=None):
    """
    Load the CSV files of every time point of one shot day, see load_ela_data().
    """

    times, paths = ela_files(shotDatePrefix, dataDir)

    headers, tables = zip(*[_read_ela_csv(path) for path in paths])

    halfWidths = {float(header['Window half width [ms]'][0]) for header in headers}
    if len(halfWidths) > 1:
        raise ValueError(f'The ELA files of {shotDatePrefix} have different time windows, '
                         f'{sorted(halfWidths)} ms')

    nAnalyzers = {len(table['radius']) for table in tables}
    if len(nAnalyzers) > 1:
        raise ValueError(f'The ELA files of {shotDatePrefix} have different numbers of analyzers')

    ela = {key: np.array([table[key] for table in tables]) for key in _columnKeys.values()}
    ela['times'] = times
    # [ms] to [s]
    ela['windowHalfWidth'] = halfWidths.pop()*1e-3
    ela['shots'] = [int(shot) for shot in headers[0].get('Shots', [])]
    ela['analyzerNum'] = None

    return ela

def _load_ela_npz_file(path, quantityKeys):
    """
    Load the radius, analyzer number and the requested quantities at every time of one ELA NPZ
    file.

    Parameters
    ----------
    path : str
        Path of the file.
    quantityKeys : list of str
        Keys of the quantities to load, without the f'surrogate_t_{time}ms_' prefix.

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

    data = np.load(path)

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

def _load_ela_npz_data(shotDatePrefix, dataDir=None):
    """
    Load the NPZ density and temperature files of one shot day, see load_ela_data().
    """

    if dataDir is None:
        dataDir = config.elaDataDir

    times, radius, analyzerNum, dens, windowHalfWidth = _load_ela_npz_file(
        os.path.join(dataDir, f'{shotDatePrefix}_ELA_Density.npz'),
        ['density_m3', 'density_error_m3'])
    tempTimes, tempRadius, _, temp, _ = _load_ela_npz_file(
        os.path.join(dataDir, f'{shotDatePrefix}_ELA_Temp.npz'),
        ['Te_eV', 'Te_err_eV', 'Ti_eV', 'Ti_err_eV'])

    if not (np.allclose(times, tempTimes) and np.allclose(radius, tempRadius)):
        raise ValueError(f'The ELA density and temperature files of {shotDatePrefix} are on '
                         'different times or radii')

    return {'times': times,
            'radius': radius,
            'radiusErrMinus': np.zeros_like(radius),
            'radiusErrPlus': np.zeros_like(radius),
            'dens': dens['density_m3'],
            'densSigma': dens['density_error_m3'],
            'Te': temp['Te_eV'],
            'TeSigma': temp['Te_err_eV'],
            'Ti': temp['Ti_eV'],
            'TiSigma': temp['Ti_err_eV'],
            'windowHalfWidth': windowHalfWidth,
            'shots': [],
            'analyzerNum': analyzerNum}
