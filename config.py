"""
Locations of the simulations, the diagnostic data and the outputs of the pipeline.

Every module reads these at call time as config.<name>, so changing one from a script, e.g.

    import config
    config.simulationScanDir = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/'

points the whole pipeline at the new location.
"""

import os

# Directory of this repository
repoDir = os.path.dirname(os.path.abspath(__file__)) + '/'

#### Simulations

# Directory holding one sub-directory per CQL3D + KN1D (IPS) simulation
simulationScanDir = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/withRadialDiff/'

#### SEE (shine-through) array

# The SEE detectors are processed in the shinethru repository
shinethruDir = '/home/sanwalka/shinethru/'

# Geometry (lines of sight, impact parameters) of the SEE detectors
seeDetectorDictPath = shinethruDir + 'lookup_tables/see_detector_dictionary.pkl'

# Processed SEE data, stored as f'{shotnum}.pkl'
seeDataDir = shinethruDir + 'data/'

# Script that processes a shot from MDSplus into seeDataDir, run when a shot is not there yet
seeProcessingScript = shinethruDir + 'compare_to_simulation/experimental_data.py'

#### AXUV

# Processed AXUV data, stored as f'AXUV_results*{shotnum}*DA{diodeArrayNum}*.npz'
axuvDataDir = shinethruDir + 'data/'

# Directory of axuv_data_output.py, which processes a shot into axuvDataDir, run when a shot
# is not there yet
axuvScriptDir = shinethruDir + 'axuv_scripts/'

#### Thomson scattering

# Thomson scattering data, in the mirror_fit input format
thomsonDataPath = '/home/sanwalka/mirror_fit/1kev_day/data.yaml'

#### End loss analyzers (ELA)

# Processed ELA data, stored as f'{shotDatePrefix}_ELA_Density.npz' and
# f'{shotDatePrefix}_ELA_Temp.npz'
elaDataDir = repoDir + 'ela_data/'

#### Atomic data

# Python of the environment with the aurora package, used to read the ADAS PEC files
auroraPython = '/share/envs/aurora_env/bin/python'

# ADAS files downloaded by aurora, and the PEC tables read from them
atomicDataDir = repoDir + 'atomic_data/'

#### Outputs

# Plots made by the pipeline
plotDir = repoDir + 'plots/'
