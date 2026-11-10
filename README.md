# Integrated Data Analysis

Compares CQL3D + KN1D (IPS) simulations of WHAM against several diagnostics at once.

Each diagnostic has a module that loads the measured data and, where it applies, runs a synthetic version of that diagnostic on the simulated density. The plotting scripts then put the simulation and the measurements side by side.

The code started as the SEE (shine-through) comparison in `shinethru/compare_to_simulation/`. It currently includes the SEE array, AXUV (the plasma radius from DIODEARRAY1 and the H-alpha signal from DIODEARRAY2) and Thomson scattering. On the simulation side it loads the CQL3D density, temperatures and flux tubes, and the KN1D neutral particle source rates and neutral densities.

## Quick start

Each comparison script takes the standard inputs in its `__main__` block: the simulation name, the shot number, and the simulated and experimental times in seconds. Edit those, then run the script:

```bash
cd /home/sanwalka/integrated_data_analysis
~/anaconda3/bin/python density_comparison.py           # SEE array (+ Thomson)
~/anaconda3/bin/python axuv_comparison.py              # H-alpha AXUV (DIODEARRAY2)
~/anaconda3/bin/python multi_diagnostic_comparison.py  # all of the above (+ Thomson n_e and T_e, + ELA, + flux loops) in one figure
~/anaconda3/bin/python electron_temperature_comparison.py  # simulated T_e vs. Thomson
~/anaconda3/bin/python fluxloops_comparison.py         # flux loops vs. the change in the simulated eqdsk flux
~/anaconda3/bin/python ela_comparison.py               # end loss analyzer n, T_e, T_i at the mirror throat (+ simulated n, T_e, T_i there)
```

| Script | Plots | Saved as (in `plots/`) |
|---|---|---|
| `density_comparison.py` | Simulated midplane density and its synthetic SEE signal against the measured SEE signal, optionally with Thomson n<sub>e</sub>. `plot_profile_comparison()` shows two simulation/shot pairs side by side. | `{shotnum1}_{shotnum2}_profile_comparison.png` |
| `axuv_comparison.py` | Simulated midplane density and T<sub>e</sub> (CQL3D grid only, no AXUV extension), the KN1D midplane neutral density per flux tube [m<sup>-3</sup>], the simulated H-alpha emissivity per flux tube calculated from those three, and the measured line-integrated H-alpha signal of each DIODEARRAY2 chord against the synthetic one, both normalized to their peak. | `{shotnum}_{simulationName}_axuv_DA2_comparison.png` |
| `multi_diagnostic_comparison.py` | The headline comparison of each diagnostic for one simulation against one shot. The left column has one panel each on the same x axis: the synthetic vs. measured SEE signal (∫n<sub>i</sub> dl), the simulated n<sub>i</sub> profile vs. Thomson n<sub>e</sub>, the simulated vs. Thomson T<sub>e</sub>, and the synthetic vs. measured H-alpha signal (normalized). The right column has the ELA n, T<sub>e</sub> and T<sub>i</sub> panels of `ela_comparison.py`, at the ELA time point closest to the experimental time only, on their own x axis (radius at the mirror throat). Below them is the flux loop panel of `fluxloops_comparison.py` (measured vs. synthetic flux against Z). The ELA files are found by `elaShotDatePrefix`, which defaults to the date of the shot (its first 6 digits). The detailed plots stay in `density_comparison.py`, `axuv_comparison.py`, `electron_temperature_comparison.py`, `ela_comparison.py` and `fluxloops_comparison.py`. | `{shotnum}_{simulationName}_multi_diagnostic_comparison.png` |
| `electron_temperature_comparison.py` | Simulated midplane T<sub>e</sub> (CQL3D grid only) against the Thomson T<sub>e</sub> profile. The Thomson data is not tied to a shot, so its `__main__` takes only the simulation name and the simulated time. | `{simulationName}_electron_temperature_comparison.png` |
| `fluxloops_comparison.py` | The measured flux of the two flux loops (with its 1σ error) against the synthetic one, in µWb vs. the Z of each loop. The synthetic flux is the change in 2πψ at each loop from the initialization eqdsk (IPS step 1) to the eqdsk of the IPS timestep that covers `simTime` (`timestepIdx=` picks a timestep directly). Like the Thomson data, the flux loop data is not tied to a shot, so its `__main__` takes only the simulation name and the simulated time. | `{simulationName}_fluxloops_comparison.png` |
| `ela_comparison.py` | The end loss analyzer (ELA) density, T<sub>e</sub> and T<sub>i</sub> against the radius of each analyzer, one curve per time point (4-8 ms), from the files in `config.elaDataDir`. The ELA data is given at the midplane (mapped there with GDT scaling), and is taken on to the mirror throat, where the simulated plasma is Maxwellian: the density is multiplied by 1/√e (the GDT drop from the midplane to the throat), the radius of each analyzer is followed along its flux surface in the eqdsk of the IPS timestep that covers `simTime` (`map_midplane_radius_to_throat`), and the temperatures are left alone (GDT is isothermal along the flux tube). The simulated density, T<sub>e</sub> and T<sub>i</sub> at the mirror throat are drawn on each panel, and everything is against the radius at the mirror throat. Its `__main__` takes the simulation name, the shot date prefix (e.g. 260709) and the simulated time. Pass `expTime=` to draw only the time point closest to it. | `{shotDatePrefix}_{simulationName}_ela_comparison.png` |

From your own script:

```python
from density_comparison import plot_profile_comparison

fig, axs = plot_profile_comparison(
    'nneut_3e16_gb_1e18_NBI_800kW_ECH_0kW_ionDrrOn',   # simName1
    'nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW_ionDrrOn',   # simName2
    260709061, 261012050,                              # shotnum1, shotnum2
    7.83e-3, 7.83e-3,                                  # expTime1, expTime2 [s]
    3e-3, 3e-3,                                        # simTime1, simTime2 [s]
    withThomson1=True, withThomson2=False,
    detectorsToDrop1=[15], detectorsToDrop2=[9, 13, 15],
    useExtension=True, saveplot=True)
```

## Layout

| Module | What it holds |
|---|---|
| `config.py` | Every path the pipeline reads from or writes to. This is the only file you should need to edit to point the pipeline somewhere else. |
| `cql3d_simulation.py` | Loads the CQL3D simulation. This covers the IPS timestep directories (`ips_timestep_dirs`), the 2D ion density (`load_density_data`, `generate_times_and_functions`), its (r, z) interpolation, the ion density at the mirror throat, where each flux surface is closest to the axis (`load_throat_density`), the midplane electron and ion temperatures (`load_temperature_data`) and those at the mirror throat (`load_throat_temperature`, from `energyz`), the flux tube radii and volumes at each IPS timestep (`load_flux_tube_data`, `flux_tube_edges`), the eqdsk flux of any IPS timestep (`load_eqdsk_flux`, with `ips_timestep_index` to find the timestep that covers a simulation time), the mapping of midplane radii to the mirror throat along the eqdsk flux surfaces (`map_midplane_radius_to_throat`; the throat is where each surface is closest to the axis, and it matches the CQL3D throat radii to <0.1%), dropping the CQL3D restart slices, and parsing the neutral densities out of the simulation name. |
| `kn1d_simulation.py` | Loads the KN1D source rates (`load_source_rate_data`) and neutral densities (`load_neutral_density_data`), and puts them on the CQL3D flux tubes (`source_rate_per_flux_tube` in s<sup>-1</sup>, `neutral_density_per_flux_tube`). Also separates the gas-box contribution from the main vessel one (`disambiguate_source_rate`). Ported from `kn1dc_parser.py`; see [KN1D source rates and neutral densities](#kn1d-source-rates-and-neutral-densities). |
| `axuv.py` | Loads the AXUV diode arrays. It returns the plasma radius vs. time (`load_axuv_radius`) and the time-averaged line-integrated signal of each chord (`load_axuv_signal`). It also processes a shot that has not been processed yet (`generate_axuv_data`). The chord geometry is here too: the pinhole of each array (`pinholePositions`, (x, y) in m with x east to west and y bottom to top), the impact parameters (`load_axuv_impact_params`), the direction of each chord from the pinhole and its impact parameter (`chord_directions`, b = P<sub>x</sub>d<sub>y</sub> − P<sub>y</sub>d<sub>x</sub>, the same convention as the shinethru AXUV code), and where each chord crosses a circle around the axis (`chord_circle_intersections`). |
| `axuv_extension.py` | Extends the CQL3D density beyond the edge of its grid out to the AXUV plasma radius. The fall-off is a tanh, mapped along the eqdsk flux surfaces (`build_density_extension`). `extend_midplane_profile` applies the same fall-off to any midplane profile (e.g. T<sub>e</sub>), and `extended_midplane_density` uses it for the density at one pair of times. |
| `see_array.py` | The SEE detector array. It loads the measured line-integrated density and makes it up-down symmetric (`load_experimental_data`). It also runs the synthetic diagnostic on the plain simulation (`time_dependent_see_detector`) or on the extended one (`time_dependent_see_detector_extended`). |
| `thomson.py` | Loads the Thomson scattering electron density (`load_thomson_density`) and temperature (`load_thomson_temperature`) profiles. |
| `ela.py` | Loads the end loss analyzer density (`load_ela_density`) and electron and ion temperature (`load_ela_temperature`) profiles, as [time, analyzer] arrays, from `f'{shotDatePrefix}_ELA_Density.npz'` and `f'{shotDatePrefix}_ELA_Temp.npz'`. The density and the radius of each analyzer in these files have already been mapped back to the midplane with GDT scaling. `midplane_to_throat_density` takes the density on to the mirror throat, n<sub>throat</sub> = n<sub>mid</sub>/√e (`gdtThroatDensityFactor`); the radius is mapped with `cql3d_simulation.map_midplane_radius_to_throat`. The temperatures need no mapping, since GDT is isothermal along the flux tube. |
| `fluxloops.py` | Loads the flux loop measurements, their 1σ errors and the (R, Z) of each loop (`load_flux_loops`) from the same data.yaml as Thomson (`config.fluxLoopDataPath`). `synthetic_flux_loop_signal` gives the simulated signal, Φ = 2π(ψ<sub>ref</sub> − ψ) at each loop, with ψ interpolated (cubic) from the eqdsk of the reference IPS timestep (default: the first, which is the initialization eqdsk) and of the timestep being compared. The eqdsk ψ is per radian, the pleiades convention (pleiades' own synthetic flux loop signal is 2π&#124;ψ<sub>vac</sub> − ψ&#124;). This was checked by B<sub>0</sub> = 2ψ/r² near the axis matching CQL3D's `bmidplne` (0.408 T). The sign is kept, so a diamagnetic plasma gives a positive flux. |
| `atomic_data.py` | ADAS photon emissivity coefficients (PECs) for synthetic line emission. `load_pec_table` gives the excitation and recombination PECs of one line (H-alpha by default) in SI units, and `evaluate_pec` interpolates them onto n<sub>e</sub>, T<sub>e</sub>. The ADF15 file is read with aurora, which is run in its own environment; see [Atomic data](#atomic-data). |
| `plotting.py` | Plot style (font size, TkAgg backend), `save_plot`, and `plot_mirrored_profile` (draws a midplane profile mirrored about r = 0, simulated part solid and extension dashed). Shared by every comparison script so their panels can be combined. |
| `density_comparison.py` | SEE plots: `plot_radial_and_synthetic`, `plot_radial_and_synthetic_with_thomson`, `plot_profile_comparison`. |
| `axuv_comparison.py` | H-alpha AXUV plots: `load_simulated_profiles` (density, T<sub>e</sub> and the KN1D neutral density per flux tube at one time), `halpha_emissivity` (the H-alpha emissivity per flux tube from those profiles), `synthetic_halpha_signal` (its line integral along each chord), `add_halpha_signal_to_axis` (the normalized measured vs. synthetic signal panel), `plot_axuv_comparison`, and `plot_axuv_chords` (the line of sight of each chord in the midplane cross-section, to check the geometry). |
| `electron_temperature_comparison.py` | T<sub>e</sub> plots: `add_thomson_temperature_to_axis` (overlays the Thomson T<sub>e</sub> on any radial profile axis in eV, e.g. the T<sub>e</sub> panel of `plot_axuv_comparison`) and `plot_electron_temperature_comparison`. |

The modules fall into three layers. Each one only imports from the layers above it, so nothing imports in a circle:

```
config
  │
  ├─ Simulation        cql3d_simulation ──> axuv_extension
  │                         └──> kn1d_simulation  ^
  ├─ Diagnostics       axuv ─────────────────────┘
  │                    see_array   (synthetic SEE uses cql3d_simulation + axuv_extension)
  │                    thomson
  │                    fluxloops   (synthetic signal uses the cql3d_simulation eqdsk)
  │                    ela
  │                    atomic_data (ADAS PECs, read with aurora in its own environment)
  │
  └─ Comparisons       plotting
                       density_comparison           (SEE + Thomson)
                       axuv_comparison              (H-alpha AXUV)
                       electron_temperature_comparison (T_e vs. Thomson)
                       fluxloops_comparison         (flux loops)
                       ela_comparison               (ELA n, T_e, T_i at the mirror throat)
                       multi_diagnostic_comparison  (headline panel of each of the above)
```

Every plotting function takes an `axes=` argument and `showPlot=False`. This lets the panels for several diagnostics be drawn into one figure for the same simulation, e.g. `plot_radial_and_synthetic(..., axes=(ax1, ax2))` next to `plot_axuv_comparison(..., axes=(ax3, ax4, ax5, ax6, ax7))`.

## Data flow of `plot_profile_comparison()`

For each of the two columns:

1. **Simulation.** `load_density_data()` reads `density_interp_data.pkl` from the simulation directory.
2. **Extension** (`useExtension=True`). `build_density_extension()` loads the AXUV plasma radius of the shot. It then adds a tanh fall-off from the edge density of the CQL3D grid out to that radius. The extension depends on the experimental time, so `simTime` and `expTime` are used as a pair.
3. **Synthetic SEE.** The density is line-integrated along each SEE detector's line of sight.
4. **Measured SEE.** `load_experimental_data()` reads the processed shot, drops `detectorsToDrop` and makes the profile up-down symmetric.
5. **Thomson** (`withThomson=True`). `add_thomson_to_axis()` overlays the Thomson n<sub>e</sub> on the radial profile panel.

Both columns then get the same x axis, and each row gets the same y axis.

## Data flow of `plot_axuv_comparison()`

The simulated profiles in this plot are on the CQL3D grid only; the AXUV extension is not applied. (`extend_midplane_profile()` is still available in `axuv_extension.py` if you want it back.)

1. **Density.** `load_density_data()` gives the midplane density at the CQL3D timestep closest to `simTime`.
2. **Electron temperature.** `load_temperature_data()` gives T<sub>e</sub> = (2/3)⟨E⟩ of the electron distribution function at the midplane (`energym`). CQL3D's `temp` variable isn't used because it is only updated for Maxwellian species, and here the electrons are a general species.
3. **Neutral density.** `neutral_density_per_flux_tube()` gives the midplane neutral density of the main vessel (midplane KN1D run) in each CQL3D flux tube, in m<sup>-3</sup>, at the end of the IPS timestep closest to `simTime`. This is the neutral density the H-alpha emission is calculated from. The panel shows one point per flux tube at its midplane radius, on a log axis, with the time used (t<sub>KN1D</sub>) in the legend. The source rate per flux tube is still available from `source_rate_per_flux_tube()`.
4. **H-alpha emissivity.** `halpha_emissivity()` interpolates the density and T<sub>e</sub> onto the flux tube radii and calculates, per flux tube,

   ε<sub>Hα</sub> = n<sub>e</sub> n<sub>H</sub> PEC<sub>excit</sub>(n<sub>e</sub>, T<sub>e</sub>) + n<sub>e</sub> n<sub>i</sub> PEC<sub>recom</sub>(n<sub>e</sub>, T<sub>e</sub>)   [photons m<sup>-3</sup> s<sup>-1</sup>]

   with n<sub>e</sub> = n<sub>i</sub> (quasi-neutral hydrogen) and the PECs of `pec12#h_balmer#h0.dat` (see [Atomic data](#atomic-data)). The panel shows the total and both parts on a log axis. In the core, where n<sub>H</sub> is ~10<sup>13</sup> m<sup>-3</sup>, recombination can be a large fraction of the total (~45% for `nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW_ionDrrOn` at 3 ms). What is left out:
   - The PECs assume Maxwellian electrons at T<sub>e</sub>, while T<sub>e</sub> here is (2/3)⟨E⟩ of the non-Maxwellian CQL3D electrons.
   - Only atomic neutrals: no molecular contribution (e.g. dissociative excitation of H<sub>2</sub>), and no charge exchange emission from the fast ions.
   - The PECs are for hydrogen. For a deuterium plasma, D-alpha is still well inside the 654-659 nm filter and the change in the PECs is small.
5. **H-alpha signal.** `load_axuv_signal()` averages the raw signal of each DIODEARRAY2 chord (`signalDiodeArrayNum`) over a 1/`rateKHz` window centred on `expTime`. The error bars are the standard deviation within that window, i.e. the noise and fluctuation level.
6. **Synthetic H-alpha signal.** `synthetic_halpha_signal()` integrates the emissivity along the line of sight of each chord (from `axuv.chord_directions()`, see `plot_axuv_chords()`), in photons m<sup>-2</sup> s<sup>-1</sup>. Along a chord the emissivity is interpolated linearly between the flux tube centres, held at the innermost tube's value inside its centre (the simulation does not reach r = 0), held at the outermost tube's value out to that tube's outer edge (`flux_tube_edges`), and zero beyond it. A chord that misses the simulated plasma gets zero.

The measured signal (nA) and the synthetic one are not absolutely calibrated against each other, so the bottom panel shows each normalized to its own peak. Only the shapes are compared.

## Atomic data

The PECs come from the ADAS ADF15 files, read with the aurora package. aurora is not in the main Python environment, so `atomic_data.py` runs itself as a script with `config.auroraPython` (the `aurora_env` conda environment) the first time a line is asked for. That run downloads the ADF15 file from OPEN-ADAS into `config.atomicDataDir` (via `AURORA_ADAS_DIR`, so nothing is written into the shared aurora environment). It then saves the excitation and recombination PECs of the line, converted to SI (n<sub>e</sub> in m<sup>-3</sup>, T<sub>e</sub> in eV, PEC in photons m<sup>3</sup> s<sup>-1</sup>), as an NPZ table next to it. From then on the table is read with numpy alone.

H-alpha uses `pec12#h_balmer#h0.dat` (n-resolved Balmer series of H<sup>0</sup>), at 6561.9 Å. Its grid covers n<sub>e</sub> = 5×10<sup>13</sup>–2×10<sup>21</sup> m<sup>-3</sup> and T<sub>e</sub> = 0.2 eV–10 keV. `evaluate_pec()` interpolates linearly in log10(PEC) vs. (log10 n<sub>e</sub>, log10 T<sub>e</sub>), as aurora does, and holds values outside the grid at its edge.

aurora's own H-alpha model including the molecular channels (`aurora.h_am_pecs`, AMJUEL rates) cannot be used at the moment: it downloads `http://www.eirene.de/amjuel.tex`, which now returns a 404 page, and then fails to parse it.

## KN1D source rates and neutral densities

KN1D runs once per IPS timestep, after CQL3D, at two axial locations. `kn1d_simulation.py` can load any of its three lite output files (`sourceComponent` / `component`):

| Component | File | What it is |
|---|---|---|
| `'total'` | `kn1dc_lite.nc` | The midplane and gas-box runs added together on the midplane grid. This is what `kn1dc_parser.py` calls the main vessel source rate before disambiguation, and it is the file CQL3D reads (`file_kn1d` in `cqlinput`). Its neutral distribution is the midplane run's. |
| `'midplane'` | `kn1dc_midplane_lite.nc` | The midplane (z = 0) run alone, with the main vessel neutral density. |
| `'gasbox'` | `kn1dc_gasbox_lite.nc` | The gas-box (z = 0.8 m) run alone, on the gas-box radial grid. |

**Lite vs. full files.** KN1D's own wall source sets the absolute neutral density of each run. KN1DC's `convert_to_lite_netcdf()` then rescales everything the neutrals drive (f<sub>H</sub>, CX and ionization; recombination is left alone) by `nH_wall_force / nH[wall]`. This makes the neutral density at the wall equal the `nH_wall_force` in `KN1DC_*_inputs.jl`, i.e. the main vessel and gas-box neutral densities in the simulation name. The lite files are therefore what CQL3D actually used. The full outputs (`kn1dc.nc`, `kn1dc_gasbox.nc`) are not rescaled; for example, their `nH` and `Sion` are ~3000× too large for a 10<sup>15</sup> m<sup>-3</sup> main vessel. Only the lite files are used here.

**Source rate.** The velocity integral of `net_ion_source_vperpvpar` (CX in − CX out + ionization − recombination) gives a source rate density in m<sup>-3</sup> s<sup>-1</sup>. It can be negative. `kn1dc_parser.py` labelled it #/s.

**Per flux tube.** `source_rate_per_flux_tube()` converts it to s<sup>-1</sup> in each CQL3D flux tube:

S<sub>tube,j</sub> [s<sup>-1</sup>] = ⟨S⟩<sub>j</sub> [m<sup>-3</sup> s<sup>-1</sup>] × dvol<sub>j</sub> [m<sup>3</sup>]

- ⟨S⟩<sub>j</sub> is the KN1D profile averaged over the midplane cross-section of tube j (`flux_tube_average`, area weighted).
- The tube boundaries are halfway between the CQL3D midplane radii (`flux_tube_edges`). This is the same binning `run_KN1DC.jl` uses, and it gives a smooth dvol / A<sub>mid</sub> of 8.5–10.5 m across the tubes.
- `dvol` is CQL3D's volume of each flux tube over the whole field line, ±2 m including the expander. It is taken from the same IPS timestep as the source, since the equilibrium changes slightly between timesteps.
- Multiplying by `dvol` treats the midplane source density as uniform along the whole tube.

**Neutral density.** `load_neutral_density_data()` integrates the neutral distribution f<sub>H</sub> over velocity, n<sub>H</sub> = Σ f<sub>H</sub> · vol_2v, in m<sup>-3</sup>. This is the neutral density the source rate was calculated with: it equals `nH_wall_force` at the wall, and it matches the full file's `nH` × the lite scale factor. `neutral_density_per_flux_tube()` averages it over each tube's midplane cross-section, the same way as the source rate.

KN1D only gives the neutrals at two axial locations: the midplane (`'midplane'`/`'total'`, at the wall ≈ the main vessel n<sub>n</sub>) and the gas box (`'gasbox'`, at z = 0.8 m, ≈ the gas box n<sub>n</sub>). So the per-tube value is the *midplane* neutral density of each tube, not an average along it. The number of neutrals in a tube would need n<sub>H</sub>(z) along the field line, which KN1D does not provide.

**Known issue in `kn1dc_lite.nc`.** In STEP 3 of `run_KN1DC.jl`, the gas-box run is added to the midplane run with

```julia
Ionize_vperpvpar_0 .+= Ionize_vperpvpar_1[:,:,ridx_gp]
```

This adds a 2D slice to the full 3D array, so every loop iteration adds a gas-box slice at *every* radius instead of only at radius `i`. As a result, `kn1dc_lite.nc` − `kn1dc_midplane_lite.nc` is the same at every radius (checked to ~1e-8 relative, at several timesteps in two simulations). The `'total'` profile is therefore a radially uniform offset of ~4×10<sup>22</sup> m<sup>-3</sup> s<sup>-1</sup> plus the midplane run. The same applies to the `Recomb`, `CX_i_to_H` and `CX_H_to_i` lines. The intended line is presumably `Ionize_vperpvpar_0[:,:,i] .+= Ionize_vperpvpar_1[:,:,ridx_gp]`.

## Paths and caches

All paths live in `config.py`. Modules read them at call time as `config.<name>`, so you can override one for a whole session:

```python
import config
config.simulationScanDir = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/'
```

The raw diagnostic processing still lives in the `shinethru` repository. This pipeline reads its outputs and calls its scripts when an output is missing:

| Input | Location (`config` name) | Made by, if missing |
|---|---|---|
| Simulations | `simulationScanDir` | IPS runs |
| SEE detector geometry | `seeDetectorDictPath` | (fixed lookup table) |
| Processed SEE shot `{shotnum}.pkl` | `seeDataDir` | `seeProcessingScript` (`experimental_data.py`, needs MDSplus) |
| Processed AXUV shot `AXUV_results*{shotnum}*DA{n}*.npz` | `axuvDataDir` | `axuv_data_output.py` in `axuvScriptDir` |
| Thomson profile and flux loops `data.yaml` | `thomsonDataPath`, `fluxLoopDataPath` | mirror_fit |
| ADAS PEC table `{adf15 file}_{wavelength}A.npz` (and the ADF15 file in `adf15/`) | `atomicDataDir` | `atomic_data.py`, run with `auroraPython` (downloads from OPEN-ADAS) |

The slow steps are cached inside each simulation directory. They are rebuilt automatically when missing:

| Cache | Made by | Rebuilt when |
|---|---|---|
| `density_interp_data.pkl` | `generate_times_and_functions()` | Missing (it reads every `WHAM.nc`). |
| `temperature_data.pkl` | `load_temperature_data()` | Missing (it reads every `WHAM.nc`), or when `redoAnalysis=True`. |
| `throat_temperature_data.pkl` | `load_throat_temperature()` | Missing (it reads every `WHAM.nc`), or when `redoAnalysis=True`. |
| `flux_tube_data.pkl` | `load_flux_tube_data()` | Missing (it reads every `WHAM.nc`), or when `redoAnalysis=True`. |
| `kn1d_data_{component}.pkl` | `load_kn1d_data()` (source rate and neutral density) | Missing (it reads every KN1D lite file), or when `redoAnalysis=True`. |
| `synthetic_detector_data.npz` | `time_dependent_see_detector()` | Missing. |
| `density_interp_data_{shotnum}_extension.pkl` | `build_density_extension()` | Missing, built with different settings (`diodeArrayNum`, `rateKHz`, `tanhSteepness`, time window) or on a different simulation grid, or when `redoAnalysis=True`. |
| `synthetic_detector_data_{shotnum}_extension.npz` | `time_dependent_see_detector_extended()` | Missing, does not cover the requested AXUV time window, or when `redoAnalysis=True`. |

These are the same files the old `shinethru/compare_to_simulation` code wrote, so the existing caches are reused.

Plots are saved to `config.plotDir`, which defaults to `integrated_data_analysis/plots/`.

## Adding a diagnostic

1. Add its data location to `config.py`.
2. Make a `<diagnostic>.py` module. Give it a `load_experimental_data`-style loader and, if the diagnostic can be forward modelled, a synthetic version that takes the density from `cql3d_simulation.load_density_data()` (or the extension from `axuv_extension.build_density_extension()`).
3. Cache anything expensive in the simulation directory, named after the diagnostic and shot, as the SEE caches are.
4. Add a `<diagnostic>_comparison.py` script, following `axuv_comparison.py`. Its plotting function should take `axes=` and `showPlot=` so it can share a figure with the other diagnostics. Import `plotting` for the shared style and `save_plot`. Its `__main__` should take the standard inputs (simulation name, shot number, simulated and experimental times).

## Environment

Use `~/anaconda3/bin/python`, which has numpy, scipy, matplotlib, xarray and pyyaml. MDSplus is only needed when a new SEE shot has to be processed. aurora is only needed, in its own environment (`config.auroraPython`), when a new PEC table has to be made; `atomic_data.py` calls it for you. `plotting.py` switches matplotlib to the TkAgg backend for interactive windows, as the old code did. Only the comparison scripts import it; the loader modules never change the backend.

## Changes from `shinethru/compare_to_simulation`

The old code is still in place, because `taup_comparison.py` and `cql3d_confinement_times.py` import from it. Only the functions `plot_profile_comparison()` uses were brought over, plus the source rate functions of `kn1dc_parser.py` (`kn1d_simulation.py`). `b_field_interpolation()` and `source_rate_flux()` were left out, as neither returns anything. The functional changes are:

- **One `config.py`.** Previously `cql3d_vs_data.py`, `cql3d_radial_profiles.py` and `extend_cql3d_with_axuv.py` each kept their own `simulationScanDir`, and set each other's copies at runtime.
- **No circular imports.** The eqdsk reader moved into `cql3d_simulation.py`, so `axuv_extension.py` no longer imports the plotting module.
- **One density loader.** The "load `density_interp_data.pkl`, drop the restart slices, clip" block was repeated in several places. It is now `load_density_data()`.
- **Processing scripts run from any directory.** `experimental_data.py` used to be run by a relative path, so it only worked from inside `compare_to_simulation/`.
- **AXUV processing happens in `build_density_extension()`**, and only when the extension is rebuilt. Any diagnostic that uses the extension gets it, not just the SEE array.
- **Narrower exception handling.** Bare `except:` around the cache loads is now `except FileNotFoundError`, so real errors (or Ctrl-C) are no longer mistaken for a missing cache.
- **Debug animations removed.** The `makeplot` options of the loaders and the synthetic diagnostic are gone. They are still in the old modules if you need them.
- **Plots go to `config.plotDir`** instead of `shinethru/plots/`.

The figure from `plot_profile_comparison()` was checked against the old code for these two shots, with and without the extension. The data in every line, the axis limits, the labels and the legends are identical. The source rates, radial grids, timestep times and disambiguation from `kn1d_simulation.py` are also identical to `kn1dc_parser.py`.
