#!/usr/bin/env python3
"""Single-grid-cell MEGAN3 driver with meteorology defined as Python variables.

No namelist, meteorology CSV, output files, or plots.
EF/LDF and PFT fractions still use the original upstream CSV readers.
Run with the MEGAN3_in_python repository's ``src`` package importable.

All meteorological numbers below are illustrative, NOT measured site data.
Historical quantities must be supplied by the host model or prepared from
past observations before each call. They are not inferred from one snapshot.
"""
from pathlib import Path
import sys

import numpy as np
from .. import biogenic_emission_megan_v3 as adapter
from ..src import megan_model as original

# ============================================================================
# 1. FILES: only EF/LDF and PFT CSV remain
# ============================================================================
# Directory containing this driver
MEGAN_DIR = Path(__file__).resolve().parent.parent
emission_factors_file = MEGAN_DIR / "inputs" / "EF_LDF.csv"
pft_fractions_file = MEGAN_DIR / "inputs" / "PFT_Fraction.csv"
# ============================================================================
# 2. MODEL / SITE: former namelist settings
# ============================================================================
n_class = 19
NLayers = 5
NRTYP = 6
solar_constant_w_m2 = 1361.5
solar_to_ppfd = 2.1
air_quality_index = 40.0
co2_ppm = 403.0
kc_min = 0.0
kc_max = 0.82
wilting_point = 0.196
humidity_mode = 'RH'  # Current emission adapter accepts RH only

# ============================================================================
# 3. ACTIVITY SWITCHES: former namelist settings
# ============================================================================
GAMBD_YN = True
GAMAQ_YN = True
GAMHT_YN = False
GAMLT_YN = False
GAMHW_YN = False
GAMSM_YN = False
GAMCO2_YN = False

# ============================================================================
# 4. METEOROLOGY FOR ONE GRID CELL AND ONE TIMESTEP (SCALARS)
#    Illustrative values only. Update these values before each online call.
# ============================================================================
latitude_deg = 45.4
day = 180.0                            # Day of year
hour = 12.0                            # Hour convention follows upstream
T_k = 298.15                           # Air temperature, K
P_pa = 101325.0                        # Surface air pressure, Pa
wind_m_s = 2.0                         # Wind speed, m s-1
RH_percent = 60.0                      # Relative humidity, %
PPFD = 1000.0                          # Photosynthetic photon flux, umol m-2 s-1
LAI = 3.0                              # Current LAI, m2 m-2
previous_LAI = 2.8                     # Previous LAI for leaf-age response
SWC30D = 0.25                           # Soil water content at 30 cm depth, m3 m-3
kc_7d = float('nan')                   # Seven-day coefficient, if applicable

# Historical statistics supplied explicitly by the host model.
# Upstream uses current-day statistics, not causal rolling-24h means.
T24_k = 295.15                         # Original: current-day mean T, K
P24 = 600.0                            # Original: current-day mean PPFD
T10d_k = 294.15                        # Mean T of previous 10 complete days, K
Tmax_k = 303.15                        # Original: current-day maximum T, K
Tmin_k = 289.15                        # Original: current-day minimum T, K
Windmax_m_s = 4.0                      # Original: current-day maximum wind, m s-1

# ============================================================================
# 5. OPTIONAL MIXING-RATIO UPDATE AND SCREEN PRINT
# ============================================================================
integrate_mixing_ratio = False
bottom_layer_dz_m = 50.0
step_seconds = 3600.0
tracer_initial_mol_mol = 0.0


def main():
    if humidity_mode != 'RH':
        raise ValueError('Current MEGAN adapter supports RH, not specific humidity.')
    if not np.isfinite(SWC30D) or not (0.0 <= SWC30D <= 1.0):
        raise ValueError('SWC30D must be a volumetric soil water content between 0 and 1 m3/m3.')
    if GAMSM_YN and not np.isfinite(kc_7d):
        raise ValueError('Upstream MEGAN GAMSM_YN uses kc_7d, not SWC30D. Provide kc_7d or keep GAMSM_YN=False.')

    # The two non-meteorological inputs retain the original CSV format.
    species = original.load_species_parameters(emission_factors_file)
    pfts = original.load_pft_parameters(pft_fractions_file)
    if len(species.names) != n_class:
        raise ValueError(f'n_class={n_class}, but EF/LDF has {len(species.names)} classes')
    if len(pfts.fractions_percent) != NRTYP:
        raise ValueError(f'NRTYP={NRTYP}, but PFT input has {len(pfts.fractions_percent)} types')

    settings = adapter.MeganSettings(
        n_class=n_class, NLayers=NLayers, NRTYP=NRTYP,
        solar_constant_w_m2=solar_constant_w_m2,
        solar_to_ppfd=solar_to_ppfd,
        air_quality_index=air_quality_index,
        co2_ppm=co2_ppm, kc_min=kc_min, kc_max=kc_max,
    )
    switches = dict(
        bidirectional_lai_response=GAMBD_YN,
        air_quality_response=GAMAQ_YN,
        high_temperature_response=GAMHT_YN,
        low_temperature_response=GAMLT_YN,
        high_wind_response=GAMHW_YN,
        soil_moisture_response=GAMSM_YN,
        co2_response=GAMCO2_YN,
    )
    pft_fraction = pfts.fractions_percent / 100.0
    ef = species.emission_factors_nmol_m2_s
    ldf = species.light_dependent_fraction

    # Single-cell, single-timestep call: no meteorological arrays or index loop.
    Fbio = adapter.biogenic_emission_megan_v3(
        day=day, hour=hour, latitude_deg=latitude_deg,
        temperature_k=T_k, pressure_pa=P_pa,
        wind_m_s=wind_m_s, relative_humidity_percent=RH_percent,
        ppfd=PPFD, lai=LAI, previous_lai=previous_LAI,
        pft_fraction=pft_fraction, emission_factor=ef,
        light_dependent_fraction=ldf,
        t24_k=T24_k, p24=P24, t10d_k=T10d_k,
        tmax_k=Tmax_k, tmin_k=Tmin_k, windmax_m_s=Windmax_m_s,
        kc_7d=kc_7d, swc30d=SWC30D, settings=settings, switches=switches,
    )

    np.set_printoptions(precision=6, suppress=True, linewidth=160)
    print(f'MEGAN3: one grid cell, one timestep, {n_class} VOC classes')
    print('Flux unit: nmol m-2 s-1')
    print(f'day={day:g}, hour={hour:g}, SWC30D={SWC30D:g} m3/m3')
    for name, flux in zip(species.names, Fbio):
        print(f'  {name}: {flux:.6g}')

    if integrate_mixing_ratio:
        n_air = P_pa / (8.314462618 * T_k)  # mol m-3
        tendency = Fbio * 1.0e-9 / (n_air * bottom_layer_dz_m)
        x = np.full(n_class, tracer_initial_mol_mol, dtype=float)
        x += tendency * step_seconds
        print('Tracer mixing ratios after one timestep (mol/mol):', x)


if __name__ == '__main__':
    main()
