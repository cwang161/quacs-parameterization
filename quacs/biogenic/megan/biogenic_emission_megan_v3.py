"""Single-column MEGAN3 emission adapter.

Requires the upstream MEGAN3_in_python repository on PYTHONPATH.
Uses upstream science functions without reimplementing their equations.
IMPORTANT: current upstream implementation averages canopy states across PFTs
before evaluating nonlinear activity responses; this adapter preserves that
behavior for regression compatibility. Fractions are normalized to vegetated
fractions, exactly as upstream, not grid-area fractions.
"""
from dataclasses import dataclass
import numpy as np
from .src import MEGCAN, MEGVEA
from .src import megan_model as model

@dataclass(frozen=True)
class MeganSettings:
    n_class: int = 19
    NLayers: int = 5
    NRTYP: int = 6
    solar_constant_w_m2: float = 1361.5
    solar_to_ppfd: float = 2.1
    air_quality_index: float = 40.0
    co2_ppm: float = 403.0
    kc_min: float = 0.0
    kc_max: float = 0.82


def biogenic_emission_megan_v3(*, day, hour, latitude_deg, temperature_k,
        ppfd, lai, previous_lai, wind_m_s, relative_humidity_percent,
        pressure_pa, pft_fraction, emission_factor, light_dependent_fraction,
        t24_k, t10d_k, p24, tmax_k, tmin_k, windmax_m_s,
        kc_7d=np.nan, swc30d=np.nan, settings=MeganSettings(), switches=None):
    """Return (n_class,) flux in nmol m-2 s-1.

    `pft_fraction` is upstream-style canopy composition, normalized internally.
    `t24_k` and `p24` must be supplied from a causal rolling 24-h history.
    `t10d_k` is mean of previous ten complete days.
    """
    # SWC30D is carried as an explicit scalar input for future coupling.
    # The upstream GAMSM response uses kc_7d, not SWC30D; do not silently
    # substitute soil moisture for the crop coefficient.
    if np.isfinite(swc30d) and not (0.0 <= swc30d <= 1.0):
        raise ValueError('swc30d must be between 0 and 1 m3/m3')
    if switches is not None and switches.get('soil_moisture_response', False) and not np.isfinite(kc_7d):
        raise ValueError('Soil moisture response requires kc_7d in upstream implementation; swc30d alone is insufficient')
    if settings.n_class < 1 or settings.n_class > MEGVEA.N_ACTIVITY_CLASSES:
        raise ValueError('n_class exceeds upstream activity-class tables')
    if settings.NRTYP != MEGCAN.N_CANOPY_TYPES:
        raise ValueError('NRTYP changes require a matching expanded MEGCAN canopy parameter table')
    if settings.NLayers < 1:
        raise ValueError('NLayers must be positive')
    fractions = np.asarray(pft_fraction, dtype=float)
    ef = np.asarray(emission_factor, dtype=float)
    ldf = np.asarray(light_dependent_fraction, dtype=float)
    if fractions.shape != (settings.NRTYP,) or ef.shape != (settings.n_class,) or ldf.shape != (settings.n_class,):
        raise ValueError('Input array shape does not match NRTYP/n_class')
    if not np.all(np.isfinite(fractions)) or np.any(fractions < 0) or fractions.sum() <= 0:
        raise ValueError('Invalid canopy fractions')
    if not np.all(np.isfinite(ef)) or np.any(ef < 0) or not np.all(np.isfinite(ldf)) or np.any((ldf < 0) | (ldf > 1)):
        raise ValueError('Invalid EF or LDF')
    if not (0 <= relative_humidity_percent <= 100) or pressure_pa <= 0:
        raise ValueError('Invalid RH or pressure')
    if switches is None:
        switches = dict(bidirectional_lai_response=False, air_quality_response=False,
            high_temperature_response=False, low_temperature_response=False,
            high_wind_response=False, soil_moisture_response=False,
            co2_response=False)
    controls = model.ControlParameters(**switches, humidity_mode='RH',
        latitude_deg=float(latitude_deg), wilting_point=0.1)
    options = model.ModelOptions(number_of_layers=settings.NLayers,
        solar_constant_w_m2=settings.solar_constant_w_m2,
        solar_to_ppfd=settings.solar_to_ppfd,
        air_quality_index=settings.air_quality_index, co2_ppm=settings.co2_ppm,
        kc_min=settings.kc_min, kc_max=settings.kc_max)
    species = model.SpeciesParameters(tuple(['Isoprene'] + [f'class_{i}' for i in range(1, settings.n_class)]), ef, ldf)
    pfts = model.PFTParameters(tuple(f'canopy_{i}' for i in range(settings.NRTYP)), fractions * 100.0)
    positions = MEGCAN.gaussian_layer_positions(settings.NLayers)
    canopy_state = model._calculate_canopy_state(day=float(day), hour=float(hour),
        temperature_k=float(temperature_k), ppfd=float(ppfd), lai=float(lai),
        wind_m_s=float(wind_m_s), humidity_input=float(relative_humidity_percent),
        pressure_pa=float(pressure_pa), controls=controls, pfts=pfts,
        options=options, layer_positions=positions)
    flux = model._calculate_record_emissions(current_lai=float(lai),
        previous_lai=float(previous_lai), canopy_state=canopy_state,
        daily_mean_ppfd=float(p24), daily_mean_temperature_k=float(t24_k),
        previous_10_day_temperature_k=float(t10d_k),
        daily_max_temperature_k=float(tmax_k), daily_min_temperature_k=float(tmin_k),
        daily_max_wind_m_s=float(windmax_m_s), kc_7d=float(kc_7d),
        controls=controls, species=species, options=options,
        layer_weights=model._layer_weights(settings.NLayers))
    return np.asarray(flux, dtype=float)
