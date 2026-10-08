"""Single-column MEGAN3 emission adapter.

Self-contained online MEGAN3 adapter for QUACS.
Uses upstream science functions without reimplementing their equations.
IMPORTANT: current upstream implementation averages canopy states across PFTs
before evaluating nonlinear activity responses; this adapter preserves that
behavior for regression compatibility. Fractions are normalized to vegetated
fractions, exactly as upstream, not grid-area fractions.
"""
from dataclasses import dataclass
import numpy as np
from .src import MEGCAN, MEGVEA


# Online MEGAN3 canopy state, activity coupling and EF/PFT CSV support.
# Retained from the upstream model without changing scientific equations.
from pathlib import Path
from typing import Literal, NamedTuple
import pandas as pd
from .src import MEGCAN as canopy
from .src import MEGVEA as activity

_LAYER_WEIGHTS_5 = np.array([0.1184635, 0.2393144, 0.284444444, 0.2393144, 0.1184635], dtype=float)

@dataclass(frozen=True)
class ModelOptions:
    """Numerical and diagnostic options that are not stored in CSV files."""

    number_of_layers: int = 5
    solar_constant_w_m2: float = 1361.5
    solar_to_ppfd: float = 2.1
    air_quality_index: float = 40.0
    co2_ppm: float = 403.0
    kc_min: float = 0.0
    kc_max: float = 0.82
    isoprene_molecular_weight_g_mol: float = 68.12
    daytime_start_hour: float = 9.0
    daytime_end_hour: float = 17.0

    def __post_init__(self) -> None:
        """Validate option ranges immediately after dataclass construction."""

        if self.number_of_layers <= 0:
            raise ValueError("number_of_layers must be positive")
        if self.solar_constant_w_m2 <= 0.0:
            raise ValueError("solar_constant_w_m2 must be positive")
        if self.solar_to_ppfd <= 0.0:
            raise ValueError("solar_to_ppfd must be positive")
        if self.air_quality_index < 0.0:
            raise ValueError("air_quality_index cannot be negative")
        if self.kc_max <= self.kc_min:
            raise ValueError("kc_max must be greater than kc_min")
        if self.isoprene_molecular_weight_g_mol <= 0.0:
            raise ValueError("isoprene_molecular_weight_g_mol must be positive")
        if not 0.0 <= self.daytime_start_hour <= 24.0:
            raise ValueError("daytime_start_hour must be between 0 and 24")
        if not 0.0 <= self.daytime_end_hour <= 24.0:
            raise ValueError("daytime_end_hour must be between 0 and 24")
        if self.daytime_end_hour < self.daytime_start_hour:
            raise ValueError("daytime_end_hour must be >= daytime_start_hour")
        if self.co2_ppm <= 0.0:
            raise ValueError("co2_ppm must be positive")


@dataclass(frozen=True)
class ControlParameters:
    """Activity switches and site parameters for one model run."""

    bidirectional_lai_response: bool
    air_quality_response: bool
    high_temperature_response: bool
    low_temperature_response: bool
    high_wind_response: bool
    soil_moisture_response: bool
    co2_response: bool
    humidity_mode: Literal["RH", "QV"]
    latitude_deg: float
    wilting_point: float


@dataclass(frozen=True)
class SpeciesParameters:
    """Species/activity-class metadata read from ``3.EF_LDF.csv``."""

    names: tuple[str, ...]
    emission_factors_nmol_m2_s: np.ndarray
    light_dependent_fraction: np.ndarray


@dataclass(frozen=True)
class PFTParameters:
    """Canopy-type fractions read from ``4.PFT_Fraction.csv``."""

    names: tuple[str, ...]
    fractions_percent: np.ndarray


class CanopyState(NamedTuple):
    """PFT-weighted sun/shade canopy state for all vertical layers."""

    sun_leaf_temperature_k: np.ndarray
    shade_leaf_temperature_k: np.ndarray
    sun_ppfd: np.ndarray
    shade_ppfd: np.ndarray
    sun_fraction: np.ndarray


def _clean_text(value: object) -> str:
    """Strip whitespace and byte-order marks from CSV labels."""

    return str(value).replace("\ufeff", "").strip()


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV robustly when it may contain a UTF-8 BOM."""

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [_clean_text(column) for column in frame.columns]
    return frame


def _find_column(
    frame: pd.DataFrame,
    aliases: tuple[str, ...],
    *,
    required: bool = True,
) -> str | None:
    """Find the first matching column name from a list of accepted aliases."""

    normalized = {_clean_text(column).lower(): column for column in frame.columns}
    for alias in aliases:
        match = normalized.get(_clean_text(alias).lower())
        if match is not None:
            return match
    if required:
        raise ValueError(
            "Missing required column. Expected one of: " + ", ".join(aliases)
        )
    return None


def _numeric_series(
    frame: pd.DataFrame,
    aliases: tuple[str, ...],
    *,
    required: bool = True,
    fill_value: float = np.nan,
) -> pd.Series:
    """Load one numeric input column, optionally creating a missing-value series."""

    column = _find_column(frame, aliases, required=required)
    if column is None:
        return pd.Series(fill_value, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def load_species_parameters(path: Path) -> SpeciesParameters:
    """Read species names, emission factors, and light-dependent fractions."""

    frame = _read_csv(path)
    category_column = _find_column(frame, ("Categories", "Species", "Compound"))
    emission_factor_column = _find_column(
        frame,
        (
            "Emission Factor (nmol/m-2/s-1)",
            "Emission Factor",
            "EF",
        ),
    )
    ldf_column = _find_column(frame, ("LDF", "Light Dependent Fraction"))

    names = tuple(_clean_text(value) for value in frame[category_column])
    emission_factors = pd.to_numeric(
        frame[emission_factor_column], errors="coerce"
    ).to_numpy(dtype=float)
    light_dependent_fraction = pd.to_numeric(
        frame[ldf_column], errors="coerce"
    ).to_numpy(dtype=float)

    if not names:
        raise ValueError("At least one species/activity class is required")
    if len(names) > activity.N_ACTIVITY_CLASSES:
        raise ValueError(
            f"The activity-factor tables support at most "
            f"{activity.N_ACTIVITY_CLASSES} classes; received {len(names)}"
        )
    if not names[0].lower().startswith("isoprene"):
        raise ValueError(
            "The first row of the EF/LDF file must be isoprene because the "
            "current activity-class coefficient ordering assumes index 0."
        )
    if not np.all(np.isfinite(emission_factors)) or np.any(emission_factors < 0.0):
        raise ValueError("Emission factors must be finite and non-negative")
    if not np.all(np.isfinite(light_dependent_fraction)) or np.any(
        (light_dependent_fraction < 0.0) | (light_dependent_fraction > 1.0)
    ):
        raise ValueError("LDF values must be finite and between 0 and 1")

    return SpeciesParameters(names, emission_factors, light_dependent_fraction)


def load_pft_parameters(path: Path) -> PFTParameters:
    """Read and validate the six canopy-type fractions (percent)."""

    frame = _read_csv(path)
    pft_column = _find_column(frame, ("PFT", "Canopy Type", "Vegetation Type"))
    fraction_column = _find_column(frame, ("Fraction", "Fraction(%)", "Percent"))

    names = tuple(_clean_text(value) for value in frame[pft_column])
    fractions = pd.to_numeric(frame[fraction_column], errors="coerce").to_numpy(
        dtype=float
    )

    if len(fractions) != canopy.N_CANOPY_TYPES:
        raise ValueError(
            f"Exactly {canopy.N_CANOPY_TYPES} PFT fractions are required in the "
            f"MEGCAN order; received {len(fractions)}"
        )
    if not np.all(np.isfinite(fractions)) or np.any(fractions < 0.0):
        raise ValueError("PFT fractions must be finite and non-negative")
    if fractions.sum() <= 0.0:
        raise ValueError("At least one PFT fraction must be greater than zero")

    return PFTParameters(names, fractions)


def _layer_weights(number_of_layers: int) -> np.ndarray:
    """Return vertical quadrature weights used to integrate activity factors."""

    if number_of_layers == 5:
        return _LAYER_WEIGHTS_5.copy()
    return np.full(number_of_layers, 1.0 / number_of_layers, dtype=float)


def _water_vapor_pressure_pa(
    humidity_input: float,
    temperature_k: float,
    pressure_pa: float,
    controls: ControlParameters,
) -> float:
    """Convert the selected humidity input to canopy vapor pressure (Pa)."""

    if controls.humidity_mode == "RH":
        return canopy.relative_humidity_to_vapor_pressure(
            humidity_input,
            temperature_k,
        )
    return canopy.mixing_ratio_to_vapor_pressure(humidity_input, pressure_pa)


def _calculate_canopy_state(
    *,
    day: float,
    hour: float,
    temperature_k: float,
    ppfd: float,
    lai: float,
    wind_m_s: float,
    humidity_input: float,
    pressure_pa: float,
    controls: ControlParameters,
    pfts: PFTParameters,
    options: ModelOptions,
    layer_positions: np.ndarray,
) -> CanopyState:
    """Calculate the PFT-weighted canopy light and leaf-temperature state."""

    n_layers = options.number_of_layers
    if lai <= 0.0:
        # Emissions are exactly zero after multiplying by LAI.  A finite canopy
        # state keeps intermediate activity-factor calculations well behaved.
        return CanopyState(
            np.full(n_layers, temperature_k, dtype=float),
            np.full(n_layers, temperature_k, dtype=float),
            np.zeros(n_layers, dtype=float),
            np.zeros(n_layers, dtype=float),
            np.zeros(n_layers, dtype=float),
        )

    solar_elevation_deg = canopy.solar_elevation_angle(
        day,
        controls.latitude_deg,
        hour,
    )
    sin_solar_elevation = np.sin(solar_elevation_deg / 57.29578)
    solar_w_m2 = ppfd / options.solar_to_ppfd
    maximum_solar = (
        sin_solar_elevation
        * options.solar_constant_w_m2
        * canopy.solar_eccentricity_factor(day)
    )
    (
        diffuse_visible,
        beam_visible,
        diffuse_nir,
        beam_nir,
    ) = canopy.partition_solar_radiation(solar_w_m2, maximum_solar)

    vapor_pressure_pa = _water_vapor_pressure_pa(
        humidity_input,
        temperature_k,
        pressure_pa,
        controls,
    )

    pft_weights = pfts.fractions_percent / pfts.fractions_percent.sum()
    active_pfts = np.flatnonzero(pft_weights > 0.0)

    sun_temperature = np.zeros(n_layers, dtype=float)
    shade_temperature = np.zeros(n_layers, dtype=float)
    sun_ppfd = np.zeros(n_layers, dtype=float)
    shade_ppfd = np.zeros(n_layers, dtype=float)
    sun_fraction = np.zeros(n_layers, dtype=float)

    # A major performance improvement: canopy radiation/energy balance is
    # calculated only for PFTs with non-zero fractions.  The original code ran
    # all six PFTs even when five had zero weight.
    for pft_index in active_pfts:
        radiation = canopy.canopy_radiation(
            layer_positions,
            n_layers,
            lai,
            sin_solar_elevation,
            beam_visible,
            diffuse_visible,
            beam_nir,
            diffuse_nir,
            int(pft_index),
        )
        temperature_lapse_rate = canopy.canopy_temperature_lapse_rate(
            int(pft_index),
            solar_w_m2,
        )
        energy = canopy.canopy_energy_balance(
            temperature_lapse_rate,
            n_layers,
            layer_positions,
            int(pft_index),
            temperature_k,
            wind_m_s,
            radiation.sun_ppfd,
            radiation.shade_ppfd,
            radiation.sun_visible,
            radiation.shade_visible,
            radiation.sun_nir,
            radiation.shade_nir,
            vapor_pressure_pa,
        )

        weight = pft_weights[pft_index]
        sun_temperature += weight * energy.sun_leaf_temperature_k
        shade_temperature += weight * energy.shade_leaf_temperature_k
        sun_ppfd += weight * radiation.sun_ppfd
        shade_ppfd += weight * radiation.shade_ppfd
        sun_fraction += weight * radiation.sun_fraction

    return CanopyState(
        sun_temperature,
        shade_temperature,
        sun_ppfd,
        shade_ppfd,
        sun_fraction,
    )


def _stress_factors(
    species_index: int,
    *,
    daily_max_temperature_k: float,
    daily_min_temperature_k: float,
    daily_max_wind_m_s: float,
    controls: ControlParameters,
) -> tuple[float, float, float]:
    """Return high-temperature, low-temperature, and high-wind factors."""

    gamma_high_temperature = (
        activity.gamma_ht(species_index, daily_max_temperature_k)
        if controls.high_temperature_response
        else 1.0
    )
    gamma_low_temperature = (
        activity.gamma_lt(species_index, daily_min_temperature_k)
        if controls.low_temperature_response
        else 1.0
    )
    gamma_high_wind = (
        activity.gamma_hw(species_index, daily_max_wind_m_s)
        if controls.high_wind_response
        else 1.0
    )
    return gamma_high_temperature, gamma_low_temperature, gamma_high_wind


def _calculate_record_emissions(
    *,
    current_lai: float,
    previous_lai: float,
    canopy_state: CanopyState,
    daily_mean_ppfd: float,
    daily_mean_temperature_k: float,
    previous_10_day_temperature_k: float,
    daily_max_temperature_k: float,
    daily_min_temperature_k: float,
    daily_max_wind_m_s: float,
    kc_7d: float,
    controls: ControlParameters,
    species: SpeciesParameters,
    options: ModelOptions,
    layer_weights: np.ndarray,
) -> np.ndarray:
    """Calculate all species emissions for one valid model record."""

    n_species = len(species.names)
    emissions = np.full(n_species, np.nan, dtype=float)

    if current_lai <= 0.0:
        emissions.fill(0.0)
        return emissions

    gamma_leaf_age = activity.gamma_a(
        previous_lai,
        current_lai,
        daily_mean_temperature_k,
    )
    gamma_canopy_depth = activity.gamma_cd(
        options.number_of_layers,
        current_lai,
    )
    gamma_bidirectional = (
        activity.gamma_laibidir(current_lai)
        if controls.bidirectional_lai_response
        else 1.0
    )

    # CO2 response is controlled independently by GAMCO2_YN and applies only
    # to isoprene.
    gamma_co2_isoprene = (
        activity.gamma_co2(options.co2_ppm) if controls.co2_response else 1.0
    )

    light_sun = np.array(
        [
            activity.gamp(value, daily_mean_ppfd)
            for value in canopy_state.sun_ppfd
        ],
        dtype=float,
    )
    light_shade = np.array(
        [
            activity.gamp(value, daily_mean_ppfd)
            for value in canopy_state.shade_ppfd
        ],
        dtype=float,
    )

    for species_index in range(n_species):
        light_dependent_fraction = species.light_dependent_fraction[species_index]

        gamma_air_quality = (
            activity.gamma_aq(species_index, options.air_quality_index)
            if controls.air_quality_response
            else 1.0
        )
        (
            gamma_high_temperature,
            gamma_low_temperature,
            gamma_high_wind,
        ) = _stress_factors(
            species_index,
            daily_max_temperature_k=daily_max_temperature_k,
            daily_min_temperature_k=daily_min_temperature_k,
            daily_max_wind_m_s=daily_max_wind_m_s,
            controls=controls,
        )

        if species_index == 0 and controls.soil_moisture_response:
            gamma_soil_moisture = activity.gamma_sm_kc(
                kc_7d,
                options.kc_max,
                options.kc_min,
            )
        else:
            gamma_soil_moisture = 1.0

        gamma_temperature_light = 0.0
        for layer_index in range(options.number_of_layers):
            sun_fraction = canopy_state.sun_fraction[layer_index]
            shade_fraction = 1.0 - sun_fraction

            light_dependent_activity = gamma_canopy_depth[layer_index] * (
                activity.gamtld(
                    canopy_state.sun_leaf_temperature_k[layer_index],
                    daily_mean_temperature_k,
                    previous_10_day_temperature_k,
                    species_index,
                )
                * light_sun[layer_index]
                * sun_fraction
                + activity.gamtld(
                    canopy_state.shade_leaf_temperature_k[layer_index],
                    daily_mean_temperature_k,
                    previous_10_day_temperature_k,
                    species_index,
                )
                * light_shade[layer_index]
                * shade_fraction
            )
            light_independent_activity = (
                activity.gamtli(
                    canopy_state.sun_leaf_temperature_k[layer_index],
                    species_index,
                )
                * sun_fraction
                + activity.gamtli(
                    canopy_state.shade_leaf_temperature_k[layer_index],
                    species_index,
                )
                * shade_fraction
            )
            gamma_temperature_light += layer_weights[layer_index] * (
                light_dependent_activity * light_dependent_fraction
                + light_independent_activity
                * (1.0 - light_dependent_fraction)
            )

        common_activity = (
            current_lai
            * gamma_temperature_light
            * gamma_leaf_age[species_index]
            * gamma_high_wind
            * gamma_air_quality
            * gamma_high_temperature
            * gamma_low_temperature
            * gamma_soil_moisture
        )

        if species_index == 0:
            common_activity *= gamma_co2_isoprene
        elif species_index == 12:
            # The bidirectional LAI factor is applied only to the combined
            # acetaldehyde/ethanol activity class, matching the source code.
            common_activity *= gamma_bidirectional

        emissions[species_index] = (
            common_activity
            * species.emission_factors_nmol_m2_s[species_index]
        )

    return emissions


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
    controls = ControlParameters(**switches, humidity_mode='RH',
        latitude_deg=float(latitude_deg), wilting_point=0.1)
    options = ModelOptions(number_of_layers=settings.NLayers,
        solar_constant_w_m2=settings.solar_constant_w_m2,
        solar_to_ppfd=settings.solar_to_ppfd,
        air_quality_index=settings.air_quality_index, co2_ppm=settings.co2_ppm,
        kc_min=settings.kc_min, kc_max=settings.kc_max)
    species = SpeciesParameters(tuple(['Isoprene'] + [f'class_{i}' for i in range(1, settings.n_class)]), ef, ldf)
    pfts = PFTParameters(tuple(f'canopy_{i}' for i in range(settings.NRTYP)), fractions * 100.0)
    positions = MEGCAN.gaussian_layer_positions(settings.NLayers)
    canopy_state = _calculate_canopy_state(day=float(day), hour=float(hour),
        temperature_k=float(temperature_k), ppfd=float(ppfd), lai=float(lai),
        wind_m_s=float(wind_m_s), humidity_input=float(relative_humidity_percent),
        pressure_pa=float(pressure_pa), controls=controls, pfts=pfts,
        options=options, layer_positions=positions)
    flux = _calculate_record_emissions(current_lai=float(lai),
        previous_lai=float(previous_lai), canopy_state=canopy_state,
        daily_mean_ppfd=float(p24), daily_mean_temperature_k=float(t24_k),
        previous_10_day_temperature_k=float(t10d_k),
        daily_max_temperature_k=float(tmax_k), daily_min_temperature_k=float(tmin_k),
        daily_max_wind_m_s=float(windmax_m_s), kc_7d=float(kc_7d),
        controls=controls, species=species, options=options,
        layer_weights=_layer_weights(settings.NLayers))
    return np.asarray(flux, dtype=float)
