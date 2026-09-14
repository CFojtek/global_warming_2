# Use functional programming to pipe similar steps for business as usual and without us scenario.
# All results per run are stored in a Scenario DataClass object

# %%
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import reduce
from typing import Self

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray


@dataclass(frozen=True)
class Config:
    """Baes config needed for a climate scenario run."""

    start_year: int = 1900  # start year of simulation
    max_end_year: int = 2200  # max end year of the simulation
    timestep: int = 1  # years

    co2_init: float = 290.0  # Initial CO2 concentration
    co2_radiative_forcing_constant: float = (
        4.0  # Radiative forcing of CO2 in W/m^2 for a doubling of the atmospheric CO2
    )

    masking_radiative_forcing_constant: float = (
        -1.5
    )  # Radiative forcing due to aerosols in W/m^2

    climate_sensitivity: float = 5.85  # deg / doubling pCO2
    response_time: float = 20  # Temperature response time in years, 20 is good good balance between deep ocean 100 year uptake and land uptake of couple of years


@dataclass(frozen=True)
class Scenario:
    """Store climate quantities of a scenario run"""

    # years of model run
    time: NDArray = field(default_factory=lambda: np.array([], dtype=float))
    # The years where a different phase starts
    phase_year_cutoffs: list[float] = field(default_factory=list)

    # concentration in ppm
    co2_p: NDArray = field(default_factory=lambda: np.array([], dtype=float))

    # Radiative forcings in W/m^2
    co2_rf: NDArray = field(default_factory=lambda: np.array([], dtype=float))
    masking_rf: NDArray = field(default_factory=lambda: np.array([], dtype=float))
    total_rf: NDArray = field(default_factory=lambda: np.array([], dtype=float))

    # Masking scaling factor for aerosol radiative forcing. One per phase
    masking_factors: list[float] = field(default_factory=list)

    # Equilibrium and transient temperature changes in deg C
    T_eq: NDArray = field(default_factory=lambda: np.array([], dtype=float))
    dT: NDArray = field(default_factory=lambda: np.array([], dtype=float))


@dataclass(frozen=True)
class PhaseConfig:
    """Config for a single segment within a climate run. Can chain multiple segments after one another to create complex scenarios."""

    name: str
    target_pco2: float | None = None
    num_years: int | None = None

    co2_equilibrium: float = 280.0  # Equilibrium atmospheric CO2 in ppm
    co2_growth_rate: float = 0.0225  # Fitted growth rate of CO2 in business as usual scenario. 2.5ppm increase when pCO2 is 400ppm
    co2_drawdown: float = 0.01  # The degrowth rate of CO2 when emissions stop

    steps: list[Callable[[Scenario, Self, Config], Scenario]] = field(
        default_factory=list
    )  # pipeline steps for this phase specifically

    def __post_init__(self) -> None:
        if not self.target_pco2 and not self.num_years:
            raise ValueError("One of target_co2 or num_years should be set")


def pipe(
    scenario: Scenario,
    phase: PhaseConfig,
    config: Config,
    steps: list[Callable[[Scenario, PhaseConfig, Config], Scenario]],
) -> Scenario:
    """Given a list of functions in steps, will pass the Scenario and config to those functions in order and return a scenario."""
    return reduce(lambda s, f: f(s, phase, config), steps, scenario)


def initialize_phase(
    scenario: Scenario, phase: PhaseConfig, config: Config
) -> Scenario:
    """Initialize phase by checking if it's the first phase in a scenario, or using last values of previous phase in the scenario."""
    start_year = scenario.time[-1] if scenario.time.size else config.start_year

    if phase.num_years:
        time = np.arange(
            start_year,
            start_year + phase.num_years + 1,
            config.timestep,
        )

    # If target pCO2 is set we need to cut the time array short
    elif phase.target_pco2 is not None:
        co2_p = scenario.co2_p[-1] if scenario.co2_p.size else config.co2_init
        years = [start_year]
        while co2_p < phase.target_pco2 and years[-1] < config.max_end_year:
            co2_p = phase.co2_equilibrium + (co2_p - phase.co2_equilibrium) * (
                1 + phase.co2_growth_rate * config.timestep
            )
            years.append(years[-1] + config.timestep)
        time = np.array(years)
    else:
        time = np.arange(start_year, config.max_end_year + 1, config.timestep)

    return replace(
        scenario, time=time, phase_year_cutoffs=[*scenario.phase_year_cutoffs, time[-1]]
    )


def step_co2_growth(scenario: Scenario, phase: PhaseConfig, config: Config) -> Scenario:
    """Create exponentially rising atmospheric CO2 conditions. Additionally check if values are not"""
    co2_p = np.zeros_like(scenario.time, dtype=float)

    # Check if we need init condition from previous phase
    co2_p[0] = scenario.co2_p[-1] if scenario.co2_p.size > 0 else config.co2_init

    for i in np.arange(1, len(co2_p)):
        co2_p[i] = phase.co2_equilibrium + (co2_p[i - 1] - phase.co2_equilibrium) * (
            1 + phase.co2_growth_rate * config.timestep
        )

    return replace(scenario, co2_p=co2_p)


def step_co2_drawdown(
    scenario: Scenario, phase: PhaseConfig, config: Config
) -> Scenario:
    """Create exponentially decreasing atmospheric CO2 conditions."""
    co2_p = np.zeros_like(scenario.time, dtype=float)

    # Check if we need init condition from previous phase
    co2_p[0] = scenario.co2_p[-1] if scenario.co2_p.size > 0 else config.co2_init

    for i in np.arange(1, len(co2_p)):
        co2_p[i] = (
            co2_p[i - 1]
            + (phase.co2_equilibrium - co2_p[i - 1])
            * phase.co2_drawdown
            * config.timestep
        )

    return replace(scenario, co2_p=co2_p)


def radiative_forcing_co2(
    scenario: Scenario, phase: PhaseConfig, config: Config
) -> Scenario:
    """Compute radiative forcing due to atmospheric CO2. Radiation scales linearly with every doubling of atmoshperic CO2."""
    ratio = scenario.co2_p / phase.co2_equilibrium
    rf = config.co2_radiative_forcing_constant * (np.log(ratio) / np.log(2))

    return replace(scenario, co2_rf=rf)


def radiative_forcing_masking(
    scenario: Scenario, phase: PhaseConfig, config: Config
) -> Scenario:
    """Compute the radiatve forcing due to masking of aerosols.
    1. Estimate the scaling factor by using aerosol radiative forcing in 2015 (RF in that year is known).
       This is computed analytically from the phase's own CO2 growth formula rather than
       looked up by calendar year inside this phase's simulated time array: a phase does
       not necessarily span 2014/2015 (e.g. a target-pCO2 phase that ends before then), and
       a failed lookup previously fell back to index 0 for both years, producing a silent
       divide-by-zero.
    2. Calculate the RF for all timesteps
    3. The max masking is clipped to value at 2015 since aerosol output does not scale linearly with CO2 output (factories getting better at filtering out aerosols).
    """

    def co2_at_year(year: float) -> float:
        n = (year - config.start_year) / config.timestep
        return (
            phase.co2_equilibrium
            + (config.co2_init - phase.co2_equilibrium)
            * (1 + phase.co2_growth_rate * config.timestep) ** n
        )

    # Calculate mask scaling factor
    co2_2015 = co2_at_year(2015)
    co2_2014 = co2_at_year(2014)
    masking_factor = (
        config.masking_radiative_forcing_constant
        / (co2_2015 - co2_2014)
        * config.timestep
    )

    # Calculate RF for aerosols
    rf = np.zeros_like(scenario.time, dtype=float)
    for i in np.arange(1, len(rf)):
        rf[i] = (
            masking_factor
            * (scenario.co2_p[i] - scenario.co2_p[i - 1])
            / config.timestep
        )

    # Clip to max value. RF from aerosols is negative, need to clip minimum!
    max_masking = masking_factor * (co2_2015 - co2_2014) / config.timestep
    rf = np.clip(rf, a_min=max_masking, a_max=None)

    return replace(
        scenario,
        masking_rf=rf,
        masking_factors=[*scenario.masking_factors, masking_factor],
    )


def step_total_rf_masked(
    scenario: Scenario, phase: PhaseConfig, config: Config
) -> Scenario:
    """Compute total radiative forcing including masking"""
    # Use pipe function to chain functions one after the other
    scenario_new = pipe(
        scenario, phase, config, [radiative_forcing_co2, radiative_forcing_masking]
    )

    return replace(scenario_new, total_rf=scenario_new.co2_rf + scenario_new.masking_rf)


def step_total_rf_co2_only(
    scenario: Scenario, phase: PhaseConfig, config: Config
) -> Scenario:
    """Compute total radiative forcing of CO2 only"""
    scenario_new = radiative_forcing_co2(scenario, phase, config)
    return replace(
        scenario_new,
        total_rf=scenario_new.co2_rf,
        # Zero out masking_rf explicitly rather than leaving it stale from a
        # previous phase, where it can carry a mismatched array length.
        masking_rf=np.zeros_like(scenario_new.time),
    )


def step_temperatures(
    scenario: Scenario, phase: PhaseConfig, config: Config
) -> Scenario:
    """Calculate the temperatures
    1. Convert climate sensititivity to m^2 / W
    2. Calculate the equilibrium temperatures from the sensitivity AT EACH TIMESTEP
    3. Calculate transient temperatures
    """
    sensitivity_watts = (
        config.climate_sensitivity / config.co2_radiative_forcing_constant
    )
    T_eq = sensitivity_watts * scenario.total_rf

    dT = np.zeros_like(scenario.time, dtype=float)
    dT[0] = scenario.dT[-1] if scenario.dT.size > 0 else 0.0
    for i in np.arange(1, len(dT)):
        dT[i] = (
            dT[i - 1] + (T_eq[i] - dT[i - 1]) * config.timestep / config.response_time
        )

    return replace(scenario, T_eq=T_eq, dT=dT)


def append_scenarios(scenario: Scenario, scenario_phase: Scenario) -> Scenario:
    """Appends a phase of a scenario to an existing scenario. If multiple phases were used, the duplicate boundary between phases needs to be deleted."""
    common_values, _idx1, idx2 = np.intersect1d(
        scenario.time, scenario_phase.time, return_indices=True
    )

    phase_start = idx2[-1] + 1 if common_values.size else 0
    phase_time = scenario_phase.time[phase_start:]
    phase_co2_p = scenario_phase.co2_p[phase_start:]
    phase_co2_rf = scenario_phase.co2_rf[phase_start:]
    phase_masking_rf = scenario_phase.masking_rf[phase_start:]
    phase_total_rf = scenario_phase.total_rf[phase_start:]
    phase_T_eq = scenario_phase.T_eq[phase_start:]
    phase_dT = scenario_phase.dT[phase_start:]

    phase_year_cutoffs = scenario.phase_year_cutoffs + scenario_phase.phase_year_cutoffs
    masking_factors = scenario.masking_factors + scenario_phase.masking_factors

    return replace(
        scenario,
        time=np.concat([scenario.time, phase_time]),
        phase_year_cutoffs=phase_year_cutoffs,
        co2_p=np.concat([scenario.co2_p, phase_co2_p]),
        co2_rf=np.concat([scenario.co2_rf, phase_co2_rf]),
        masking_rf=np.concat([scenario.masking_rf, phase_masking_rf]),
        total_rf=np.concat([scenario.total_rf, phase_total_rf]),
        masking_factors=masking_factors,
        T_eq=np.concat([scenario.T_eq, phase_T_eq]),
        dT=np.concat([scenario.dT, phase_dT]),
    )


def run_phased_scenario(config: Config, phases: list[PhaseConfig]) -> Scenario:
    scenario = Scenario()

    for phase in phases:
        # Check if it's first phase or need to use initial values of previous phase
        scenario_phase = initialize_phase(scenario, phase, config)

        # Do phase calculations
        scenario_phase = pipe(scenario_phase, phase, config, phase.steps)

        # Add phase to scenario
        scenario = append_scenarios(scenario, scenario_phase)

    return scenario


def run_model() -> dict[str, Scenario]:
    config = Config()

    # Business as usual
    bau_phase = PhaseConfig(
        name="Business as Usual",
        num_years=200,
        co2_equilibrium=280,
        steps=[
            step_co2_growth,
            step_total_rf_masked,
            step_temperatures,
        ],
    )

    # Without us (two phases)
    phase_1_bau = PhaseConfig(
        name="Pre 400ppm",
        target_pco2=400,
        co2_equilibrium=280,
        steps=[
            step_co2_growth,
            step_total_rf_masked,
            step_temperatures,
        ],
    )

    phase_2_collapse = PhaseConfig(
        name="post 400ppm collapse",
        num_years=100,
        co2_equilibrium=340,
        steps=[
            step_co2_drawdown,
            step_total_rf_co2_only,
            step_temperatures,
        ],
    )

    return {
        "Business as Usual": run_phased_scenario(config, [bau_phase]),
        "Without Us": run_phased_scenario(config, [phase_1_bau, phase_2_collapse]),
    }


def plot_scenarios(scenarios: dict[str, Scenario]) -> None:
    """Plot CO2 concentrations, radiative forcings and temperatures for all passed scenarios"""
    fig, (ax_co2, ax_rf, ax_temp) = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    fig.suptitle("Climate Model Scenarios", fontsize=14, fontweight="bold")

    for name, scenario in scenarios.items():
        ax_co2.plot(scenario.time, scenario.co2_p, label=name)
        ax_rf.plot(scenario.time, scenario.total_rf, label=name)

        (line,) = ax_temp.plot(scenario.time, scenario.dT, label=name)
        ax_temp.axhline(
            scenario.T_eq[-1],  # plot last equilibrium value
            color=line.get_color(),
            linestyle=":",
            label=f"{name} (equilibrium)",
        )

    ax_co2.set_title("Atmospheric CO2 Concentration")
    ax_co2.set_ylabel("CO2 concentration (ppm)")
    ax_co2.grid(True)
    ax_co2.legend()

    ax_rf.set_title("Total Radiative Forcing")
    ax_rf.set_ylabel("Radiative forcing (W/m$^2$)")
    ax_rf.grid(True)
    ax_rf.legend()

    ax_temp.set_title("Transient Temperature Change")
    ax_temp.set_xlabel("Year")
    ax_temp.set_ylabel("Temperature change (\u00b0C)")
    ax_temp.grid(True)
    ax_temp.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    scenarios = run_model()
    plot_scenarios(scenarios)
