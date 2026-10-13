from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = "data/ice_albedo.csv"

# Solver config
n_iterations: int = 200
solar_constant_step: int = 10
albedo_clamp: tuple[float, float] = (0.15, 0.65)
ice_latitude_clamp: tuple[float, float] = (0, 90)
sweep_modes = ["increasing", "decreasing"]

# Physics config
STEFAN_BOLTZMANN: float = 5.67e-8  # W/m^2/K^4
emissivity: float = 1  # dimensionless
solar_constant_range: tuple[float, float] = (1200, 1600)  # W/m^2
albedo_init: float = 0.15  # dimensionless


@dataclass
class SweepResults:
    temperature_trace: pd.DataFrame  # columns: iterations, rows: solar constants
    summary_results: pd.DataFrame  # Final convergence values


def linear_fit(x, y) -> tuple[float, float]:
    """Calculates and returns slope and intercept coefficients for given data (x, y)"""
    slope, intercept = np.polyfit(x, y, 1)

    return float(slope), float(intercept)


def calculate_temperature_from_energy_balance(
    solar_constant: float, albedo: float, emissivity: float, stefan_boltzmann: float
) -> float:
    """Calculate the temperature in Kelvin from the earth's energy balance equation"""
    incoming_flux = solar_constant * (1 - albedo) / 4

    return (incoming_flux / (emissivity * stefan_boltzmann)) ** (1 / 4)


def evaluate_linear_fit(temperature: float, slope: float, intercept: float) -> float:
    """Get y value from the temperature in Kelvin, given a linear function with slope and intercept"""

    return slope * temperature + intercept


def clamp(value: float, bounds: tuple[float, float]) -> float:
    """Clamp a value between minimum and maximum value"""

    return max(bounds[0], min(value, bounds[1]))


def run_iterative_solar_constant_sweep(
    albedo_init,
    min_solar_constant,
    max_solar_constant,
    step_solar_constant,
    n_iterations,
    slope_albedo,
    intercept_albedo,
    slope_ice,
    intercept_ice,
    mode="decreasing",
):
    """Run iterative model to calculate temperature (K), albedo and ice latitude for a range of solar constant values"""
    if mode not in sweep_modes:
        raise ValueError(f"Sweep mode must be one of {sweep_modes}, got {mode}")

    # Setup values for iterative search
    solar_constants = np.arange(
        min_solar_constant,
        max_solar_constant + step_solar_constant,
        step_solar_constant,
        dtype=float,
    )
    if mode == "decreasing":
        solar_constants = solar_constants[::-1]

    albedo = albedo_init
    temperature = 0.0

    # Create empty data structures to store intermediate results
    df_temperature = pd.DataFrame(
        np.nan, index=solar_constants, columns=range(n_iterations)
    )
    albedos = []
    ice_latitudes = []

    # Sweep from max to min solar constant
    for solar_constant in solar_constants:
        temperatures = []

        for i in range(n_iterations):
            temperature = calculate_temperature_from_energy_balance(
                solar_constant, albedo, emissivity, STEFAN_BOLTZMANN
            )
            albedo = evaluate_linear_fit(temperature, slope_albedo, intercept_albedo)
            albedo = clamp(albedo, albedo_clamp)

            temperatures.append(temperature)

        # Calculate ice latitude only once with convergent temperature
        ice_latitude = evaluate_linear_fit(temperature, slope_ice, intercept_ice)
        ice_latitude = clamp(ice_latitude, ice_latitude_clamp)

        # Store final values
        albedos.append(float(albedo))
        ice_latitudes.append(float(ice_latitude))
        df_temperature.loc[solar_constant] = temperatures

        # Set albedo to last convergent value for next iteration
        albedo = albedos[-1]

    df_summary = pd.DataFrame(
        {
            "solar_constant": solar_constants,
            "temperature": df_temperature.iloc[:, -1].values,
            "albedo": albedos,
            "ice_latitude": ice_latitudes,
            "mode": mode,
        }
    )

    return SweepResults(df_temperature, df_summary)


def plot_iteration_temperatures(df_temperature_down, df_temperature_up):
    """Plot temperature at every iteration for different values of solar constant. A plot for down and up sweep is generated"""
    df_temperature_down.T.plot(legend=False)  # Need to transpose df for correct plot
    plt.title("Temperature as function of number of iterations, down sweep")
    plt.xlabel("n iterations")
    plt.ylabel("T [K]")
    plt.savefig("plots/ice_albedo/temperature_iterations_down.png", dpi=600)

    df_temperature_up.T.plot(legend=False)
    plt.title("Temperature as function of number of iterations, up sweep")
    plt.xlabel("n iterations")
    plt.ylabel("T [K]")
    plt.tight_layout()
    plt.savefig("plots/ice_albedo/temperature_iterations_up.png", dpi=600)


def plot_convergence_values(df_summary, quantity, ylabel):
    """Plot the convergent values of defined quantity"""
    df_pivot = df_summary.pivot(index="solar_constant", columns="mode", values=quantity)

    df_pivot.plot()
    plt.title(f"Convergent {quantity} as function of solar constant")
    plt.xlabel("L [W/m^2]")
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/ice_albedo/convergent_{quantity}.png", dpi=600)


if __name__ == "__main__":
    # Read in data and fit linear functions
    df = pd.read_csv(DATA_PATH)

    slope_ice, intercept_ice = linear_fit(
        df["Mean Planetary Temperature"], df["Ice Latitude"]
    )
    slope_albedo, intercept_albedo = linear_fit(
        df["Mean Planetary Temperature"], df["Planetary Albedo"]
    )

    common_kwargs = {
        "albedo_init": albedo_init,
        "min_solar_constant": min(solar_constant_range),
        "max_solar_constant": max(solar_constant_range),
        "step_solar_constant": solar_constant_step,
        "n_iterations": n_iterations,
        "slope_albedo": slope_albedo,
        "intercept_albedo": intercept_albedo,
        "slope_ice": slope_ice,
        "intercept_ice": intercept_ice,
    }

    sweep_down = run_iterative_solar_constant_sweep(**common_kwargs, mode="decreasing")

    # Set init albedo to last calculated value in the previous run
    common_kwargs["albedo_init"] = sweep_down.summary_results["albedo"].iloc[-1]
    sweep_up = run_iterative_solar_constant_sweep(**common_kwargs, mode="increasing")
    df_summary = pd.concat([sweep_down.summary_results, sweep_up.summary_results])

    plot_iteration_temperatures(
        sweep_down.temperature_trace, sweep_up.temperature_trace
    )
    plot_convergence_values(df_summary, "temperature", "T [K]")
    plot_convergence_values(df_summary, "albedo", "alpha")
    plot_convergence_values(df_summary, "ice_latitude", "lat [deg]")
