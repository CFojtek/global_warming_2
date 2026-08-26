# %%

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

STEFAN_BOLTZMANN: float = 5.67e-8  # W/m^2/K^4
WATER_DENSITY: float = 1000  # kg/m^3
WATER_SPECIFIC_HEAT: float = 4184  # J/kg/
SECONDS_PER_YEAR: int = 356 * 24 * 60 * 60

timestep: int = 10  # years
n_iterations: int = 1000  # number of iterations

water_depth: float = 4000  # meters
solar_constant: float = 1350  # W/m^2
albedo: float = 0.3  # dimensionless
epsilon: float = 1  # dimensionless

initial_temperature: float = 0  # K
initial_heat_content: float = 0  # J


Vector = NDArray[np.float64]


def calculate_heat_capacity_water(water_depth: int) -> float:
    """Calculate heat capacity of body of water"""

    return WATER_DENSITY * WATER_SPECIFIC_HEAT * water_depth


def calculate_energy_influx(solar_constant: float, albedo: float) -> float:
    """Calculate energy influx from the sun"""

    return (solar_constant * (1 - albedo)) / 4


def calculate_energy_outflux(temperature: float, epsilon: float) -> float:
    """Calculate energy outflux from the earth"""

    return epsilon * STEFAN_BOLTZMANN * temperature**4


def calculate_temperature(heat_content: float, heat_capacity: float) -> float:
    """Calculate temperature of the earth"""

    return heat_content / heat_capacity


if __name__ == "__main__":
    time_array: Vector = np.arange(
        0, n_iterations * timestep, step=timestep, dtype=np.float64
    )
    heat_content_array: Vector = np.zeros_like(time_array)
    temperature_array: Vector = np.zeros_like(time_array)

    timestep_seconds: float = timestep * SECONDS_PER_YEAR
    heat_capacity_water: float = calculate_heat_capacity_water(water_depth)
    heat_in: float = calculate_energy_influx(solar_constant, albedo)

    # Set initial conditions explicitly
    heat_content_array[0] = initial_heat_content
    temperature_array[0] = initial_temperature

    # Start looping from index 1
    for i in range(1, len(time_array)):
        heat_out: float = calculate_energy_outflux(temperature_array[i - 1], epsilon)

        heat_content_array[i] = (
            heat_content_array[i - 1] + (heat_in - heat_out) * timestep_seconds
        )
        temperature_array[i] = calculate_temperature(
            heat_content_array[i], heat_capacity_water
        )

    plt.plot(time_array, temperature_array)
    plt.xlabel("Time (years)")
    plt.ylabel("Temperature (K)")
    plt.title("Bare rock earth model with time dependant energy balance")
    plt.grid()
    plt.tight_layout()
    plt.show()

# %%
