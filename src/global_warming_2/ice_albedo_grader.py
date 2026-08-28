# Physics config
STEFAN_BOLTZMANN = 5.67e-8  # W/m^2/K^4
emissivity = 1  # dimensionless
albedo_clamp = (0.15, 0.65)

# Linear fit coefficients (from data/ice_albedo.csv)
slope_albedo = -0.00999999999999999
intercept_albedo = 2.7999999999999985


def calculate_temperature_from_energy_balance(
    solar_constant, albedo, emissivity, stefan_boltzmann
):
    """Calculate the temperature in Kelvin from the earth's energy balance equation"""
    incoming_flux = solar_constant * (1 - albedo) / 4

    return (incoming_flux / (emissivity * stefan_boltzmann)) ** (1 / 4)


def evaluate_linear_fit(temperature, slope, intercept):
    """Get y value from the temperature in Kelvin, given a linear function with slope and intercept"""

    return slope * temperature + intercept


def clamp(value, bounds):
    """Clamp a value between minimum and maximum value"""

    return max(bounds[0], min(value, bounds[1]))


def run_energy_balance_iteration(
    solar_constant,
    albedo_init,
    n_iterations,
    slope_albedo,
    intercept_albedo,
):
    """Iteratively solve for temperature and albedo given a fixed solar constant"""
    albedo = albedo_init
    temperature = 0.0

    for i in range(n_iterations):
        temperature = calculate_temperature_from_energy_balance(
            solar_constant, albedo, emissivity, STEFAN_BOLTZMANN
        )
        albedo = evaluate_linear_fit(temperature, slope_albedo, intercept_albedo)
        albedo = clamp(albedo, albedo_clamp)

    return temperature, albedo


if __name__ == "__main__":
    L, albedo, nIters = input("").split()
    L, albedo, nIters = [float(L), float(albedo), int(nIters)]

    temperature, albedo = run_energy_balance_iteration(
        L, albedo, nIters, slope_albedo, intercept_albedo
    )

    print(temperature, albedo)
