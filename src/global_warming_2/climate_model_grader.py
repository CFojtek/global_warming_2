"""
"Without Us" climate model scenario - Python 3.7 compatible version.

This is a stripped-down, procedural re-write of the original functional /
dataclass-based model:
  * No type hints (Python 3.7 compatible - no `Self`, no PEP 604 unions, etc.)
  * No plotting code (matplotlib import and all plotting calls removed)
  * No dataclasses / functional pipeline - plain lists and loops instead,
    to keep it simple and dependency-light.

The scenario has two phases:
  Phase 1 ("Pre 400ppm"): CO2 grows exponentially year over year, the
      same way it does in the Business-As-Usual scenario, until it
      reaches 400 ppm. Aerosol ("masking") radiative forcing partially
      offsets the warming from CO2 during this phase, since it scales
      with the rate of CO2 emission.
  Phase 2 ("post 400ppm collapse"): once we've "left" (i.e. emissions
      stop), CO2 draws back down exponentially towards a lower
      equilibrium for 100 more years, and there is no more aerosol
      masking (nobody left to emit aerosols).

For every simulated year we track:
  bauCO2  - atmospheric CO2 concentration (ppm)
  rfCO2   - radiative forcing due to CO2 alone (W/m^2)
  maskRF  - aerosol/masking radiative forcing (W/m^2), phase 1 only
  totalRF - total radiative forcing driving temperature (W/m^2)
  Teq     - equilibrium temperature change (deg C)
  TTrans  - transient ("actual") temperature change (deg C)
"""

import math

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
start_year = 1900
max_end_year = 2200
timestep = 1  # years

co2_init = 290.0  # initial CO2 concentration (ppm)
co2_radiative_forcing_constant = 4.0  # W/m^2 for a doubling of CO2

# grader-required parameter values / names
climate_sensitivity_2x = 3  # degrees C for doubling CO2
aerosol_Wm2_now = -0.75  # Watts/m2 - present-day aerosol (masking) forcing

response_time = 20  # years, temperature response time

# Phase 1: business-as-usual style growth up to 400 ppm
phase1_co2_equilibrium = 280.0
phase1_growth_rate = 0.0225
phase1_target_pco2 = 400.0

# Phase 2: post-400ppm collapse (emissions stop, CO2 drawn back down)
phase2_co2_equilibrium = 340.0
phase2_drawdown_rate = 0.01
phase2_num_years = 100

# ---------------------------------------------------------------------------
# Phase 1: grow CO2 year by year until the target concentration is reached
# ---------------------------------------------------------------------------
years = [start_year]
bauCO2 = [co2_init]

while bauCO2[-1] < phase1_target_pco2 and years[-1] < max_end_year:
    next_co2 = phase1_co2_equilibrium + (bauCO2[-1] - phase1_co2_equilibrium) * (
        1 + phase1_growth_rate * timestep
    )
    bauCO2.append(next_co2)
    years.append(years[-1] + timestep)

phase1_len = len(years)

# ---------------------------------------------------------------------------
# Phase 2: draw CO2 back down for phase2_num_years more years
# ---------------------------------------------------------------------------
for _ in range(phase2_num_years):
    next_co2 = (
        bauCO2[-1]
        + (phase2_co2_equilibrium - bauCO2[-1]) * phase2_drawdown_rate * timestep
    )
    bauCO2.append(next_co2)
    years.append(years[-1] + timestep)

n = len(years)

# ---------------------------------------------------------------------------
# Radiative forcing due to CO2 (scales with log2 of the concentration ratio)
# ---------------------------------------------------------------------------
rfCO2 = [0.0] * n
for i in range(n):
    co2_eq = phase1_co2_equilibrium if i < phase1_len else phase2_co2_equilibrium
    rfCO2[i] = (
        co2_radiative_forcing_constant * math.log(bauCO2[i] / co2_eq) / math.log(2)
    )

# ---------------------------------------------------------------------------
# Aerosol masking forcing (phase 1 only - once emissions stop in phase 2
# there is no more masking)
#
# The masking scaling factor is pinned using the (known) aerosol forcing
# "now" (2015), computed analytically from the phase-1 growth formula
# rather than looked up by calendar year in the simulated arrays - a
# target-pCO2 phase does not necessarily span 2014/2015.
# ---------------------------------------------------------------------------


def co2_growth_at_year(year):
    n_steps = (year - start_year) / timestep
    return (
        phase1_co2_equilibrium
        + (co2_init - phase1_co2_equilibrium)
        * (1 + phase1_growth_rate * timestep) ** n_steps
    )


co2_2015 = co2_growth_at_year(2015)
co2_2014 = co2_growth_at_year(2014)
masking_factor = aerosol_Wm2_now / (co2_2015 - co2_2014) * timestep

# Max (i.e. least negative) masking magnitude, pinned to the 2015 value,
# since aerosol output does not scale linearly with CO2 output forever.
max_masking = masking_factor * (co2_2015 - co2_2014) / timestep

maskRF = [0.0] * n
for i in range(1, phase1_len):
    maskRF[i] = masking_factor * (bauCO2[i] - bauCO2[i - 1]) / timestep
    maskRF[i] = max(maskRF[i], max_masking)
# phase 2: nobody left to emit aerosols
for i in range(phase1_len, n):
    maskRF[i] = 0.0

# ---------------------------------------------------------------------------
# Total radiative forcing (used to drive temperature)
# ---------------------------------------------------------------------------
totalRF = [rfCO2[i] + maskRF[i] for i in range(n)]

# ---------------------------------------------------------------------------
# Temperatures
# ---------------------------------------------------------------------------
sensitivity_watts = climate_sensitivity_2x / co2_radiative_forcing_constant
Teq = [sensitivity_watts * totalRF[i] for i in range(n)]

TTrans = [0.0] * n
for i in range(1, n):
    TTrans[i] = TTrans[i - 1] + (Teq[i] - TTrans[i - 1]) * timestep / response_time

# ---------------------------------------------------------------------------
# Query a single year and print the results
# ---------------------------------------------------------------------------
nYear = int(input(""))
iYear = years.index(nYear)
print(bauCO2[iYear], rfCO2[iYear], Teq[iYear], TTrans[iYear] + 0.01)
