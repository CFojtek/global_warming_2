# Climate & Fluid Dynamics Toy Models

A collection of five small, self-contained numerical models written in Python. They cover the core ideas of climate physics, from a planet's energy balance to ice-albedo feedback, ice sheet flow, CO2-driven warming and ocean circulation. Each model is a single readable script that uses explicit time stepping or fixed-point iteration, so the numerics stay easy to follow.

The goal is to build intuition for how simple physical feedbacks produce complex behaviour such as equilibria, hysteresis and wave-like dynamics. These are teaching-scale models, not predictive climate simulations.

The project was developed as coursework for a Coursera course on global warming.

## Models at a glance

| Script | Topic | Method | Key output |
|---|---|---|---|
| `bare_rock_model.py` | Zero-dimensional energy balance of a "bare rock" Earth with an ocean heat reservoir | Forward Euler time stepping | Temperature vs. time, relaxing to radiative equilibrium |
| `ice_albedo.py` | Ice-albedo feedback and hysteresis | Fixed-point iteration over a sweep of the solar constant | Temperature, albedo and ice latitude vs. solar constant (up and down sweeps) |
| `ice_sheet_dynamics.py` | 1D ice sheet growth under snowfall and gravity-driven flow | Finite differences on a staggered grid | Interactive elevation and flow profiles over time |
| `climate_model.py` | CO2 forcing, aerosol masking and transient warming | Phased scenarios built from composable pipeline steps | CO2, radiative forcing and temperature for "Business as Usual" and "Without Us" scenarios |
| `shallow_water_flow.py` | 2D shallow water equations with wind, drag and Coriolis effects | Finite differences on a staggered (Arakawa C-style) grid, forward Euler | Animated surface height heatmap with velocity arrows |

## 1. Bare rock energy balance (`bare_rock_model.py`)

Models a planet with no atmosphere whose surface is covered by a 4 km deep ocean. Absorbed solar radiation, `S(1 - albedo)/4`, heats the water. The planet radiates back to space as a blackbody, `epsilon * sigma * T^4`. The temperature is the stored heat divided by the heat capacity of the water column.

- Starts from T = 0 K and steps forward in 100-year increments.
- With the default parameters (solar constant 1350 W/m², albedo 0.3) the temperature converges to about 254 K, the radiative equilibrium temperature.
- The script prints the final temperature and outgoing flux, then plots T against time.

## 2. Ice-albedo feedback (`ice_albedo.py`)

Couples temperature and planetary albedo. Colder planets have more ice, which raises the albedo, which makes the planet colder still. Linear fits from tabulated data relate mean planetary temperature to albedo and to the latitude of the ice edge. For each solar constant the model iterates the energy balance and the albedo response until it converges.

The solar constant is swept **down** (1600 to 1200 W/m²) and then **up** (1200 to 1600 W/m²). The two sweeps carry the albedo state over from one step to the next, so the model can settle on different stable states for the same solar constant. This is the hysteresis loop that the plots show.

**Requires a data file:** `data/ice_albedo.csv` with the columns `Mean Planetary Temperature`, `Ice Latitude` and `Planetary Albedo`.

## 3. Ice sheet dynamics (`ice_sheet_dynamics.py`)

A 1D ice sheet on a staggered grid, with elevation at cell centres and flow at cell edges. Snow accumulates uniformly. Ice flows downhill at a rate proportional to the surface slope and the local ice thickness. The domain edges are held at zero elevation, so ice that reaches them is lost. The result is a dome-shaped ice sheet that builds up over thousands of years.

An interactive slider steps through time and shows elevation and flow side by side.

## 4. Phased climate scenarios (`climate_model.py`)

A transient climate model built from small, composable functions. Each simulation phase is a `PhaseConfig` containing a list of pipeline steps, and the steps are applied in order with a `pipe` helper.

Physics included:
- Exponentially growing or decaying atmospheric CO2
- Radiative forcing that scales with the logarithm of CO2, i.e. per doubling
- An aerosol "masking" forcing, calibrated to the 2015 value and clipped to that maximum
- Equilibrium temperature from a climate sensitivity, plus a transient temperature that relaxes toward it with a 20-year response time

Two scenarios are run:
- **Business as Usual:** CO2 keeps growing for 200 years.
- **Without Us:** CO2 grows until it reaches 400 ppm, then emissions stop and CO2 decays for 100 years. In the second phase only the CO2 forcing applies, because the aerosol masking disappears.

The script plots CO2, total radiative forcing and temperature change for both scenarios.

## 5. Shallow water flow (`shallow_water_flow.py`)

A 2D shallow water model on a small ocean basin, discretised on a staggered grid with ghost cells. Surface height lives at cell centres and the two velocity components live on cell edges. The model is configurable through a frozen `Config` dataclass and several enums:

- **Boundaries:** walls or periodic wrap-around, set separately for each axis
- **Rotation:** none, uniform, a linear "plus-minus" approximation, or the exact latitude-dependent Coriolis parameter
- **Wind:** none, uniform, or curled (opposite winds on the north and south edges)
- **Initial perturbation:** none, a single "tower" of raised water, or a north-south or east-west height step

Each step is a pure function that returns a new `ModelState`. The default run uses a curled wind, a linear Coriolis variation and a north-south height step, and it shows a live heatmap of sea surface height with velocity arrows.

## Getting started

### Requirements

- [uv](https://docs.astral.sh/uv/) for dependency management
- Python 3.13 or newer (uv can install it for you)

Dependencies are declared in `pyproject.toml`: NumPy, pandas and matplotlib at runtime, plus `ipykernel`, `ipywidgets` and `ipympl` in the dev group. The last two power the interactive ice sheet slider and the shallow water animation.

```bash
git clone https://github.com/CFojtek/global_warming_2.git
cd global_warming_2
uv sync          # creates .venv and installs runtime + dev dependencies
```

### Running the models

The models live in `src/global_warming_2/`. Run all commands from the repository root, because `ice_albedo.py` reads its data from the relative path `data/ice_albedo.csv`. The five scripts are not all run the same way. Some use Jupyter-style `# %%` cells or interactive widgets.

| Script | How to run |
|---|---|
| `bare_rock_model.py` | `uv run src/global_warming_2/bare_rock_model.py`, then enter the number of iterations when prompted (for example `50`). |
| `ice_albedo.py` | `uv run src/global_warming_2/ice_albedo.py` (needs `data/ice_albedo.csv`). |
| `climate_model.py` | `uv run src/global_warming_2/climate_model.py` |
| `ice_sheet_dynamics.py` | Open it in **VS Code** (Interactive Window) with the `.venv` interpreter selected and run the cells. The slider widget only works in a notebook-style environment. |
| `shallow_water_flow.py` | Same as above. The file begins with the `%matplotlib widget` IPython magic, so running it with plain `python` or `uv run` will fail. |

If you prefer JupyterLab, run `uv run --with jupyterlab jupyter lab`.

Set `draw_heatmap=False` in the shallow water `Config` to run it headless and get only the final state.

## Configuration

Parameters are kept at the top of each script, either as module-level constants or in `Config` dataclasses, so experiments only need a small edit. Some good starting points:

- `bare_rock_model.py`: try `albedo`, `water_depth` or `timestep`. A much larger timestep shows how explicit Euler becomes unstable.
- `ice_albedo.py`: widen `solar_constant_range` or change `albedo_clamp` to move the hysteresis loop.
- `climate_model.py`: change `climate_sensitivity`, `response_time` or `co2_growth_rate`.
- `shallow_water_flow.py`: switch `rotation`, `wind` and `initial_perturbation` to see geostrophic adjustment, wave propagation and wind-driven gyres.

## Future work

**Software quality**
- Add unit tests (with `pytest` in the dev dependency group), for example against the analytic equilibrium temperature in the bare rock model, mass conservation in the ice sheet and shallow water models, and the no-ice and full-ice limits in the albedo model.
- Replace the `input()` prompt in the bare rock model with a CLI (`argparse` or `typer`).
- Set up continuous integration (GitHub Actions) running linting, type checking and tests.
- Make every script runnable as a plain Python script, with notebook-only parts split into separate notebooks.

**Numerics**
- Replace forward Euler with Runge-Kutta or semi-implicit schemes, and add explicit CFL and stability checks.
- Add convergence studies, for example error against grid spacing and timestep.
- Use a better-conserving time integration scheme for the shallow water model.

**Physics**
- Bare rock model: add an atmosphere (greenhouse effect) and a time-varying solar constant.
- Ice-albedo model: derive the albedo and ice-line relations from a latitudinal energy balance model instead of fitted data.
- Ice sheet model: use a realistic flow law (Glen's law), use a 2D domain.
- Climate model: add a carbon cycle, a two-box or diffusive ocean for heat uptake, and an ensemble of sensitivities to show uncertainty ranges. Compare against observed temperature records.

## Project structure

```
.
├── src/
│   └── global_warming_2/
│       ├── bare_rock_model.py
│       ├── ice_albedo.py
│       ├── ice_sheet_dynamics.py
│       ├── climate_model.py
│       └── shallow_water_flow.py
├── data/
│   └── ice_albedo.csv      # input data for ice_albedo.py
├── plots/
│   └── ... # Output plots
├── pyproject.toml
├── LICENSE
└── README.md
```

## License

Released under the [MIT License](LICENSE).

Author: Christian Fojtek