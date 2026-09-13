# %%
import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipywidgets import interact, widgets
from numpy.typing import NDArray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class SolverConfig:
    domain_width: float = 1e6  # meters
    n_grid_points: int = 10  # number of grid points in mesh
    timestep: float = 100  # years
    n_years: float = 25000
    elevation_init: float = 0.0  # meters


@dataclass
class PhysicsConfig:
    flow_speed: float = 1e4  # horizontal meters / year
    snowfall_rate: float = 1.0  # meters / year


@dataclass
class SimulationResults:
    elevations: pd.DataFrame
    flows: pd.DataFrame


def update_flow(elevation: NDArray, flow_speed: float, grid_size: float) -> NDArray:
    """Update the ice sheet flow based on the elevation gradient. A correction term is used to account for the aspect ratio of the cells.
    Flow is defined at the cell edges.
    """
    flow = (
        flow_speed
        * ((elevation[:-1] - elevation[1:]) / grid_size)
        * ((elevation[:-1] + elevation[1:]) / (2 * grid_size))
    )

    return flow


def update_elevation(
    elevation: NDArray, flow: NDArray, snowfall_rate: float, timestep: float
) -> NDArray:
    """Update the ice sheet elevation based on flow and snowfall rate. Elevation is cell-centered, and has 2 ghost cells at the edges."""
    elevation = elevation.copy()
    elevation[1:-1] += (snowfall_rate + flow[:-1] - flow[1:]) * timestep

    return elevation


def format_results(history: list[NDArray], solver_conf: SolverConfig) -> pd.DataFrame:
    """Format simulation history into pandas DataFrame. The row index is the
    integer timestep number (0, 1, 2, ...), not elapsed years, so that
    lookups by row are exact integer matches with no floating point drift.
    Multiply by solver_conf.timestep to convert a step number to elapsed
    years. Elevations are cell-centered, flows are defined at the cell edges.
    """
    n_steps = np.arange(len(history))

    grid_point_width = solver_conf.domain_width / solver_conf.n_grid_points
    n_points = history[0].shape[0]

    if n_points == solver_conf.n_grid_points + 2:
        # Elevatio is cell-centered, including ghost cells at the boundaries
        grid = (np.arange(n_points) - 0.5) * grid_point_width
    elif n_points == solver_conf.n_grid_points + 1:
        # Defined at edges between elevation cells
        grid = np.arange(n_points) * grid_point_width
    else:
        raise ValueError(f"Unexpected number of columns in history: {n_points}")

    return pd.DataFrame(history, columns=grid, index=n_steps)


def run_ice_sheet_simulation(
    solver_conf: SolverConfig, physics_conf: PhysicsConfig
) -> SimulationResults:
    """Run a simple ice sheet simulation using a 1D finite difference method. The ice sheet is defined by its elevation, which is updated based on flow and snowfall rate."""
    grid_size = solver_conf.domain_width / solver_conf.n_grid_points

    elevation = np.full(
        solver_conf.n_grid_points + 2, solver_conf.elevation_init, dtype=float
    )  # Add 2 extra ghost cells for boundary conditions
    elevation[0] = 0
    elevation[-1] = 0
    flow = np.zeros(solver_conf.n_grid_points + 1)

    elevation_history = [elevation.copy()]
    flow_history = [flow.copy()]

    for elapsed_years in np.arange(
        solver_conf.timestep,
        solver_conf.n_years + solver_conf.timestep,
        solver_conf.timestep,
    ):
        logger.info(f"Running timestep {elapsed_years} of {solver_conf.n_years} years")

        flow = update_flow(elevation, physics_conf.flow_speed, grid_size)
        elevation = update_elevation(
            elevation, flow, physics_conf.snowfall_rate, solver_conf.timestep
        )

        flow_history.append(flow)
        elevation_history.append(elevation)

    flows_df = format_results(flow_history, solver_conf)
    elevations_df = format_results(elevation_history, solver_conf)

    return SimulationResults(elevations_df, flows_df)


def plot_results(results: SimulationResults, timestep: float):
    """Plot simulation results in interactive plot with slider. The slider
    moves over integer timestep numbers (exact, no float drift); the elapsed
    year shown in the plot titles is computed from the step number.
    """

    def update_plots(step: int):
        elapsed_years = step * timestep

        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(
            results.elevations.columns,
            results.elevations.loc[step],
        )
        ax1.set_title(f"Ice sheet elevation (year {elapsed_years:.0f})")
        ax1.set_ylabel("Elevation (m)")
        ax1.grid()
        ax1.set_xlabel("Distance (m)")

        ax2.plot(
            results.flows.columns,
            results.flows.loc[step],
        )
        ax2.set_title(f"Ice sheet flow (year {elapsed_years:.0f})")
        ax2.set_ylabel("Flow (m/year)")
        ax2.grid()
        ax2.set_xlabel("Distance (m)")

        plt.tight_layout()
        plt.show()

    interact(
        update_plots,
        step=widgets.IntSlider(
            min=0, max=results.elevations.index[-1], step=1, value=0
        ),
    )


if __name__ == "__main__":
    solver_conf = SolverConfig()
    physics_conf = PhysicsConfig()

    results = run_ice_sheet_simulation(solver_conf, physics_conf)

    plot_results(results, solver_conf.timestep)
