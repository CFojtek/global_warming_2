# %%
import numpy as np


class SolverConfig:
    def __init__(self):
        self.domain_width = 1e6  # meters
        self.n_grid_points = 10  # number of grid points in mesh
        self.timestep = 100  # years
        self.n_years = 50000
        self.elevation_init = 0.0  # meters


class PhysicsConfig:
    def __init__(self):
        self.flow_speed = 1e4  # horizontal meters / year
        self.snowfall_rate = 0.5  # meters / year


def update_flow(elevation, flow_speed, grid_size):
    # Update the ice sheet flow based on the elevation gradient. A correction
    # term is used to account for the aspect ratio of the cells.
    # Flow is defined at the cell edges.
    flow = (
        flow_speed
        * ((elevation[:-1] - elevation[1:]) / grid_size)
        * ((elevation[:-1] + elevation[1:]) / (2 * grid_size))
    )

    return flow


def update_elevation(elevation, flow, snowfall_rate, timestep):
    # Update the ice sheet elevation based on flow and snowfall rate.
    # Elevation is cell-centered, and has 2 ghost cells at the edges.
    elevation = elevation.copy()
    elevation[1:-1] += (snowfall_rate + flow[:-1] - flow[1:]) * timestep

    return elevation


def run_ice_sheet_simulation(solver_conf, physics_conf):
    # Run a simple ice sheet simulation using a 1D finite difference method.
    # The ice sheet is defined by its elevation, which is updated based on
    # flow and snowfall rate. Returns the final elevation array.
    grid_size = solver_conf.domain_width / solver_conf.n_grid_points

    elevation = np.full(
        solver_conf.n_grid_points + 2, solver_conf.elevation_init, dtype=float
    )  # Add 2 extra ghost cells for boundary conditions
    elevation[0] = 0
    elevation[-1] = 0
    flow = np.zeros(solver_conf.n_grid_points + 1)

    elapsed_years = 0.0
    while elapsed_years < solver_conf.n_years:
        # Use a shorter final step if n_years isn't a whole multiple of
        # timestep, so the total simulated time always equals n_years exactly.
        dt = min(solver_conf.timestep, solver_conf.n_years - elapsed_years)

        flow = update_flow(elevation, physics_conf.flow_speed, grid_size)
        elevation = update_elevation(elevation, flow, physics_conf.snowfall_rate, dt)

        elapsed_years += dt

    return elevation


if __name__ == "__main__":
    n_years = float(input(""))

    solver_conf = SolverConfig()
    solver_conf.n_years = n_years
    physics_conf = PhysicsConfig()

    elevations = run_ice_sheet_simulation(solver_conf, physics_conf)

    print(elevations[5])
