# %%
%matplotlib widget

# %%
from dataclasses import dataclass, field
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

METERS_PER_DEGREE_LAT: float = 110.0e3  # m/deg


class Boundaries(Enum):
    """Define boundary conditions for the model
    - Wall = outward velocities are 0
    - Wrap = east-west boundary is wrapped (east boundary flow is same as west boundary)
    """

    WALL = "Wall"
    WRAP = "Wrap"


class Rotation(Enum):
    """Add earth's rotation to the model
    - None = no rotation
    - Uniform = uniform rotation across all grid cells
    - PlusMinus = linear approximation of coriolis effect changing with latitude
    - WithLatitude = exact coriolis forces changing with latitude
    """

    NONE = "None"
    UNIFORM = "Uniform"
    PLUSMINUS = "PlusMinus"
    WITHLATITUDE = "WithLatitude"


class Wind(Enum):
    """Wind setting for the model
    - None = no wind
    - Uniform = uniform wind in one direction
    - Curled = south and north rows have wind in opposite direction, middle rows no wind
    """

    NONE = "None"
    UNIFORM = "Uniform"
    CURLED = "Curled"


class Perturbation(Enum):
    """Initial perturbation for the model
    - None = no perturbation
    - Tower = middle cell gets increased water elevation
    - NSGradient = step function where south grid cells receive heightened elevation
    - EWGradient = same as NSGradient but for east-west
    """

    NONE = "None"
    TOWER = "Tower"
    NSGRADIENT = "NSGradient"
    EWGRADIENT = "EWGradient"


@dataclass(frozen=True)
class Config:
    """All config needed to run the model"""

    # solver
    timestep: float = 600.0  # seconds
    n_iterations: int = 40000
    longitude_boundary: Boundaries = field(default=Boundaries.WALL)
    latitude_boundary: Boundaries = field(default=Boundaries.WALL)
    initial_perturbation: Perturbation = field(default=Perturbation.NSGRADIENT)
    wind: Wind = field(default=Wind.CURLED)
    rotation: Rotation = field(default=Rotation.PLUSMINUS)
    interpolate_rotation: bool = True

    # geometry
    ncol: int = 8  # grid size (number of cells)
    nrow: int = 8
    dx: float = 10.0e3  # m
    dy: float = 10.0e3  # m
    mean_lat: float = 30  # deg
    mean_lon: float = 0  # deg

    # physics (always active)
    gravity: float = 9.8e-4  # m/s^2
    drag: float = 1.0e-6  # 1/s, ~10 day decay time
    height_background: float = 4000  # m

    # forcing magnitudes (only active if selected above)
    wind_magnitude: float = 1.0e-8  # m/s^2
    rotation_uniform: float = 3.5e-5  # 1/s
    rotation_variation: float = 0.8  # fractional spread (0-1), used in PlusMinus
    earth_rotation: float = 7.0e-5  # 1/s, 2*Omega, used by WithLatitude

    # initial perturbation amplitudes
    tower_amplitude: float = 1.0  # m
    gradient_amplitude: float = 0.1  # m

    # Visualisation config
    draw_heatmap: bool = True
    heatmap_update_interval: float = 60 * 60 * 6  # model time in seconds


@dataclass(frozen=True)
class Derived:
    """Grid coordinates and forcing fields variables computed from config"""

    latitudes: NDArray
    longitudes: NDArray
    wind_field: NDArray
    rotation_field: NDArray


@dataclass(frozen=True)
class ModelState:
    """Stores the state of a single model run at timestep t"""

    U: NDArray  # west-east flow velocities
    V: NDArray  # south-north flow velocities
    H: NDArray  # heights
    time: float


def compute_wind_field(config: Config) -> NDArray:
    """Compute the wind forces acting on the water surface based on given config"""
    irow = np.arange(config.nrow)
    if config.wind == Wind.NONE:
        field = np.zeros(config.nrow)
    elif config.wind == Wind.UNIFORM:
        field = np.full(config.nrow, config.wind_magnitude)
    elif config.wind == Wind.CURLED:
        field = -config.wind_magnitude * np.sin((irow + 0.5) / config.nrow * 2 * np.pi)
    else:
        raise ValueError(f"Unhandled wind scheme {config.wind}")

    # Reshape to column vector since wind field changes with latitude
    return field.reshape(-1, 1)


def compute_rotation_field(config: Config, latitudes: NDArray) -> NDArray:
    """Compute the rotational forces of the earth based on given config.
    Rotational forces increase with latitude nornally.
    """
    irow = np.arange(config.nrow)
    if config.rotation == Rotation.NONE:
        field = np.zeros(config.nrow)
    elif config.rotation == Rotation.UNIFORM:
        field = np.full(config.nrow, config.rotation_uniform)
    elif config.rotation == Rotation.PLUSMINUS:
        offset = (irow - (config.nrow - 1) / 2) / config.nrow
        field = config.rotation_uniform * (1.0 + config.rotation_variation * offset)
    elif config.rotation == Rotation.WITHLATITUDE:
        field = config.earth_rotation * np.sin(np.radians(latitudes))
    else:
        raise ValueError(f"Unhandled rotation scheme {config.rotation}")

    # Reshape to column vector since rotation field changes with latitude
    return field.reshape(-1, 1)


def compute_derived(config: Config) -> Derived:
    """Pure function: config in, all derived grid/forcing data out."""
    dy_degree = config.dy / METERS_PER_DEGREE_LAT
    dx_degree = config.dx / METERS_PER_DEGREE_LAT

    latitudes = config.mean_lat + dy_degree * (
        np.arange(config.nrow) - (config.nrow - 1) / 2
    )
    longitudes = config.mean_lon + dx_degree * (
        np.arange(config.ncol) - (config.ncol - 1) / 2
    )
    wind_field = compute_wind_field(config)
    rotation_field = compute_rotation_field(config, latitudes)

    return Derived(
        latitudes=latitudes,
        longitudes=longitudes,
        wind_field=wind_field,
        rotation_field=rotation_field,
    )


def pad_field(
    field: NDArray, axis: int, pad_width: tuple[int, int], boundary: Boundaries
) -> NDArray:
    """Pad a field with ghost cells based on boundary conditions
    - Wall boundary needs no extra ghost cells
    - Wrap boundary needs one extra set of cells at the axis
    """
    new_field = field.copy()
    pad = [(0, 0)] * new_field.ndim
    pad[axis] = pad_width

    match boundary:
        case Boundaries.WALL:
            new_field = np.pad(new_field, pad_width=pad, mode="edge")
        case Boundaries.WRAP:
            new_field = np.pad(new_field, pad_width=pad, mode="wrap")

    return new_field


def pair_mean(arr: NDArray, axis: int) -> NDArray:
    """Compute average of adjacent elements along given axis.
    Resulting array will have one less element along given axis
    e.g. pair_mean(arr, 1) is equivalent to (arr[:, :-1] + arr[:, 1:]) / 2
    """
    sl0 = [slice(None)] * arr.ndim
    sl1 = [slice(None)] * arr.ndim
    sl0[axis] = slice(0, -1)
    sl1[axis] = slice(1, None)

    return (arr[tuple(sl0)] + arr[tuple(sl1)]) / 2


def initialize(config: Config) -> ModelState:
    """Initialize model with
    - Gridsize
    - Add perturbation
    - Add necessary ghost cells for differential equations to compute
    """
    nrows, ncols = config.nrow, config.ncol
    time = 0

    # H is defined at cell centers, U and V at edges -> Need extra col / row
    U = np.zeros(shape=(nrows, ncols + 1))
    V = np.zeros(shape=(nrows + 1, ncols))
    H = np.zeros(shape=(nrows, ncols))

    mid_row = nrows // 2
    mid_col = ncols // 2

    if config.initial_perturbation == Perturbation.NONE:
        pass
    elif config.initial_perturbation == Perturbation.TOWER:
        H[mid_row, mid_col] = config.tower_amplitude
    elif config.initial_perturbation == Perturbation.NSGRADIENT:
        H[:mid_row, :] = config.gradient_amplitude
    elif config.initial_perturbation == Perturbation.EWGRADIENT:
        H[:, :mid_col] = config.gradient_amplitude
    else:
        raise ValueError(f"Unhandled perturbation scheme {config.initial_perturbation}")

    # Add ghost cells after adding perturbation, so ghost cells don't influence the real physical field
    # H needs 2 ghost cells, one before and after the axis
    H = pad_field(H, 0, (1, 1), config.latitude_boundary)
    H = pad_field(H, 1, (1, 1), config.longitude_boundary)

    return ModelState(U=U, V=V, H=H, time=time)


def apply_height_boundary(H: NDArray, config: Config) -> NDArray:
    """Apply boundary conditions for ghost cells to sea surface height"""
    H = np.copy(H)

    match config.latitude_boundary:
        case Boundaries.WALL:
            H[0, :] = H[1, :]
            H[-1, :] = H[-2, :]
        case Boundaries.WRAP:
            H[0, :] = H[-2, :]
            H[-1, :] = H[1, :]

    match config.longitude_boundary:
        case Boundaries.WALL:
            H[:, 0] = H[:, 1]
            H[:, -1] = H[:, -2]
        case Boundaries.WRAP:
            H[:, 0] = H[:, -2]
            H[:, -1] = H[:, 1]

    return H


def apply_velocity_boundary(
    U: NDArray, V: NDArray, config: Config
) -> tuple[NDArray, NDArray]:
    """Apply boundary conditions to velocity fields"""
    U, V = np.copy(U), np.copy(V)

    match config.latitude_boundary:
        case Boundaries.WALL:
            V[0, :] = 0.0
            V[-1, :] = 0.0
        case Boundaries.WRAP:
            V[-1, :] = V[0, :]

    match config.longitude_boundary:
        case Boundaries.WALL:
            U[:, 0] = 0.0
            U[:, -1] = 0.0
        case Boundaries.WRAP:
            U[:, -1] = U[:, 0]

    return U, V


def apply_rotational_boundary(
    U_rot: NDArray, V_rot: NDArray, config: Config
) -> tuple[NDArray, NDArray]:
    """Apply boundary conditions to the rotational velocities by padding
    - Wall: rotational velocities set to 0
    - Wrap: interpolate between 2 cells adjacent to boundaries
    """
    # Pre-allocate arrays with the required padded shapes
    U_rot_padded = np.zeros((U_rot.shape[0] + 2, U_rot.shape[1]))
    V_rot_padded = np.zeros((V_rot.shape[0], V_rot.shape[1] + 2))
    
    # Insert the unpadded core data into the middle
    U_rot_padded[1:-1, :] = U_rot
    V_rot_padded[:, 1:-1] = V_rot
    
    # Latitude boundaries (padding top and bottom rows) 
    if config.latitude_boundary == Boundaries.WRAP:
        pad_latitude = (U_rot[0, :] + U_rot[-1, :]) / 2
        U_rot_padded[0, :] = pad_latitude
        U_rot_padded[-1, :] = pad_latitude

    # Longitude boundaries (padding left and right columns)
    if config.longitude_boundary == Boundaries.WRAP:
        pad_longitude = (V_rot[:, 0] + V_rot[:, -1]) / 2
        V_rot_padded[:, 0] = pad_longitude
        V_rot_padded[:, -1] = pad_longitude

    return U_rot_padded, V_rot_padded


def compute_interpolated_rotational_field(
    U: NDArray, V: NDArray, config: Config, derived: Derived
) -> tuple[NDArray, NDArray]:
    """Compute exact rotational field due to earths rotation by linear interpolation.
    Since U and V are not on the same grid, simply calculating the rotational field ignores the distance between
    the points on the grid. For accurate answer we need to e.g. interpolate U onto the midpoints (same grid as H).
    After, we calculate U_rot and interpolate U_rot onto the V grid.
    """
    # Since midpints are exactly in the middle, we can use average instead of linear interpolation
    U_mid = pair_mean(U, 1)
    V_mid = pair_mean(V, 0)

    U_rot_mid = derived.rotation_field * U_mid
    V_rot_mid = derived.rotation_field * V_mid

    # interpolate Urot onto V grid and vice versa. Edge values are set by padding based on boundary condition.
    # We pad because e.g. U_rot_v has shape (2, 3) after interpolating twice. We need to pad one row before and one after,
    # resulting in shape (4, 3) -> same as V which is what we want
    U_rot_v = pair_mean(U_rot_mid, 0)
    V_rot_u = pair_mean(V_rot_mid, 1)

    return apply_rotational_boundary(U_rot_v, V_rot_u, config)


def step(state: ModelState, config: Config, derived: Derived) -> ModelState:
    """Pure function: advance one timestep, return a NEW ModelState"""
    # Get interior grids (without ghost cells)
    U_interior = state.U[:, :-1]
    V_interior = state.V[:-1, :]
    H_interior = state.H[1:-1, 1:-1]

    # height gradients. Slice out ghost cells that are not necessary
    # E.g. differencing over longitutde (cols), we don't need first and last ghost row
    dHdX = np.diff(state.H[1:-1, :], axis=1) / config.dx
    dHdY = np.diff(state.H[:, 1:-1], axis=0) / config.dy

    # velocity gradients
    dUdX = np.diff(state.U, axis=1) / config.dx
    dVdY = np.diff(state.V, axis=0) / config.dy

    # Rotational terms
    if config.interpolate_rotation:
        U_rot, V_rot = compute_interpolated_rotational_field(
            state.U, state.V, config, derived
        )
        U_rot, V_rot = U_rot[:-1, :], V_rot[:, :-1]
    else:
        U_rot = derived.rotation_field * U_interior
        V_rot = derived.rotation_field * V_interior

    # Diff equations
    dUdT = (
        V_rot
        - config.gravity * dHdX[:, :-1]
        - config.drag * U_interior
        + derived.wind_field
    )
    dVdT = -U_rot - config.gravity * dHdY[:-1, :] - config.drag * V_interior
    dHdT = -(dUdX / config.dx + dVdY / config.dy) * config.height_background

    # Advance to next step via Euler method. Only update interior cells
    U_new = np.copy(state.U)
    V_new = np.copy(state.V)
    H_new = np.copy(state.H)

    U_new[:, :-1] = U_interior + dUdT * config.timestep
    V_new[:-1, :] = V_interior + dVdT * config.timestep
    H_new[1:-1, 1:-1] = H_interior + dHdT * config.timestep
    new_time = state.time + config.timestep

    # Apply boundary conditions to update ghost cells and boundary velocities
    U_new, V_new = apply_velocity_boundary(U_new, V_new, config)
    H_new = apply_height_boundary(H_new, config)

    return ModelState(U=U_new, V=V_new, H=H_new, time=new_time)


def _heatmap_arrow_components(u: NDArray, v: NDArray, config: Config):
    """Scale velocity vectors to visually prominent arrows."""
    max_speed = float(np.max(np.hypot(u, v)))
    if max_speed == 0.0:
        return np.zeros_like(u), np.zeros_like(v)

    dx_degree = config.dx / METERS_PER_DEGREE_LAT
    dy_degree = config.dy / METERS_PER_DEGREE_LAT
    target_length = 0.45 * min(dx_degree, dy_degree)
    scale = target_length / max_speed
    return u * scale, v * scale


def _create_heatmap(config: Config, derived: Derived, state: ModelState):
    """Create the Matplotlib artists used by the live heatmap."""
    import matplotlib.pyplot as plt

    if config.heatmap_update_interval <= 0:
        raise ValueError("heatmap_update_interval must be greater than 0")

    fig, ax = plt.subplots()

    h = state.H[1:-1, 1:-1]
    dx_degree = config.dx / METERS_PER_DEGREE_LAT
    dy_degree = config.dy / METERS_PER_DEGREE_LAT
    extent = [
        derived.longitudes[0] - dx_degree / 2,
        derived.longitudes[-1] + dx_degree / 2,
        derived.latitudes[0] - dy_degree / 2,
        derived.latitudes[-1] + dy_degree / 2,
    ]

    heatmap = ax.imshow(
        h,
        origin="lower",
        interpolation="nearest",
        vmin=-0.5,
        vmax=0.5,
        extent=extent,
        aspect="equal",
        cmap="RdBu_r",
    )
    fig.colorbar(heatmap, ax=ax, label="Surface height (m)")
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    title = ax.set_title(f"Model time: {state.time / 86400:.1f} days")

    quiver = _draw_velocity_arrows(ax, config, derived, state)
    return fig, ax, heatmap, quiver, title


def _draw_velocity_arrows(ax, config: Config, derived: Derived, state: ModelState):
    """Draw prominent velocity arrows at the H cell centres."""
    u = 0.5 * (state.U[:, :-1] + state.U[:, 1:])
    v = 0.5 * (state.V[:-1, :] + state.V[1:, :])
    u_plot, v_plot = _heatmap_arrow_components(u, v, config)

    x, y = np.meshgrid(derived.longitudes, derived.latitudes)
    return ax.quiver(
        x, y, u_plot, v_plot,
        color="lime",  # Changed from "white" to a highly contrasting color
        angles="xy", scale_units="xy", scale=1,
        width=0.03, headwidth=12, headlength=14, headaxislength=12,
        pivot="mid",
    )


def animate_simulation(config: Config, derived: Derived, initial_state: ModelState):
    """Handle the Matplotlib FuncAnimation loop."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax, heatmap, quiver, title = _create_heatmap(config, derived, initial_state)

    max_time = config.n_iterations * config.timestep
    n_frames = int(np.ceil(max_time / config.heatmap_update_interval))
    next_frame_time = config.heatmap_update_interval
    state = initial_state

    def update(_frame):
        nonlocal state, next_frame_time
        
        target_time = min(next_frame_time, max_time)
        
        # Advance the model to the next frame time
        while state.time < target_time and state.time < max_time:
            state = step(state, config, derived)
            
        next_frame_time += config.heatmap_update_interval

        # Update heatmap and title
        heatmap.set_data(state.H[1:-1, 1:-1])
        title.set_text(f"Model time: {state.time / 86400:.1f} days")

        # Update quiver efficiently in-place
        u = 0.5 * (state.U[:, :-1] + state.U[:, 1:])
        v = 0.5 * (state.V[:-1, :] + state.V[1:, :])
        u_plot, v_plot = _heatmap_arrow_components(u, v, config)
        quiver.set_UVC(u_plot, v_plot)
        
        return heatmap, quiver, title

    animation = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False, repeat=False)
    fig._shallow_water_animation = animation  # Prevent garbage collection
    plt.show()

    return state


def run_simulation(config: Config) -> ModelState:
    """Clean entry point for the simulation."""
    derived = compute_derived(config)
    state = initialize(config)

    if config.draw_heatmap:
        return animate_simulation(config, derived, state)

    # Clean procedural loop for headless execution
    for _ in range(config.n_iterations):
        state = step(state, config, derived)
    return state


def run_model() -> tuple[ModelState, Config]:
    config = Config()
    return run_simulation(config), config


if __name__ == "__main__":
    final_state, config = run_model()

# %%
