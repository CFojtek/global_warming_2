from enum import Enum

import numpy as np

METERS_PER_DEGREE_LAT = 110.0e3


class Boundaries(Enum):
    WALL = "Wall"
    WRAP = "Wrap"


class Rotation(Enum):
    NONE = "None"
    UNIFORM = "Uniform"
    PLUSMINUS = "PlusMinus"
    WITHLATITUDE = "WithLatitude"


class Wind(Enum):
    NONE = "None"
    UNIFORM = "Uniform"
    CURLED = "Curled"


class Perturbation(Enum):
    TOWER = "Tower"
    NSGRADIENT = "NSGradient"
    EWGRADIENT = "EWGradient"


class Config:
    def __init__(self):
        self.timestep = 600.0
        self.n_iterations = 1000000
        self.longitude_boundary = Boundaries.WALL
        self.latitude_boundary = Boundaries.WALL
        self.initial_perturbation = Perturbation.TOWER
        self.wind = Wind.NONE
        self.rotation = Rotation.PLUSMINUS
        self.interpolate_rotation = True

        self.ncol = 3
        self.nrow = 3
        self.dx = 10.0e3
        self.dy = 10.0e3
        self.mean_lat = 30
        self.mean_lon = 0

        self.gravity = 9.8e-4
        self.drag = 1.0e-6
        self.height_background = 4000

        self.wind_magnitude = 1.0e-8
        self.rotation_uniform = 3.5e-5
        self.rotation_variation = 0.8
        self.earth_rotation = 7.0e-5

        self.tower_amplitude = 1.0
        self.gradient_amplitude = 0.1


class Derived:
    def __init__(self, latitudes, longitudes, wind_field, rotation_field):
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.wind_field = wind_field
        self.rotation_field = rotation_field


class ModelState:
    def __init__(self, U, V, H, time, dHdT=None, U_rot=None):
        self.U = U
        self.V = V
        self.H = H
        self.time = time
        self.dHdT = dHdT
        self.U_rot = U_rot


def compute_wind_field(config):
    irow = np.arange(config.nrow)
    if config.wind == Wind.NONE:
        field = np.zeros(config.nrow)
    elif config.wind == Wind.UNIFORM:
        field = np.full(config.nrow, config.wind_magnitude)
    elif config.wind == Wind.CURLED:
        field = config.wind_magnitude * np.sin((irow + 0.5) / config.nrow * 2 * np.pi)
    else:
        raise ValueError("Unhandled wind scheme")
    return field.reshape(-1, 1)


def compute_rotation_field(config, latitudes):
    irow = np.arange(config.nrow)
    if config.rotation == Rotation.NONE:
        field = np.zeros(config.nrow)
    elif config.rotation == Rotation.UNIFORM:
        field = np.full(config.nrow, -config.rotation_uniform)
    elif config.rotation == Rotation.PLUSMINUS:
        offset = (irow - (config.nrow - 1) / 2) / config.nrow
        field = -config.rotation_uniform * (1.0 - config.rotation_variation * offset)
    elif config.rotation == Rotation.WITHLATITUDE:
        field = -config.earth_rotation * np.sin(np.radians(latitudes))
    else:
        raise ValueError("Unhandled rotation scheme")
    return field.reshape(-1, 1)


def compute_derived(config):
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

    return Derived(latitudes, longitudes, wind_field, rotation_field)


def pad_field(field, axis, pad_width, boundary):
    new_field = field.copy()
    pad = [(0, 0)] * new_field.ndim
    pad[axis] = pad_width

    if boundary == Boundaries.WALL:
        new_field = np.pad(new_field, pad_width=pad, mode="edge")
    elif boundary == Boundaries.WRAP:
        new_field = np.pad(new_field, pad_width=pad, mode="wrap")

    return new_field


def pair_mean(arr, axis):
    sl0 = [slice(None)] * arr.ndim
    sl1 = [slice(None)] * arr.ndim
    sl0[axis] = slice(0, -1)
    sl1[axis] = slice(1, None)
    return (arr[tuple(sl0)] + arr[tuple(sl1)]) / 2


def initialize(config):
    nrows, ncols = config.nrow, config.ncol
    time = 0

    U = np.zeros(shape=(nrows, ncols + 1))
    V = np.zeros(shape=(nrows + 1, ncols))
    H = np.zeros(shape=(nrows, ncols))

    mid_row = nrows // 2
    mid_col = ncols // 2

    if config.initial_perturbation == Perturbation.TOWER:
        H[mid_row, mid_col] = config.tower_amplitude
    elif config.initial_perturbation == Perturbation.NSGRADIENT:
        H[:mid_row, :] = config.gradient_amplitude
    elif config.initial_perturbation == Perturbation.EWGRADIENT:
        H[:, :mid_col] = config.gradient_amplitude

    H = pad_field(H, 0, (1, 1), config.latitude_boundary)
    H = pad_field(H, 1, (1, 1), config.longitude_boundary)

    dHdT = np.zeros(shape=(nrows, ncols))
    U_rot = np.zeros(shape=(nrows, ncols + 1))

    return ModelState(U=U, V=V, H=H, time=time, dHdT=dHdT, U_rot=U_rot)


def apply_height_boundary(H, config):
    H = np.copy(H)
    if config.latitude_boundary == Boundaries.WALL:
        H[0, :] = H[1, :]
        H[-1, :] = H[-2, :]
    elif config.latitude_boundary == Boundaries.WRAP:
        H[0, :] = H[-2, :]
        H[-1, :] = H[1, :]

    if config.longitude_boundary == Boundaries.WALL:
        H[:, 0] = H[:, 1]
        H[:, -1] = H[:, -2]
    elif config.longitude_boundary == Boundaries.WRAP:
        H[:, 0] = H[:, -2]
        H[:, -1] = H[:, 1]

    return H


def apply_velocity_boundary(U, V, config):
    U, V = np.copy(U), np.copy(V)

    if config.latitude_boundary == Boundaries.WALL:
        V[0, :] = 0.0
        V[-1, :] = 0.0
    elif config.latitude_boundary == Boundaries.WRAP:
        V[-1, :] = V[0, :]

    if config.longitude_boundary == Boundaries.WALL:
        U[:, 0] = 0.0
        U[:, -1] = 0.0
    elif config.longitude_boundary == Boundaries.WRAP:
        U[:, -1] = U[:, 0]

    return U, V


def apply_rotational_boundary(U_rot, V_rot, config):
    if config.latitude_boundary == Boundaries.WALL:
        pad_latitude = 0
    elif config.latitude_boundary == Boundaries.WRAP:
        pad_latitude = (U_rot[0] + U_rot[-1]) / 2

    if config.longitude_boundary == Boundaries.WALL:
        pad_longitude = 0
    elif config.longitude_boundary == Boundaries.WRAP:
        pad_longitude = (V_rot[0] + V_rot[-1]) / 2

    U_rot_padded = np.pad(
        U_rot, ((1, 1), (0, 0)), mode="constant", constant_values=pad_latitude
    )
    V_rot_padded = np.pad(
        V_rot, ((0, 0), (1, 1)), mode="constant", constant_values=pad_longitude
    )

    return U_rot_padded, V_rot_padded


def compute_interpolated_rotational_field(U, V, config, derived):
    U_mid = pair_mean(U, 1)
    V_mid = pair_mean(V, 0)

    U_rot_mid = derived.rotation_field * U_mid
    V_rot_mid = derived.rotation_field * V_mid

    U_rot_v = pair_mean(U_rot_mid, 0)
    V_rot_u = pair_mean(V_rot_mid, 1)

    return apply_rotational_boundary(U_rot_v, V_rot_u, config)


def step(state, config, derived):
    U_interior = state.U[:, :-1]
    V_interior = state.V[:-1, :]
    H_interior = state.H[1:-1, 1:-1]

    dHdX = np.diff(state.H[1:-1, :], axis=1) / config.dx
    dHdY = np.diff(state.H[:, 1:-1], axis=0) / config.dy

    dUdX = np.diff(state.U, axis=1) / config.dx
    dVdY = np.diff(state.V, axis=0) / config.dy

    if config.interpolate_rotation:
        U_rot, V_rot = compute_interpolated_rotational_field(
            state.U, state.V, config, derived
        )
        U_rot, V_rot = U_rot[:-1, :], V_rot[:, :-1]
    else:
        U_rot = derived.rotation_field * U_interior
        V_rot = derived.rotation_field * V_interior

    dUdT = (
        V_rot
        - config.gravity * dHdX[:, :-1]
        - config.drag * U_interior
        + derived.wind_field
    )
    dVdT = -U_rot - config.gravity * dHdY[:-1, :] - config.drag * V_interior
    dHdT = -(dUdX / config.dx + dVdY / config.dy) * config.height_background

    U_new = np.copy(state.U)
    V_new = np.copy(state.V)
    H_new = np.copy(state.H)

    U_new[:, :-1] = U_interior + dUdT * config.timestep
    V_new[:-1, :] = V_interior + dVdT * config.timestep
    H_new[1:-1, 1:-1] = H_interior + dHdT * config.timestep
    new_time = state.time + config.timestep

    U_new, V_new = apply_velocity_boundary(U_new, V_new, config)
    H_new = apply_height_boundary(H_new, config)

    return ModelState(U=U_new, V=V_new, H=H_new, time=new_time, dHdT=dHdT, U_rot=U_rot)


if __name__ == "__main__":
    nSlices, iRowOut, iColOut = input("").split()
    nSlices, iRowOut, iColOut = [int(nSlices), int(iRowOut), int(iColOut)]

    horizontalWrap = True
    interpolateRotation = False

    config = Config()
    config.ncol = 3
    config.nrow = 3
    config.timestep = 600
    config.gravity = 9.8e-4
    config.wind = Wind.NONE
    config.rotation = Rotation.PLUSMINUS

    if horizontalWrap:
        config.longitude_boundary = Boundaries.WRAP
    config.interpolate_rotation = interpolateRotation

    derived = compute_derived(config)
    state = initialize(config)

    for _ in range(nSlices):
        state = step(state, config, derived)

    H = state.H[1:-1, 1:-1]
    dHdT = state.dHdT
    U = state.U[:, :-1]
    V = state.V[:-1, :]
    rotU = state.U_rot

    print(
        H[iRowOut, iColOut],
        dHdT[iRowOut, iColOut],
        U[iRowOut, iColOut],
        V[iRowOut, iColOut],
        rotU[iRowOut, iColOut],
    )
