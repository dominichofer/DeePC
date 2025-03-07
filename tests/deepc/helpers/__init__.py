import numpy as np
from deepc import DiscreteLTI


def create_1D_in_1D_out_LTI() -> DiscreteLTI:
    "Create LTI system with 1D input and 1D output"
    system = DiscreteLTI(
        A=[[0.9, -0.2], [0.7, 0.1]],
        B=[[0.1], [0]],
        C=[[1, 0]],
        D=[[0.1]],
        x_ini=[1, 1],
    )
    assert system.input_dim == 1
    assert system.output_dim == 1
    assert system.is_controllable()
    assert system.is_observable()
    assert system.is_stable()
    return system


def create_2D_in_3D_out_LTI() -> DiscreteLTI:
    "Create LTI system with 3D input and 2D output"
    system = DiscreteLTI(
        A=[[0.5, 0.1, 0], [0.1, 0.5, 0.1], [0, 0.1, 0.5]],
        B=[[0.1, 0], [0.1, 0.5], [0, 0.1]],
        C=[[1, 0, 0], [0, 1, 1], [0, 0, 1]],
        D=[[0, 0], [0, 0], [0, 0]],
        x_ini=[0, 0, 0],
    )
    assert system.input_dim == 2
    assert system.output_dim == 3
    assert system.is_controllable()
    assert system.is_observable()
    assert system.is_stable()
    return system


def gather_offline_data(system: DiscreteLTI, samples: int) -> tuple:
    u_d = np.random.uniform(-1, 1, (samples, system.input_dim))
    y_d = system.apply_multiple(u_d)
    return u_d, y_d
