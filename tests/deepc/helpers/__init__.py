from deepc import DiscreteLTI, linear_chirp


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
    if system.input_dim == 1:
        u_d = linear_chirp(0, samples / 2, samples)
    else:
        chirps = [linear_chirp(0, samples / 2, samples, 0.1 * i) for i in range(system.input_dim)]
        u_d = list(zip(*chirps))  # Transpose the list of chirp signals
    y_d = system.apply_multiple(u_d)
    return u_d, y_d
