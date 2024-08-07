# DeePC	

## Description
This is a Python library that implements DeePC from the paper "Data-Enabled Predictive Control: In the Shallows of the DeePC" [https://arxiv.org/pdf/1811.05890](https://arxiv.org/pdf/1811.05890).
It requires at least Python 3.10, and the libraries listed in pyproject.toml.

## Installation
Run `pip install .` to install the package and all its dependencies.

## Usage
It provides a reference implementation of the DeePC algorithm using numpy
```python
def deePC(
    u_d: np.ndarray,
    y_d: np.ndarray,
    u_ini: np.ndarray,
    y_ini: np.ndarray,
    r: np.ndarray,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    control_constrain_fkt: Callable | None = None,
    max_pgm_iterations=300,
    pgm_tolerance=1e-6,
) -> np.ndarray:
    """
    Returns the optimal control for a given system and reference trajectory.
    According to the paper Data-Enabled Predictive Control: In the Shallows of the DeePC
    https://arxiv.org/abs/1811.05890
    Args:
        u_d: Control inputs from an offline procedure.
        y_d: Trajectories from an offline procedure.
        u_ini: Control inputs to initialize the state.
        y_ini: Trajectories to initialize the state.
        r: Reference output trajectory.
        Q: Output cost matrix, defaults to identity matrix.
        R: Control cost matrix, defaults to zero matrix. (todo change to eps*identity?)
        control_constrain_fkt: Function that constrains the control.
        max_pgm_iterations: Maximum number of iterations of the projected gradient method (PGM)
                            used to solve the constrained optimization problem.
        pgm_tolerance: Tolerance for the PGM algorithm.
    """
```

It implements a controller
```python
class DeePC:
    is_initialized() -> bool
    append(u, y)
    clear()
    control(r) -> np.ndarray
```
which stores the last `T_ini` values provided via `append(u, y)` and returns the optimal control for a given reference trajectory via `control(r)` once enough data has been appended. This can be probed with `is_initialized()`.
