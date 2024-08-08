# DeePC	

## Description
This is a Python library that implements DeePC from the paper "Data-Enabled Predictive Control: In the Shallows of the DeePC" [https://arxiv.org/pdf/1811.05890](https://arxiv.org/pdf/1811.05890).
It requires at least Python 3.10, and the libraries listed in pyproject.toml.

## Installation
Run `pip install .` to install the package and all its dependencies.

## Usage
It provides an implementation of the DeePC algorithm
```python
def deePC(
    _u_d: list[int | float],
    _y_d: list[int | float],
    _u_ini: list[int | float],
    _y_ini: list[int | float],
    _r: list[int | float],
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    control_constrain_fkt: Callable | None = None,
    max_pgm_iterations=300,
    pgm_tolerance=1e-6,
) -> list[int | float]:
    """
    Returns the optimal control for a given system and reference trajectory.
    According to the paper Data-Enabled Predictive Control: In the Shallows of the DeePC
    https://arxiv.org/abs/1811.05890
    Args:
        _u_d: Control inputs from an offline procedure.
        _y_d: Outputs from an offline procedure.
        _u_ini: Control inputs to initiate the state.
        _y_ini: Trajectories to initiate the state.
        _r: Reference trajectory.
        Q: Output cost matrix, defaults to identity matrix.
        R: Control cost matrix, defaults to zero matrix.
        control_constrain_fkt: Function that constrains the control.
        max_pgm_iterations: Maximum number of iterations of the projected gradient method (PGM)
                            used to solve the constrained optimization problem.
        pgm_tolerance: Tolerance for the PGM algorithm.
    """
```

It implements a controller
```python
class Controller:
    is_initialized() -> bool
    update(u, y)
    clear()
    apply(r) -> list[int | float]
```
which stores the last `T_ini` values provided via `update(u, y)` and returns the optimal control for a given reference trajectory via `apply(r)` if enough data has been provided. This can be checked with `is_initialized()`.
For a reference application of the controller see `examples.ipynb`.