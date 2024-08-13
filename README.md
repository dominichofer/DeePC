# DeePC	

## Description
This is a Python library and a C++ library that implement DeePC from the paper<br>
<i>Data-Enabled Predictive Control: In the Shallows of the DeePC</i><br>
[https://arxiv.org/pdf/1811.05890](https://arxiv.org/pdf/1811.05890)

## Requirements
For the Python library, at least Python 3.10 and numpy are required.

## Installation
To install the Python package `deepc`, run `pip install .`. It installs the package and all its dependencies.<br>
Use `pip install -e .` to install it in editable mode.<br>
To run the provided tests, execute `python -m unittest discover tests/deepc`.<br>
(The GitHub actions test this for every commit)<br>
<br>
To build the C++ library into a folder `build`, execute
```bash
cmake -B build
make -C build
```
To run the provided tests, execute `make -C build test`.<br>
(The GitHub actions test this for every commit)

## Usage
The Python package `deepc` provides an implementation of the DeePC algorithm
```python
def deePC(
    u_d: list | np.ndarray,
    y_d: list | np.ndarray,
    u_ini: list | np.ndarray,
    y_ini: list | np.ndarray,
    r: list | np.ndarray,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    control_constrain_fkt: Callable | None = None,
    max_pgm_iterations=300,
    pgm_tolerance=1e-6,
) -> list[float]:
    """
    Returns the optimal control for a given system and reference trajectory.
    According to the paper Data-Enabled Predictive Control: In the Shallows of the DeePC
    https://arxiv.org/abs/1811.05890
    Args:
        u_d: Control inputs from an offline procedure.
        y_d: Outputs from an offline procedure.
        u_ini: Control inputs to initiate the state.
        y_ini: Outputs to initiate the state.
        r: Reference trajectory.
        Q: Output cost matrix, defaults to identity matrix.
        R: Control cost matrix, defaults to zero matrix.
        control_constrain_fkt: Function that constrains the control.
        max_pgm_iterations: Maximum number of iterations of the projected gradient method (PGM)
                            used to solve the constrained optimization problem.
        pgm_tolerance: Tolerance for the PGM algorithm.
    """
    ...
```
and a controller
```python
class Controller:
    def __init__(
        self,
        u_d: list | np.ndarray,
        y_d: list | np.ndarray,
        T_ini: int,
        r_len: int,
        u_ini: list | np.ndarray | None = None,
        y_ini: list | np.ndarray | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        control_constrain_fkt: Callable | None = None,
        max_pgm_iterations=300,
        pgm_tolerance=1e-6,
    ) -> None:
        """
        Optimal controller for a given system and reference trajectory.
        According to the paper Data-Enabled Predictive Control: In the Shallows of the DeePC
        https://arxiv.org/abs/1811.05890
        Holds the last T_ini control inputs and trajectories to initiate the state.
        Args:
            u_d: Control inputs from an offline procedure.
            y_d: Outputs from an offline procedure.
            T_ini: Number of initial control inputs and trajectories.
            r_len: Length of the reference trajectory.
            Q: Output cost matrix, defaults to identity matrix.
            R: Control cost matrix, defaults to zero matrix.
            control_constrain_fkt: Function that constrains the control.
            max_pgm_iterations: Maximum number of iterations of the projected gradient method (PGM)
                                used to solve the constrained optimization problem.
            pgm_tolerance: Tolerance for the PGM algorithm.
        """
        ...

    def is_initialized(self) -> bool:
        "Returns whether the internal state is initialized."
        ...

    def update(self, u: list | np.ndarray, y: list | np.ndarray) -> None:
        "Updates the internal state with the given control input and trajectory."
        ...

    def clear(self) -> None:
        "Clears the internal state."
        ...

    def apply(self, r: list | np.ndarray) -> list[float]:
        """
        Returns the optimal control for a given reference trajectory.
        Args:
            r: Reference trajectory.
        """
        ...
```
which stores the last `T_ini` values provided via `update(u, y)` and returns the optimal control for a given reference trajectory via `apply(r)` if enough data has been provided beforehand; `None` otherwise. This can be checked with `is_initialized()`.<br>
For reference applications of the controller see `example_1d.ipynb` and `example_3d.ipynb`.
