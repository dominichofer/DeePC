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
```

It implements a controller
```python
class DeePC:
	is_initialized() -> bool
	append(u, y)
	clear()
	control(r) -> np.ndarray
```
which stores the last `T_ini` values provided via `append(u, v)` and returns the optimal control for a given reference trajectory via `control(r)` once enough data has been appended. This can be probed with `is_initialized()`.
