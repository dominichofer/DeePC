# DeePC

## Description
This repository provides a Python package and a C++ library, which both implement DeePC from the paper<br>
<i>Data-Enabled Predictive Control: In the Shallows of the DeePC</i><br>
[https://arxiv.org/pdf/1811.05890](https://arxiv.org/pdf/1811.05890)

## Requirements
### Python
The Python package requires at least Python 3.10 and pip.<br>
It depends on numpy, which it installs automatically through pip.
### C++
The C++ library requires a C++ compiler, cmake, and make.<br>
It depends on Eigen, Google Test, and Google Benchmark; which it installs automatically through cmake.

## Installation
### Python
To install the Python package `deepc`, run `pip install .`.<br>
It installs the package and all its dependencies.<br>
(Use `pip install -e .` to install it in [editable mode](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs).)<br>
To run the provided tests, execute `python -m unittest discover tests/deepc`.<br>
(The GitHub actions run them on every commit.)
### C++
To build the C++ library into a folder `build`, run
```bash
cmake -B build
make -C build
```
To run the provided tests, execute `make -C build test`.<br>
(The GitHub actions run them on every commit.)

## Usage
The DeePC algorithm calculates the optimal control for a given system and a target.<br>
To grasp the system it requires data from an offline procedure in the form of system input data `u_d` and system output data `y_d`.<br>
To grasp the current stat of the system it needs initialization data in the form of system input data `u_ini` and system output data `y_ini`.<br>
With this data it solves
```math
\begin{aligned}
    &\text{minimize}_{g, u, y} \quad \| y - r \|_Q^2 + \| u - u_0 \|_R^2 \\
    &\text{subject to} \quad \begin{pmatrix} U_p \\ Y_p \\ U_f \\ Y_f \end{pmatrix} g = \begin{pmatrix} u_{\text{ini}} \\ y_{\text{ini}} \\ u \\ y \end{pmatrix} \\
    &\quad u \in \mathcal{U}
\end{aligned}
```
where
```math
\| x \|_A := x^T A x
```
```math
\begin{pmatrix} U_p \\ U_f \end{pmatrix} := \mathcal{H}_{T_{\text{ini}}+N}(u_d), \begin{pmatrix} Y_p \\ Y_f \end{pmatrix} := \mathcal{H}_{T_{\text{ini}}+N}(y_d)
```
and `H` denotes the hankel matrix.<br>
The resulting `u` is the optimal control to reach the target `r`.<br>
(Note that this differs slightly from the paper.)

### Python
The Python package provides two implementations of this, the stand-alone algorithm `deePC` and wrapped in a controller.<br>
For reference applications of the controller see `example_1d.ipynb` and `example_3d.ipynb`.
### C++
(will follow soon)
