# Ising-benchmark

A simple repo to benchmark different methods for solving Ising models. An Ising model is supposed to be a mathematical description with the following shape

$$

\sum_i h_i \sigma_i + \sum_i \sum_{i\lt j} J_{ij}\sigma_i \sigma_j

$$

Quadratic combinatorial optimization problems can be mapped to this particular form. This means any solver finding the global minima/maxima for it would be a good candidate to solve quadratic combinatorial optimization problems.

## Techniques

This repository will be used to benchmark different techniques for a variety of casuistic of the Ising Hamiltonian.

Currently:
* Simulated Annealing
* Quantum Annealing 
* Simulated Bifurcation
* Exact solver (when available)
* DWave Hybrid BQM solver
