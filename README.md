# Ising-benchmark

A simple repo to benchmark different methods for solving Ising models. An Ising model is supposed to be a mathematical description with the following shape

$$
\sum_i h_i \sigma_i + \sum_i \sum_{i\lt j} J_{ij}\sigma_i \sigma_j
$$

Quadratic combinatorial optimization problems can be mapped to this particular form. This means any solver finding the global minima/maxima for it would be a good candidate to solve quadratic combinatorial optimization problems.

## Usage

The code allows for being called from the terminal in order to generate the results. Through click on can access the instruction on how to call the _random_benchmark.py_ functionality.

```sh
python src/ising_benchmark/random_benchmark.py --help
```

Something similar to this should pop up:
```
Usage: random_benchmark.py [OPTIONS]

  Randomized Ising benckmark

Options:
  --max_sites INTEGER  Maximum number of sites
  --step INTEGER       Step size for the benchmark
  --per_site INTEGER   Number of random examples per site
  --help               Show this message and exit.
```

## Techniques

This repository will be used to benchmark different techniques for a variety of casuistic of the Ising Hamiltonian.

Currently:
* Simulated Annealing
* Quantum Annealing 
* Simulated Bifurcation
* Exact solver (when available)
* DWave Hybrid BQM solver
