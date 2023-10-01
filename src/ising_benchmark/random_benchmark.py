import os
import click
import dimod
import neal
import torch
import pickle
import simulated_bifurcation as sb
from dotenv import load_dotenv

from ising_benchmark.utils import (
    ising_generator,
    ising_to_bqm,
    bqm_spin_array,
    submit_dwave
)

load_dotenv()

@click.command()
@click.option('--max_sites', default=10, help='Maximum number of sites')
@click.option('--step', default=5, help='Step size for the benchmark')
@click.option('--per_site', default=5, help='Number of random examples per site')
def random_benchmark(max_sites, step, per_site):
    """Randomized Ising benckmark"""
    results = {}

    sa_sampler = neal.SimulatedAnnealingSampler()
    for N in range(step, max_sites, step):

        exact_e = []
        exact_sol = []
        sa_e = []
        sa_sol = []
        sb_e = []
        sb_sol = []
        dwave_e = []
        dwave_sol = []
        for _ in range(per_site):
            h, J = ising_generator(N)
            bqm = ising_to_bqm(h, J)

            # Exact
            if N < 20:
                sampleset = dimod.ExactSolver().sample(bqm)

                e = bqm.energy(sampleset.first.sample)
                solution = bqm_spin_array(sampleset.first.sample)

                exact_e.append(e)
                exact_sol.append(solution)
            else:
                exact_e.append(None)
                exact_sol.append([])

            # Dwave
            try:
                sampleset = submit_dwave(bqm, os.environ["TOKEN"])

                e = bqm.energy(sampleset.first.sample)
                solution = bqm_spin_array(sampleset.first.sample)

                dwave_e.append(e)
                dwave_sol.append(solution)
            except:
                dwave_e.append(None)
                dwave_sol.append([])

            # SA
            # Run with default parameters
            sampleset = sa_sampler.sample(bqm)

            e = bqm.energy(sampleset.first.sample)
            solution = bqm_spin_array(sampleset.first.sample)

            sa_e.append(e)
            sa_sol.append(solution)

            # SB
            h_torch = torch.tensor(h, dtype=torch.float32)
            J_torch = torch.tensor(J, dtype=torch.float32)

            # Binary minimization
            solution, e = sb.minimize(J_torch, h_torch, input_type='spin')

            sb_e.append(e)
            sb_sol.append(solution)

        # Append
        results[f"{N}"] = {
            "exact" : {
                "energies" : exact_e,
                "solutions" : exact_sol
            },
            "sa" : {
                "energies" : sa_e,
                "solutions" : sa_sol
            },
            "dwave" : {
                "energies" : dwave_e,
                "solutions" : dwave_sol
            },
            "sb" : {
                "energies" : sb_e,
                "solutions" : sb_sol
            }
        }

    # Store  
    with open(f'results/{max_sites}_{step}_{per_site}.pkl', 'wb') as file:
        pickle.dump(results, file)

    # Save
if __name__ == '__main__':
    random_benchmark()