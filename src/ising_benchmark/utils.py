import dimod
import numpy as np
import time
from dwave.cloud import Client

def ising_generator(N:int):
    """
    Creates a random Ising problem
    
    """
    # Sample from normal distribution
    h = np.random.normal(0.0, 1.0, N)
    J = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1, N):
            J[i][j] = np.random.normal(0.0, 1.0)
            J[j][i] = J[i][j]

    return h, J

def ising_to_bqm(h: list, J: np.ndarray):

    N = len(h)
    h_dict = {f'x[{i}]': val for i,val in enumerate(h)}
    J_dict = {}
    for i in range(N):
        for j in range(i+1, N):
            J_dict[(f'x[{i}]',f'x[{j}]')] = J[i][j]
            J_dict[(f'x[{j}]',f'x[{i}]')] = J[i][j]

    return dimod.BinaryQuadraticModel(h_dict, J_dict, dimod.SPIN)

def bqm_spin_array(sample: dict):

    num = len(list(sample.keys()))
    spin_vector = np.zeros(num)
    for key in sample:
        indx = int(key.replace("x[","").replace("]",""))
        spin_vector[indx] = sample[key]

    return spin_vector

def compute_energy(h, J, solution):

    return np.dot(np.transpose(solution), np.dot(J, solution))+np.dot(h, solution)

def submit_dwave(problem, token:str):
    """
    Uses D-Wave annealer to solve the given problem
    """
    # Connect using the default or environment connection information
    with Client.from_config(token=token) as client:
        qpu = client.get_solver(name='hybrid_binary_quadratic_model_version2')
        sampler = qpu.sample_bqm(problem, label="Ising benchmark", time_limit = 10)

        # Wait until it finishes
        while not sampler.done():
            time.sleep(5)

        result = sampler.result()

        return result['sampleset']
