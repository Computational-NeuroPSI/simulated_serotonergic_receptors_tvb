import os
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
from enum import Enum
from dataclasses import dataclass
from typing import Callable

from tvbsim.printer import Printer
from tvbsim.common import find_file_seed, retrieve_one_sim
from tvbsim.entropy.measures_functions import calculate_LempelZiv, calculate_LempelZiv_single, calculate_ACE, calculate_SCE

@dataclass(frozen=True)
class EntropyInfo:
    file_prefix: str
    label_name: str
    func: Callable

class Entropy(Enum):
    LZc = EntropyInfo('LZc', 'Multidimensional Lempel-Ziv complexity', calculate_LempelZiv)
    LZc_single = EntropyInfo('LZc_single', 'Lempel-Ziv complexity', calculate_LempelZiv_single)
    ACE = EntropyInfo('ACE', 'Amplitude Coalition Entropy', calculate_ACE)
    SCE = EntropyInfo('SCE', 'Synchrony Coalition Entropy', calculate_SCE)
    
NAME2ENTROPY = {entropy.value.file_prefix : entropy.value for entropy in Entropy}

# TODO: begin_time, end_time ?
def parallelized_entropy(simconfig, seeds=None, entropy_name='LZc', simu_path=None, save_path=None, 
                         n_tasks_concurrent=None, ignore_exists=False, data=None):
    """
        Computes in parallel some entropy measure of the different seeds of a Simconfig on the excitatory firing rates
    
        :param simconfig: SimConfig in question
        :param seeds: seeds for which SimConfig was ran
        :param method_name: string of the wanted method
            - "LZc" : multi-dim Lempel-Ziv complexity
            - "LZc_single" : average single-channel Lempel-Ziv complexity
            - "ACE" : Amplitude Coalition Entropy
            - "SCE" : Synchrony Coalition Entropy
        :param simu_path: folder where simulation
        :param save_path: where to save the results
        :param n_tasks_concurrent: nb of parallel threads (default max - 1)
        :param data: should be a list for all seeds of tuples with 1 element (data array), as is obtained
            in data = [(retrieve_one_sim(simconfig, seed, root=simu_path),) for seed in seeds]
    """    
    if seeds is None and data is None:
        raise ValueError('Both seeds and data are none: give simulation data or specify seeds so the function gathers data.')
    n_seeds = len(data if data is not None else seeds)
    
    try:
        entropy = NAME2ENTROPY[entropy_name]
    except KeyError:
        Printer.print(f'Entropy {entropy_name} does not exist, choose among {NAME2ENTROPY.keys()}')
        return

    found_path = find_file_seed(simconfig, save_path, entropy_name, n_minimal_seeds=n_seeds)
    if not ignore_exists and found_path:
        Printer.print(f'Already exists at path {found_path} ; skipping computation.')
        return np.load(found_path)[:n_seeds]
    
    if n_tasks_concurrent is None:
        n_tasks_concurrent = cpu_count()-1
        
    if data is None:
        data = [retrieve_one_sim(simconfig, seed, root=simu_path) for seed in seeds]
    data = [(d,) for d in data]
    with Pool(n_tasks_concurrent) as p: # could be increased, then concurrent computing
        results = p.starmap(entropy.func, data)
    # Save data
    folder_root, sim_name = simconfig.get_sim_name(n_seeds)
    if save_path is None:
        save_path = folder_root
    save_file_name = f'{entropy_name}_{sim_name}.npy'
    np.save(os.path.join(save_path, save_file_name), results)
    Printer.print(f'Following config done:\n {simconfig}')
    Printer.print('Saved at :', save_file_name)
    return results