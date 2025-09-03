import os
import numpy as np
import json
from multiprocessing import Pool, cpu_count

from tvbsim.TVB.tvb_model_reference.src import nuu_tools_simulation_human as tools
from tvbsim.TVB import pci_v2 as pci_v2
from tvbsim.plotting import plot_box
from tvbsim.printer import Printer
from tvbsim.common import find_file_seed

def parallelized_PCI(simconfig, seeds, n_tasks_concurrent=None, 
                     t_analysis=300, n_trials=5, simu_path=None, save=0, save_path=None, ignore_exists=False):
    """
        Compute the Perturbational Complexity Index (PCI) across seeds in parallel.
    
        :param simconfig: Simulation configuration object.
        :param seeds: List of seed integers; length must be divisible by n_trials.
        :param n_tasks_concurrent: Number of worker processes to use; defaults to cpu_count()-1.
        :param t_analysis: Time window (ms) around stimulation used for PCI analysis.
        :param n_trials: Number of seeds per subset used to build the baseline per computation batch.
        :param simu_path: Directory containing simulation result folders; defaults to simconfig's root.
        :param save: 0 = return only; 1 = save and return.
        :param save_path: Directory to save the computed PCI array when save > 0.
        :param ignore_exists: If False, load existing results when found; if True, recompute.
        :returns: NumPy array of PCI values, one per seed.
    """
    assert(len(seeds) % n_trials == 0), "Number of seeds must be divisible by n_trials"
    
    # TODO: add n_trials t_analysis in name of file
    found_path = find_file_seed(simconfig, save_path, f'PCI_t_analysis_{t_analysis}_n_trials_{n_trials}', n_minimal_seeds=len(seeds))
    if not ignore_exists and found_path:
        Printer.print(f'Already exists at path {found_path} ; skipping computation.')
        return np.load(found_path)
    Printer.print('No files found.')
    
    default_root,sim_name = simconfig.get_sim_name(len(seeds))
    if simu_path is None:
        simu_path = default_root
        
    if n_tasks_concurrent is None:
        n_tasks_concurrent = cpu_count()-1
        
    with Pool(n_tasks_concurrent) as p:
        pcis = p.starmap(
                _calculate_PCI_seed_subset, 
                [(simconfig, seed_subset, t_analysis, simu_path) 
                for seed_subset in [seeds[i:i+n_trials] for i in range(0, len(seeds), n_trials)]])
        pcis = np.array(pcis).flatten()
        
    # File saving
    os.makedirs(save_path, exist_ok=True)
    save_file_name_PCI = os.path.join(save_path, f'PCI_t_analysis_{t_analysis}_n_trials_{n_trials}_{sim_name}.npy')
#    save_file_name_entropy = os.path.join(save_path, f'Params_entropy_{sim_name}.npy')
#    save_file_name_LempelZiv = os.path.join(save_path, f'Params_LempelZiv_{sim_name}.npy')

    if save > 0:
        np.save(save_file_name_PCI, np.array(pcis))
#        np.save(save_file_name_entropy, np.array(entropies))
#        np.save(save_file_name_LempelZiv, np.array(LZs))
    
        Printer.print(save_file_name_PCI)
#        Printer.print(save_file_name_entropy)
#        Printer.print(save_file_name_LempelZiv)

    Printer.print(pcis)
#    Printer.print(entropy_trials)
#    Printer.print(LZs)
    #clear_output(wait=False)
    Printer.print(f"Done: {simconfig}")
    return np.array(pcis)
    
def _calculate_PCI_seed_subset(simconfig, seeds, t_analysis, root):
    """
        Compute PCI for a subset of seeds using pre/post-stimulation windows.
    
        :param simconfig: Simulation configuration object.
        :param seeds: List of seed integers in this subset.
        :param t_analysis: Time window (ms) around stimulation used for PCI analysis.
        :param root: Directory containing simulation result folders.
        
        :returns: NumPy array of PCI values for the provided seeds.
    """
    np.random.seed(0)
    sig_cut_analysis = []
    t_stim_onsets = []
    for i_trials in seeds: 
        
        times_l = []
        rateE_m = []
        nstep = int(simconfig.run_sim/simconfig.general_parameters.parameter_simulation['save_time']) # number of saved files

        _, sim_name = simconfig.get_sim_name(i_trials)
        folder_path = os.path.join(root, sim_name)

        for i_step in range(nstep):
            raw_curr = np.load(os.path.join(folder_path, 'step_'+str(i_step)+'.npy'),
                               encoding = 'latin1', allow_pickle=True)
            for i_time in range(len(raw_curr[0])): 
                times_l.append(raw_curr[0][i_time][0])
                rateE_m.append(np.concatenate(raw_curr[0][i_time][1][0]))

        times_l = np.array(times_l) # in ms
        rateE_m = np.array(rateE_m) # matrix of size nbins*nregions

        # choosing variable of interest
        var_of_interest = rateE_m
        
        # discard transient
        nbins_transient = int(simconfig.cut_transient/times_l[0]) # to discard in analysis   
        sig_region_all = var_of_interest[nbins_transient:,:] 
        sig_region_all = np.transpose(sig_region_all) # now formatted as regions*times

        # load t_onset
        with open(os.path.join(folder_path, "parameter.json"), 'r') as json_file:
            data = json.load(json_file)
        onset_value = data['parameter_stimulus']['stimtime']
          
        t_stim_bins = int((onset_value - simconfig.cut_transient)/times_l[0])
        
        #save all the onsets:
        t_stim_onsets.append(t_stim_bins)

        nbins_analysis =  int(t_analysis/times_l[0])
        
        sig_cut_region =  sig_region_all[:,t_stim_bins - nbins_analysis:t_stim_bins + nbins_analysis]
        
        # append directly the sig_cut_analysis
        sig_cut_analysis.append(sig_cut_region)

    sig_all_binary = tools.binarise_signals(np.array(sig_cut_analysis), int(t_analysis/times_l[0]), 
                                    nshuffles = 10, percentile = 100)
    
    entropy_trials = []
    LZ_trials = []
    PCI_trials = []
    #return entropy
    for ijk,_ in enumerate(seeds):
        binJ=sig_all_binary.astype(int)[ijk,:,t_analysis:] # CHECK each row is a time series !
        binJs=pci_v2.sort_binJ(binJ) # sort binJ as done in Casali et al. 2013
        source_entropy=pci_v2.source_entropy(binJs)
        Printer.print('Entropy', source_entropy)

        # return Lempel-Ziv
        Lempel_Ziv_lst=pci_v2.lz_complexity_2D(binJs)
        Printer.print('Lempel-Ziv', Lempel_Ziv_lst)

        #normalization factor 
        norm=pci_v2.pci_norm_factor(binJs)

        # computing perturbational complexity index
        pci_lst = Lempel_Ziv_lst/norm
        Printer.print('PCI', pci_lst)

        entropy_trials.append(source_entropy) 
        LZ_trials.append(Lempel_Ziv_lst) 
        PCI_trials.append(pci_lst)
    
    return np.array(PCI_trials)
    
def plot_PCI(set_params=None, **kwargs):
    """
        Plot boxplots of PCI values using tvbsim.plotting.plot_box.
    
        :param set_params: Dict of axis parameters passed to the plotting function.
        :param kwargs: Additional keyword arguments forwarded to plot_box.
        :returns: None.
    """
    if set_params is None:
        set_params = {}
    set_params['ylabel'] = 'Perturbational Complexity Index'
    plot_box(file_prefix='PCI', set_params=set_params, **kwargs)