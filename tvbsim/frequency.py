import matplotlib.pyplot as plt
import numpy as np
from einops import reduce
import os
from scipy.stats import ttest_ind

from tvbsim.printer import Printer
from tvbsim.common import retrieve_one_sim, pvalue_to_asterisks, find_file_seed

SEPARATIONS = [1,4,8,13,25,60]
BAND_NAMES = ['δ','θ','α','β','γ']


def spectrogram(simconfig, seed, window, noverlap, root=None, begin=None, end=None, regions=None, data=None, 
                save_path=None, ignore_exists=False, monitor='Raw', TR=None):
    pass

def power(simconfig, seed, root=None, begin=None, end=None, regions=None, data=None, 
          save_path=None, ignore_exists=False, monitor='Raw', TR=None):
    """
    Returns the power spectrum of excitatory activity from a simulation, either using pre-saved results
    (if available) or computing it via FFT. Can return frequency values and power per region, and optionally saves results.
    
    :param simconfig: Simulation configuration object used to locate and identify results.
    :param seed: Integer seed specifying which simulation run to use.
    :param root: Optional path to the folder containing simulation results.
    :param begin: Start time (ms) for the analysis. Defaults to simconfig.cut_transient if not provided.
    :param end: End time (ms) for the analysis. Defaults to simconfig.run_sim if not provided.
    :param regions: List of region indices to include in the power computation. If None, all regions are included.
    :param data: Optional pre-loaded excitatory firing rate data (time × regions).
        If None, it will be loaded using retrieve_one_sim.
    :param save_path: Path to save computed frequency and power results.
        If None, defaults to the simulation’s folder.
    :param ignore_exists: If False, reuses precomputed files (if found).
        If True, recomputes even if saved files exist.
    :param monitor: Monitor type used when loading data (default: "Raw").
    :param TR: Optional temporal resolution. If None, it is obtained from retrieve_one_sim.
    
    :returns:
    - frq: 1D NumPy array of frequency values (Hz).
    - pwr_region_E: 2D NumPy array of power values, shape (frequencies, regions).
    """
    if begin is None:
        begin = simconfig.cut_transient
    if end is None:
        end = simconfig.run_sim
    frq_found_path = find_file_seed(simconfig, save_path, 'freq', n_minimal_seeds=seed)
    power_found_path = find_file_seed(simconfig, save_path, 'power', n_minimal_seeds=seed)
    if not ignore_exists and frq_found_path and power_found_path:
        Printer.print(f'Already exists at path {frq_found_path} ; skipping computation.')
        return np.load(frq_found_path), np.load(power_found_path)
        
    if data is None:
        data,TR = retrieve_one_sim(simconfig, seed, root=root, begin=begin, end=end, monitor=monitor, return_TR=True)
    time_s = np.arange(0, int(end - begin)//1000, 1e-3*TR) # TODO: 1000 and 0.0001 should be computed wrt dt
    data = data.T
    f_sampling = 1.*len(time_s)/(int(end-begin)*1e-3) # time in seconds, f_sampling in Hz
    frq = np.fft.fftfreq(len(time_s), 1/f_sampling)
    if regions is None:
        regions = range(data.shape[0])
    pwr_region_E = []
    for e_reg in regions:
        pwr_region_E.append(np.abs(np.fft.fft(data[e_reg]))**2)
    pwr_region_E = np.array(pwr_region_E).T
    frq = np.array(frq)
    folder_root, sim_name = simconfig.get_sim_name(seed)
    if save_path is None:
        save_path = folder_root
    np.save(os.path.join(save_path, 'freq_' + sim_name + '.npy'), frq)
    np.save(os.path.join(save_path, 'power_' + sim_name + '.npy'), pwr_region_E)
    
    return frq, pwr_region_E

def bands_diffs_pvalues(pow1, pow2, frq):
    """
        :param pow1: frequency, region
    """
    diffs = np.zeros(len(BAND_NAMES))
    pvalues = np.zeros(len(BAND_NAMES))
    for i,(low_sep,high_sep,freq_name) in enumerate(zip(SEPARATIONS,SEPARATIONS[1:],BAND_NAMES)):
        low_sep_idx = np.argmin(np.abs(low_sep - frq))
        high_sep_idx = np.argmin(np.abs(high_sep - frq))
        data1 = reduce(pow1[low_sep_idx:high_sep_idx], 'f r -> r', np.mean)
        data2 = reduce(pow2[low_sep_idx:high_sep_idx], 'f r -> r', np.mean)
        diffs[i] = (np.mean(data2) - np.mean(data1)) / np.mean(data1)
        pvalue = ttest_ind(data1, data2, equal_var=False).pvalue
        pvalues[i] = pvalue
    return diffs,pvalues

def bands_diffs_pvalues_seeds(pow1, pow2, frq):
    """
        :param pow1: seeds, frequency, region
    """
    diffs = np.zeros(len(BAND_NAMES))
    pvalues = np.zeros(len(BAND_NAMES))
    for i,(low_sep,high_sep,freq_name) in enumerate(zip(SEPARATIONS,SEPARATIONS[1:],BAND_NAMES)):
        low_sep_idx = np.argmin(np.abs(low_sep - frq))
        high_sep_idx = np.argmin(np.abs(high_sep - frq))
        data1 = reduce(pow1[:,low_sep_idx:high_sep_idx], 's f r -> (s r)', np.mean)
        data2 = reduce(pow2[:,low_sep_idx:high_sep_idx], 's f r -> (s r)', np.mean)
        diffs[i] = (np.mean(data2) - np.mean(data1)) / np.mean(data1)
        pvalue = ttest_ind(data1, data2, equal_var=False).pvalue
        pvalues[i] = pvalue
    return diffs,pvalues


# TODO: could add pvalue for specific case where several (2) simconfigs
# TODO: add possibility to average across seeds
def plot_mean_power(simconfigs, root, seed=None, seeds=None, regions=None, figsize=(6,5), labels=None, color1s=None, color2s=None, y_txt=None,
                    save=0, save_path=None, file_name=None, data=None, ignore_exists=False):
    """
    Plots the mean power spectral density (PSD) across regions for one or more simulations, for one seed per simconfig,
    with shaded error bands and statistical comparisons if there are 2 simulations.
    
    :param simconfigs: List of simulation configuration objects whose PSDs are to be plotted.
    :param root: Path to the directory containing simulation results.
    :param seed: Single seed to use for all simulations (ignored if seeds is provided).
    :param seeds: List of seeds, one for each simulation in simconfigs. Overrides seed.
    :param regions: List of region indices to include in the PSD calculation. Defaults to all regions.
    :param figsize: Tuple specifying the size of the figure (width, height). Defaults to (6, 5).
    :param labels: List of labels for each simulation in the plot legend. Defaults to None for each.
    :param color1s: List of colors for the mean PSD curves of each simulation. Defaults to matplotlib’s cycle.
    :param color2s: List of colors for the shaded error regions of each simulation. Defaults to matplotlib’s cycle.
    :param y_txt: Vertical position for frequency band annotation text. If None, defaults internally to 10**4.
    :param save: Integer flag controlling saving and displaying of the plot.
        - 0: Show only
        - 1: Save and show
        - 2: Save and close (no display)
    :param save_path: Path to save the figure. Created if it does not exist. Defaults to empty string if None.
    :param file_name: File name for saving the figure. Defaults to a generated name based on simconfigs and seeds.
    :param data: Optional precomputed tuple (frq, pwr_region_E). If provided, bypasses recomputation with power().
    :param ignore_exists: If True, forces recomputation of power even if files already exist.
    
    :returns: None. Displays and/or saves the generated PSD plot depending on save.
    """
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = ''
        
    if labels is None:
        labels = [None] * len(simconfigs)
    if color1s is None:
        color1s = [None] * len(simconfigs)
    if color2s is None:
        color2s = [None] * len(simconfigs)
        
    if seeds is None:
        seeds = [seed] * len(simconfigs)
    
    fig, ax = plt.subplots(figsize=figsize)
    pwr_region_E_simconfigs = []
    for simconfig,seed,color1,color2,label in zip(simconfigs,seeds,color1s,color2s,labels):
        if data is None:
            frq,pwr_region_E = power(simconfig, seed, root, save_path=root, regions=regions, ignore_exists=ignore_exists)
        else:
            frq,pwr_region_E = data
        pwr_region_E = np.array(pwr_region_E)
        pwr_region_E_simconfigs.append(pwr_region_E)
        mean_E_Hz = reduce(pwr_region_E, 'f r -> f', np.mean)
        std_e = reduce(pwr_region_E, 'f r -> f', np.std) #std fft between regions
        high_e = mean_E_Hz[frq > 0]+std_e[frq > 0]/np.sqrt(pwr_region_E.shape[1])
        low_e =  mean_E_Hz[frq > 0]-std_e[frq > 0]/np.sqrt(pwr_region_E.shape[1])
        ax.loglog(frq[frq > 0], mean_E_Hz[frq > 0], color=color1, label=label)
        ax.fill_between(frq[frq > 0], high_e, low_e, color=color2, alpha=0.4)

    # print 
    y_txt = 10**4
    pvalue_asterisks = ''
    if len(simconfigs) == 2:
        _,pvalues = bands_diffs_pvalues(pwr_region_E_simconfigs[0], pwr_region_E_simconfigs[1], frq)
        for low_sep,high_sep,freq_name,pvalue in zip(SEPARATIONS,SEPARATIONS[1:],BAND_NAMES,pvalues):
            pvalue_asterisks = pvalue_to_asterisks(pvalue)
            plt.text(x=np.exp((np.log(low_sep)+np.log(high_sep))/2), y=y_txt-3000, s=pvalue_asterisks)
            ax.vlines(x=low_sep, ymin=0, ymax=1e15, ls='--')
            plt.text(x=np.exp((np.log(low_sep)+np.log(high_sep))/2), y=y_txt, s=freq_name)

    ax.legend()
    ax.set_xlim(left=1, right=100)
    ax.set_ylim(bottom=10**3)
    if save >= 1:
        if file_name is None:
            file_name = 'freq_' + ', '.join([simconfig.get_plot_title() for simconfig in simconfigs]) + f'_seeds_{seeds}.npy'
        fig.savefig(os.path.join(save_path, file_name))
    if save < 2:
        plt.tight_layout()
        plt.show()
    plt.close()

def plot_mean_power_seeds(
        simconfigs, root, seeds=None, seeds_sets=None, regions=None, figsize=(6,5), labels=None, color1s=None, color2s=None, y_txt=None,
        save=0, save_path=None, file_name=None, monitor='Raw', ignore_exists=False, data=None, TR=None):
    """
    Plots the mean power spectral density (PSD) across seeds and regions for one or more simulations,
    with shaded error bands and optional statistical comparisons between two simulations.
    
    :param simconfigs: List of simulation configuration objects whose PSDs are to be plotted.
    :param root: Path to the directory containing simulation results.
    :param seeds: List of seeds to use for all simulations (applied if `seeds_sets` is not provided).
    :param seeds_sets: List of lists of seeds, one list per simulation. Overrides `seeds`.
    :param regions: List of region indices to include in the PSD calculation. Defaults to all regions.
    :param figsize: Tuple specifying the size of the figure (width, height). Defaults to (6, 5).
    :param labels: List of labels for each simulation in the plot legend. Defaults to None for each.
    :param color1s: List of colors for the mean PSD curves of each simulation. Defaults to matplotlib’s cycle.
    :param color2s: List of colors for the shaded error regions of each simulation. Defaults to matplotlib’s cycle.
    :param y_txt: Vertical position for frequency band annotation text. If None, defaults internally to `10**4`.
    :param save: Integer flag controlling saving and displaying of the plot.
                 - 0: Show only
                 - 1: Save and show
                 - 2: Save and close (no display)
    :param save_path: Path to save the figure. Created if it does not exist. Defaults to empty string if None.
    :param file_name: File name for saving the figure. Defaults to a generated name based on simconfigs and seeds.
    :param monitor: Which monitor type to retrieve results from (default: "Raw").
    :param ignore_exists: If True, forces recomputation of power even if files already exist.
    
    :returns: None. Displays and/or saves the generated PSD plot depending on `save`.
    """
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = ''
        
    if labels is None:
        labels = [None] * len(simconfigs)
    if color1s is None:
        color1s = [None] * len(simconfigs)
    if color2s is None:
        color2s = [None] * len(simconfigs)
        
    if seeds_sets is None:
        seeds_sets = [seeds] * len(simconfigs)
    
    fig, ax = plt.subplots(figsize=figsize)
    pwr_region_E_simconfigs = []
    for i,(simconfig,seeds_set,color1,color2,label) in enumerate(zip(simconfigs,seeds_sets,color1s,color2s,labels)):
        pwr_region_E_seeds = []
        for j,seed in enumerate(seeds_set):
            frq,powr = power(simconfig, seed, root, save_path=root, regions=regions, monitor=monitor, ignore_exists=ignore_exists,
                             data=data[i,j] if data is not None else None, TR=TR)
            pwr_region_E_seeds.append(powr)
        pwr_region_E_seeds = np.array(pwr_region_E_seeds)
        pwr_region_E_simconfigs.append(pwr_region_E_seeds)
        mean_E_Hz = reduce(pwr_region_E_seeds, 's f r -> f', np.mean)
        std_e = reduce(pwr_region_E_seeds, 's f r -> f', np.std) #std fft between regions
        high_e = mean_E_Hz[frq > 0]+std_e[frq > 0]/(np.sqrt(pwr_region_E_seeds.shape[-1])*len(seeds_set))
        low_e =  mean_E_Hz[frq > 0]-std_e[frq > 0]/(np.sqrt(pwr_region_E_seeds.shape[-1])*len(seeds_set))
        ax.loglog(frq[frq > 0], mean_E_Hz[frq > 0], color=color1, label=label)
        ax.fill_between(frq[frq > 0], high_e, low_e, color=color2, alpha=0.4)
    # pwr_region_E_simconfigs : (n_simconfigs, s, f, r)

    y_txt = 10**4
    pvalue_asterisks = ''
    if len(simconfigs) == 2:
        _,pvalues = bands_diffs_pvalues_seeds(pwr_region_E_simconfigs[0], pwr_region_E_simconfigs[1], frq)
        for low_sep,high_sep,freq_name,pvalue in zip(SEPARATIONS,SEPARATIONS[1:],BAND_NAMES,pvalues):
            pvalue_asterisks = pvalue_to_asterisks(pvalue)
            plt.text(x=np.exp((np.log(low_sep)+np.log(high_sep))/2), y=y_txt-3000, s=pvalue_asterisks)
            ax.vlines(x=low_sep, ymin=0, ymax=1e15, ls='--')
            plt.text(x=np.exp((np.log(low_sep)+np.log(high_sep))/2), y=y_txt, s=freq_name)

    ax.legend()
    ax.set_xlim(left=1, right=100)
    ax.set_ylim(bottom=10**3)
    if save >= 1:
        if file_name is None:
            file_name = 'freq_' + ', '.join([simconfig.get_plot_title() for simconfig in simconfigs]) + f'_seeds_{seeds}.npy'
        fig.savefig(os.path.join(save_path, file_name))
    if save < 2:
        plt.tight_layout()
        plt.show()
    plt.close()
