import os
import re
from scipy.stats import zscore
import numpy as np
import argparse
import ast

from tvbsim.printer import Printer

def pvalue_to_asterisks(pvalue):
    if pvalue < 0.001:
        return '***'
    if pvalue < 0.01:
        return '**'
    if pvalue < 0.05:
        return '*'
    return 'ns'
        
def get_result(simconfig, additional_path_folder='', time_begin=None, time_end=None, seed=10, vars_int = ['E', 'I', 'W_e'], 
               simu_path=None):
    """
    Loads and returns the results of a simulation for the specified variables of interest within a given time window.
    Results are aggregated across monitors and returned as arrays containing the selected variables, time points, and nodes.
    
    :param simconfig: Simulation configuration object containing simulation parameters and metadata.
    :param additional_path_folder: Optional subfolder within the simulation output directory. Defaults to ''.
    :param time_begin: Start time (in ms) for the extracted results. If None, uses simconfig.cut_transient.
    :param time_end: End time (in ms) for the extracted results. If None, uses simconfig.run_sim.
    :param seed: Integer seed identifying which simulation run to load. Defaults to 10.
    :param vars_int: List of variables of interest to extract. Defaults to ['E', 'I', 'W_e'].
    Available variables include:
    - 'E': excitatory firing rate
    - 'I': inhibitory firing rate
    - 'C_ee': excitatory covariance
    - 'C_ei': excitatory-inhibitory covariance
    - 'C_ii': inhibitory covariance
    - 'W_e': excitatory adaptation
    - 'W_i': inhibitory adaptation
    - 'noise': noise input
    :param simu_path: Optional root path to override the simulation folder. If None, uses simconfig.get_sim_name(seed).
    
    :returns:
    - result: List of length equal to the number of monitors. Each element is a NumPy array with shape
    (n_vars_int, n_time_points, n_nodes), containing the requested variables.
    - (parameter_monitor, vars_int, shape): A tuple with simulation monitor settings, list of extracted variables,
    and shape of the result arrays.
    """
    if time_begin is None:
        time_begin = simconfig.cut_transient
    if time_end is None:
        time_end = simconfig.run_sim
    folder_root, sim_name = simconfig.get_sim_name(seed)
    if simu_path is not None:
        folder_root = simu_path
    Printer.print("Loading: ", folder_root, sim_name)
    path = os.path.join(folder_root, sim_name, additional_path_folder)
#    with open(os.path.join(path, 'parameter.json')) as f:
#        parameters = jsonpickle.decode(f.read())
#    parameter_simulation = parameters['parameter_simulation']
#    parameter_monitor = parameters['parameter_monitor']
    parameter_simulation = simconfig.general_parameters.parameter_simulation
    parameter_monitor = simconfig.general_parameters.parameter_monitor
    # print("parameter monitor: ", parameter_monitor)
    count_begin = int(time_begin/parameter_simulation['save_time'])
    # following is necessary as simulatins could have been aborted earlier than desired end time
    count_end = min(
            int(time_end/parameter_simulation['save_time'])+1,
            # -1 for step_init
            len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name)) and name.startswith('step_')])-1
            )
    nb_monitor = parameter_monitor['Raw'] + parameter_monitor['TemporalAverage'] + parameter_monitor['Bold'] + parameter_monitor['Ca'] #nuu added Ca monitor
    if 'Afferent_coupling' in parameter_monitor.keys() and parameter_monitor['Afferent_coupling']:
        nb_monitor+=1
    output = []
    for count in range(count_begin,count_end):
        result = np.load(path+'/step_'+str(count)+'.npy',allow_pickle=True)
        for i in range(result.shape[0]):
            tmp = np.array(result[i], dtype='object')
            if tmp.size != 0:
                tmp = tmp[np.where((time_begin <= tmp[:,0]) &  (tmp[:,0]<= time_end)),:]
                tmp_time = tmp[0][:,0]
                if tmp_time.shape[0] != 0:
                    one = tmp[0][:,1][0]
                    tmp_value = np.concatenate(tmp[0][:,1]).reshape(tmp_time.shape[0],one.shape[0],one.shape[1])
                    if len(output) == nb_monitor:
                        output[i]=[np.concatenate([output[i][0],tmp_time]),np.concatenate([output[i][1],tmp_value])]
                    else:
                        output.append([tmp_time,tmp_value])
    
    # indices of each variable in the result
    dict_var_int = {'E':0 ,'I': 1 ,'C_ee': 2 ,'C_ei': 3,'C_ii': 4,'W_e': 5,'W_i': 6,'noise': 7}
    len_var = len(vars_int)


    #first iterate the monitors
    result = []
    for i in range(nb_monitor):
        time_s = output[i][0]
        n_nodes = output[i][1][:,0,:].shape[1]
        #create empty array with shape (number of variables of interest, time)
        var_arr = np.zeros((len_var, time_s.shape[0], n_nodes))

        c = 0
        for var in vars_int:
            index_var = dict_var_int[var] #get the index of the variables
            if index_var < 2 or index_var==7: #if it is the exc, inh FR or noise, transform from KHz to Hz
                res = output[i][1][:,index_var,:]*1e3
                var_arr[c] = res
            else:
                res = output[i][1][:,index_var,:]
                var_arr[c] = res
            c+=1 
        
        result.append(var_arr)
    shape = np.shape(result[0][0])
    del output
    # access_results(parameter_monitor,vars_int,shape)
    return result, (parameter_monitor,vars_int,shape)

def retrieve_one_sim(simconfig=None, seed=None, results=None, return_inh=False, root=None, 
                     monitor='Raw', begin=None, end=None, return_TR=False):
    """
    Simplifies access to excitatory (and optionally inhibitory) firing rate data from a simulation by returning NumPy arrays.
    Can either load results from disk via simconfig and seed or use pre-provided results.
    
    :param simconfig: Simulation configuration object used to locate and interpret results. Required if results is not provided.
    :param seed: Simulation seed used to identify which run to load. Required if results is not provided.
    :param results: Pre-loaded simulation results. If provided, bypasses file loading via simconfig and seed.
    :param return_inh: If True, also returns inhibitory firing rates in addition to excitatory ones.
    :param root: Path to the directory containing simulation results. Used if loading from disk.
    :param monitor: String specifying which monitor type to extract data from (default: "Raw").
    :param begin: Start time in ms for the extracted results. Overrides simconfig.cut_transient if given.
    :param end: End time in ms for the extracted results. Overrides simconfig.run_sim if given.
    :param return_TR: If True, also returns the TR (time resolution or monitor metadata) along with the firing rates.
    
    :returns:
    - If return_inh is False and return_TR is False:
    excitatory firing rates (array of shape (time, region))
    - If return_inh is True and return_TR is False:
    (excitatory rates, inhibitory rates)
    - If return_inh is False and return_TR is True:
    (excitatory rates, TR)
    - If return_inh is True and return_TR is True:
    (excitatory rates, inhibitory rates, TR)
    """
    rateE_m = [] # excitatory data
    rateI_m = [] # inhibitory data
    if results is None:
        results, monitor_params = get_result(
                simconfig, seed=seed, vars_int=['E','I'] if return_inh else ['E'], simu_path=root,
                time_begin=begin, time_end=end)
    
    result_fin,TR = create_dicts(
        simconfig, results, monitor, monitor_params, ['E','I'] if return_inh else ['E'], return_TR=True)

    rateE_m = np.array(result_fin['E']) # results[0] : raw monitor
    Printer.print(rateE_m.shape)
    if return_inh:
        rateI_m = np.array(result_fin['I']) 
        if return_TR:
            return rateE_m,rateI_m,TR
        return rateE_m,rateI_m,
    if return_TR:
        return rateE_m,TR
    return rateE_m

def find_file_seed(simconfig, path, file_prefix, seeds=None, n_minimal_seeds=None):
    """
    Finds a simulation result file corresponding to a given configuration and either an explicit list of seeds
    or a minimum required number of seeds. Priority is given to exact seed list matching.
    
    :param simconfig: Simulation configuration object used to derive the simulation name.
    :param path: Optional directory path where simulation result files are stored.
    If None, uses the default path from simconfig.get_sim_name().
    :param file_prefix: String prefix used to identify the simulation result file.
    :param seeds: Optional explicit list of seeds (e.g., [1, 4, 5]) to match in the filename.
    If provided, takes priority over n_minimal_seeds.
    :param n_minimal_seeds: Optional integer specifying the minimal number of seeds required.
    Used only if seeds is not provided.
    
    :returns:
    - Full path to the matching .npy result file if found.
    - False if no matching file is found.
    """
    folder_root, sim_name = simconfig.get_sim_name(0)
    if path is not None:
        folder_root = path

    for file in os.listdir(folder_root):
        filename = os.fsdecode(file)

        # Match filenames ending in _[1,2,3].npy
        match_list = re.match(r'^(.+)_\[(.+)\]\.npy$', filename)
        if match_list and seeds is not None:
            sim_name_found = match_list.group(1)
            seed_str = match_list.group(2)
            try:
                seed_list = ast.literal_eval(f'[{seed_str}]')
            except (SyntaxError, ValueError):
                continue
            if sim_name_found == file_prefix + '_' + sim_name[:-2] and seed_list == seeds:
                Printer.print(filename, 'found (exact seed list match).')
                _, simconfig_name = simconfig.get_sim_name(len(seeds))
                return os.path.join(folder_root, f'{file_prefix}_{simconfig_name}.npy')

        # Match filenames ending in _10.npy (single integer)
        match_num = re.match(r'^(.+)_([0-9]+)\.npy$', filename)
        if match_num and n_minimal_seeds is not None and seeds is None:
            sim_name_found = match_num.group(1)
            n_seeds = int(match_num.group(2))
            if sim_name_found == file_prefix + '_' + sim_name[:-2] and n_seeds >= n_minimal_seeds:
                Printer.print(filename, f'found (minimal seed count match) at path {path}.')
                _, simconfig_name = simconfig.get_sim_name(n_seeds)
                return os.path.join(folder_root, f'{file_prefix}_{simconfig_name}.npy')

    return False

def create_dicts(simconfig, result, monitor, for_explan, var_select, return_TR=False):
    """
    Creates a dictionary mapping selected simulation variables to their corresponding result arrays,
    for a given monitor type. Can optionally return the temporal resolution (TR).
    
    :param simconfig: Simulation configuration object containing general and custom parameters.
    :param result: List of result arrays from monitors, as returned by get_result.
    :param monitor: String specifying which monitor to extract (e.g., "Raw", "TemporalAverage", "Bold").
    :param for_explan: Tuple (parameter_monitor, vars_int, Nnodes) returned by get_result,
    containing monitor settings, variables of interest, and node count.
    :param var_select: List of variables to extract from the results (e.g., ["E", "I"]).
    :param seed: Simulation seed used for naming or reproducibility (currently unused in this function).
    :param additional_path_folder: Optional string specifying an extra path for locating results (currently unused in this function).
    :param return_TR: If True, also returns the temporal resolution (TR).
    - For "Raw": TR = integration step (dt).
    - For other monitors: TR = monitor period from parameter_monitor.
    
    :returns:
    - If return_TR is False: Dictionary mapping variable names to arrays of shape (time, n_regions_filtered).
    - If return_TR is True: (result_dict, TR)
    """

    parameters = simconfig.general_parameters
    parameter_monitor = parameters.parameter_monitor

    list_monitors = {} # {Raw : 0, Temporal:1, etc}
    c = 0
    for key in parameter_monitor.keys():
        if parameter_monitor[key] is True:
            list_monitors[key] = c
            c += 1
    
    result = result[list_monitors[monitor]] #take the wanted monitor

    #Take the variables of interest
    (_,vars_int,_) = for_explan

    list_vars = {}
    k = 0
    for var in vars_int:
        list_vars[var] = k
        k += 1
    
    result_fin = {}

    n_regions = result[0].shape[1]
    for var in var_select:
        result_fin[var] = result[list_vars[var]][:,[x for x in range(n_regions) if x not in simconfig.custom_parameters.get('disconnect_regions',[])]]
        if monitor == 'Bold': #Z score if it is bold
            if var == 'E' or var == 'I':
                result_fin[var] = zscore(result[list_vars[var]])

    if return_TR:
        TR = parameters.parameter_integrator['dt'] if monitor == 'Raw' else parameter_monitor[f"parameter_{monitor}"]["period"]
        return result_fin,TR
    else:
        return result_fin

def get_np_linspace(value):
   """
   solution to input np.arange in the argparser 
   """
   try:
       values = [float(i) for i in value.split(',')]
       assert len(values) in (1, 3)
   except (ValueError, AssertionError):
       raise argparse.ArgumentTypeError(
           'Provide a CSV list of 1 or 3 integers'
       )

   # return our value as is if there is only one
   if len(values) == 1:
       return np.array(values)

   # if there are three - return a range
   values[-1] = int(values[-1])
   return np.linspace(*values)

def get_np_arange(value):
   """
#   solution to input np.arange in the argparser 
   """
   try:
       values = [float(i) for i in value.split(',')]
       assert len(values) in (1, 3)
   except (ValueError, AssertionError):
       raise argparse.ArgumentTypeError(
           'Provide a CSV list of 1 or 3 integers'
       )

   # return our value as is if there is only one
   if len(values) == 1:
       return np.array(values)

   # if there are three - return a range
   return np.arange(*values)

def replace_dict_values(main_dict, replacements):
    for name,value in replacements.items():
        try:
            main_dict[name] = value
        except KeyError:
            Printer.print(f'Parameter {name} with value {value} is not a valid key in dictionary {main_dict}.')

def print_dict_differences(dict1, dict2):
    # Iterate through keys of the first dictionary
    for key in dict1:
        # Check if the key exists in the second dictionary
        if key in dict2:
            # Check if the values are different
            if dict1[key] != dict2[key]:
                print(f"Difference in key '{key}':")
                print(f"   New dictionary value: {dict1[key]}")
                print(f"   Preexisting dictionary value: {dict2[key]}")
            else:
                no_diff = True
        else:
            print(f"Key '{key}' not found in preexisting dictionary")
            no_diff=False
    # Check for keys in the second dictionary not present in the first
    for key in dict2:
        if key not in dict1:
            print(f"Key '{key}' not found in new dictionary")
            no_diff=False
    return no_diff

def compare_dicts(dict1, dict2):
    """
    Compare two dictionaries for equality.
    """
    if len(dict1) != len(dict2):
        return False
    
    for key in dict1:
        if key not in dict2:
            return False
        if dict1[key] != dict2[key]:
            return False
    
    return True