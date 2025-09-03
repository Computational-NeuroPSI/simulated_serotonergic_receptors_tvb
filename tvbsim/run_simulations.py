import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count
import os
import shutil
import jsonpickle

from tvbsim.simconfig import sim_init
from tvbsim.printer import Printer
from tvbsim.entropy.measures import parallelized_entropy

def run_n_complete_parallelized(simconfig, max_seed, n_minimal_seeds=None, n_tasks_concurrent=None, 
                                save_path_root=None, no_skip=False, param_change_with_seed=None):
    """
        Run simulations for a single simconfig in parallel across seeds until a number of seeds complete.
    
        :param simconfig: SimConfig to run.
        :param max_seed: Maximum number of seeds to try
        :param n_minimal_seeds: Minimum number of successfully completed seeds required; if None, defaults to len(seeds).
        :param n_tasks_concurrent: Number of parallel worker processes; defaults to cpu_count() - 1.
        :param save_path_root: Root directory where simulation result folders will be created.
        :param no_skip: If True, run simulations even when existing results would normally cause a skip.
        :param param_change_with_seed: Optional list of dicts, each dict representing parameter changes for a specific seed (useful to have different stimtimes for each seed for instance when doing PCI).
        :returns: List of completed seed integers when requirement met; otherwise False.
    """

    if n_tasks_concurrent is None:
        n_tasks_concurrent = cpu_count() - 1 # - 1 so the computer is still usable if it's ran on a computer...
    if param_change_with_seed is None:
        param_change_with_seed = [{}] * max_seed
    if n_minimal_seeds is None:
        n_minimal_seeds = max_seed
    seeds = list(range(max_seed))
    seeds_completed = []
    for i in range(0, max_seed, n_tasks_concurrent):
        with Pool(n_tasks_concurrent) as p: # could be increased, then concurrent computing
            results = p.starmap(run_simulation,
                                [(simconfig, seed, False, save_path_root, no_skip, param_change_with_seed[seed]) 
                                for seed in seeds[i:i+n_tasks_concurrent]])
            for j,res in enumerate(results):
                if res:
                    seeds_completed.append(i+j)
                    if len(seeds_completed) == n_minimal_seeds:
                        return seeds_completed
    return False

# multiprocessing optimization only worth it here if len(simconfigs) > n_cpu
# (should be the case; the point of this function is to run simulations on many configs without storing all the data)
def run_and_compute_parallel(simconfigs, seeds, file_prefix, fn, fn_kwargs, root=None, save_path=None, n_tasks_concurrent=None):
    """
        This runs "run_and_compute" with different simconfigs in parallel
    """
    if n_tasks_concurrent is None:
        n_tasks_concurrent = cpu_count() - 1
    result = []
    for i in range(0, len(simconfigs), n_tasks_concurrent):
        with Pool(n_tasks_concurrent) as p:
            result.extend(p.starmap(run_and_compute, 
                      [(simconfig, seeds, file_prefix, fn, fn_kwargs, root, save_path) 
                      for simconfig in simconfigs[i:i+n_tasks_concurrent]]))
    return result

def run_n_complete(simconfig, seeds, n_minimal_seeds, simu_path):
    seeds_completed = []
    for s,seed in enumerate(seeds):
        finished = run_simulation(simconfig=simconfig, seed=seed, save_path_root=simu_path)
        if finished:
            seeds_completed.append(seed)
            if len(seeds_completed) == n_minimal_seeds:
                break
    if len(seeds_completed) != n_minimal_seeds:
        Printer.print(
                f'Not enough seeds given ({len(seeds)}) to reach {n_minimal_seeds} simulations completed ({len(seeds_completed)}).')
        return False
    return seeds_completed

# TODO : clarify seed/seeds difference for PCI and LZ, currently this is only tested with PCI
def run_and_compute(simconfig, seeds, file_prefix='LZc', fn=parallelized_entropy, fn_kwargs={'method_name':'LZc'},
                    simu_path=None, save_path=None, n_minimal_seeds=None, delete_sim=False):
    """
        This function:
            1) runs a simconfig with all its seeds at {simu_path}/{simconfig_name}
            2) computes some function fn on the result (LempelZiv or PCI for instance)
            3) saves the result at save_path (or by default in root)
            3) deletes the simulation data
            4) returns the result
        The function fn should:
            * have these necessary parameters: simconfig, seeds, simu_path, save_path
            * check if results already exists
            * save file
        Other arguments to fn can be given via fn_kwargs
        Useful if testing a fn on a lot of parameters
        If n_minimal_seeds is set, will run all seeds given in seeds UNTIL n_minimal_seeds simulations are completed
        (Useful when explosions stopping simulations are expected)
    """
    if n_minimal_seeds is None:
        n_minimal_seeds = len(seeds)

    print('Simconfig :\n', simconfig.custom_parameters)
    if simu_path is None:
        default_simu_path,_ = simconfig.get_sim_name(n_minimal_seeds)
        simu_path = default_simu_path
    
    # Check if file to compute already exists
    try:
        result = fn(simconfig=simconfig, seeds=seeds, simu_path=simu_path, save_path=save_path, **fn_kwargs)
        return result
    except FileNotFoundError as e:
        Printer.print(f'File has not been computed yet: {e}')
    
    # Run simulations for the minimal nb of seeds possible
    seeds_completed = run_n_complete(simconfig, seeds, n_minimal_seeds, simu_path)
    
    print('Computing function on simulation data...')
    result = fn(simconfig=simconfig, seeds=seeds_completed, simu_path=simu_path, save_path=save_path, **fn_kwargs)
    print('Computed result:', result)
    
    # Delete simulation data
    if delete_sim:
        Printer.print('Deleting simulation data.')
        for seed in seeds:
            try:
                default_simu_path,_ = simconfig.get_sim_name(seed)
                shutil.rmtree(simu_path)
            except:
                pass
    print('Computation finished.')
    return result


def setup_files(simulator, parameters, seed, initial_condition=None):
    """
        Create parameter.json in the simulation output folder and save initial condition if needed.
    
        :param simulator: Simulator object containing connectivity and history.
        :param parameters: Parameters object with nested dicts for serialization.
        :param seed: Seed integer stored in parameter file as "myseed".
        :param initial_condition: Optional initial condition object to save; if None, simulator history is saved.
        :returns: None
    """
    path_result = parameters.parameter_simulation['path_result']
    param_path = os.path.join(path_result, 'parameter.json')
    if not os.path.exists(param_path):
        os.makedirs(path_result, exist_ok=True)
        f = open(param_path, "w")
        f.write("{\n")
        for name,dic in [('parameter_simulation',parameters.parameter_simulation),
                        ('parameter_model',parameters.parameter_model),
                        ('parameter_connection_between_region',parameters.parameter_connection_between_region),
                        ('parameter_coupling',parameters.parameter_coupling),
                        ('parameter_integrator',parameters.parameter_integrator),
                        ('parameter_monitor',parameters.parameter_monitor),
                        ('parameter_stimulus',parameters.parameter_stimulus)]:
            f.write(f'"{name}" : ')
            try:
                f.write(jsonpickle.encode(dic, unpicklable=True))
                f.write(",\n")
            except TypeError:
                Printer.print(f"{dic} not serialisable", level=2)

        f.write('"myseed":'+str(seed)+"\n}\n")
        f.close()

    if initial_condition is None:
        np.save(os.path.join(path_result, 'step_init.npy'),
                simulator.history.buffer)


def get_n_step_files(path):
    """
        Count saved step files in a directory, excluding step_init.
    
        :param path: Directory path where step_*.npy files are stored.
        :returns: Integer count of step_ files excluding step_init.
    """
    n_step_files = 0
    if not os.path.exists(path):
        return 0
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.startswith('step_') and not 'init' in filename:
            n_step_files += 1
    return n_step_files

# TODO : THERE IS A PROBLEM WITH EITHER BOLD OR SUPER LONG SIMULATIONS IN GENERAL WHERE IT WILL NOT RECOGNIZE RESULTS ALREADY EXIST
def is_sim_enough(simconfig, parameter, path):
    """
        Check whether a saved simulation at path has run for at least the desired duration in simconfig.
    
        :param simconfig: SimConfig containing desired run_sim time.
        :param parameter: Parameter dict/object loaded from parameter.json.
        :param path: Directory containing step_*.npy files.
        :returns: True if saved simulation length >= simconfig.run_sim, otherwise False.
    """
    n_step_files = get_n_step_files(path)
    if n_step_files == 0:
        return False
    # There is at least one step file (step_{n_step_files-1})that is not step_init
    # Done sim time = completed step_ and the last one that's potentially incomplete
    done_sim_time = (n_step_files-1) * parameter['parameter_simulation']['save_time']
    try:
        last_step_data = np.load(os.path.join(path, f'step_{n_step_files-1}.npy'), allow_pickle=True)
        done_sim_time += last_step_data.shape[1] * parameter['parameter_integrator']['dt']
    except:
        pass

    if done_sim_time >= simconfig.run_sim:
        return True
#    for mon,var,min_v,max_v,t in simconfig.stop_conditions:
#        if t > last_step_data[0][0]:
#            continue
#        if (last_step_data[0][mon][var] < min_v).any() or (last_step_data[0][mon][var] > max_v).any():
#            Printer.print(f'Skipping simulation : simulation was stopped because of stop condition ({mon},{var},{min_v},{max_v},{t})', level=2)
#            return True
    return False
     
def skip_sim(simconfig, seed, simu_path):
    """
        Inspect simu_path for an existing simulation folder that matches simconfig parameters and seed.
    
        :param simconfig: SimConfig to compare against.
        :param seed: Seed integer to match with stored "myseed".
        :param simu_path: Root directory to search for simulation subfolders.
        :returns: Tuple (code, folder_name) where code: 0=no folder, 1=exists and sufficient, 2=exists but stop-condition file found.
    """
    if not os.path.exists(simu_path):
        return 0,None
    for dir in os.listdir(simu_path):
        if not os.path.isdir(os.path.join(simu_path, dir)):
            continue
        try:
            with open(os.path.join(simu_path, dir, 'parameter.json'), 'r') as f:
                parameter = jsonpickle.decode(f.read())
            if parameter['myseed'] != seed:
                continue

            if parameter == simconfig.general_parameters:
                Printer.print('Folder with same parameter.json file found.')
                if os.path.exists(os.path.join(simu_path, dir, 'stop_condition_satisfied.txt')):
                    return 2,dir
                if is_sim_enough(simconfig, parameter, os.path.join(simu_path, dir)):
                    Printer.print('Same or superior simulation length.')
                    return 1,dir
        except Exception as e:
            # couldn't open parameter.json file
            Printer.print('Could not open parameter.json file', 2)
            print(e)
            continue
    return 0,None

# TODO : should implement save option and return result option
# TODO: if simulation exists but has lower duration, should continue from where it left off (actually no bcs would need the random state too ! otw not reproducible)
def run_simulation(simconfig, seed=10, print_connectome=False, save_path_root=None, no_skip=False, param_change_with_seed={}):
    """
        Run a single simulation and save its results to disk.
    
        :param simconfig: SimConfig describing the simulation.
        :param seed: Integer seed for the simulation.
        :param print_connectome: If True, plot the connectivity matrix.
        :param save_path_root: Root directory where simulation folder will be created; if None, uses simconfig default.
        :param no_skip: If True, force running even if matching existing results are found.
        :param param_change_with_seed: Optional dict of parameter changes applied before running.
        :returns: True if simulation completed to the requested end time; False on exceptions or early stop.
    """
    
    parameters = simconfig.general_parameters
    cus_parameters = simconfig.custom_parameters
        
    for k,v in param_change_with_seed.items():
        cus_parameters[k] = v
    simconfig._adjust_parameters()
    
    folder_root,sim_name = simconfig.get_sim_name(seed)
    if save_path_root is None:
        save_path_root = folder_root
    save_path = os.path.join(save_path_root, sim_name)
            
        
#    if not os.path.exists(save_path):
#        os.makedirs(save_path)
#    elif not no_skip:
#        Printer.print("Existing folder for simulation: ", simconfig, 'with seed', seed)
#        skip_decision = skip_sim(simconfig, save_path)
#        if skip_decision:
#            Printer.print('Simulation already exists, skipping it.')
#            return True
#        Printer.print('Simulation is not sufficient, overriding.')
#        shutil.rmtree(save_path)
    
    Printer.print("path = ", save_path)
    Printer.print('Initialize Simulator')
    
    parameters.parameter_simulation['path_result'] = save_path
    simulator = sim_init(simconfig, seed=seed)
    
    # CHECK IF SIMULATION ALREADY EXISTS
    skip_code,existing_dir = skip_sim(simconfig, seed, save_path_root)
    if not no_skip and skip_code > 0:
        Printer.print('Existing folder for simulation:', simconfig, 'with seed', seed)
        # Update name of the folder (in case new params_to_report, or order different)
        os.rename(os.path.join(save_path_root, existing_dir), save_path)
        return skip_code == 1
    try:
        shutil.rmtree(save_path)
    except:
        pass
    setup_files(simulator, parameters, seed,)
    
    Printer.print('Start Simulation')
    parameter_simulation,parameter_monitor= parameters.parameter_simulation, parameters.parameter_monitor
    
    time = simconfig.run_sim
    
    if parameters.parameter_stimulus['stimval']:
        Printer.print ('    Stimulating for {1} ms, {2} nS in the {0} at time {3}ms'.format(
                ', '.join([simulator.connectivity.region_labels[r] for r in parameters.parameter_stimulus['stimregion']]),
                parameters.parameter_stimulus['stimdur'],
                parameters.parameter_stimulus['stimval'],
                parameters.parameter_stimulus['stimtime']))
        if [r for r in cus_parameters['stimregion'] if r in parameters.parameter_connection_between_region['disconnect_regions']]:
            Printer.print('/!\ Warning /!\ Stimulating a disconnected region /!\ Warning /!\ ', label=1)

    nb_monitor = parameter_monitor['Raw'] + parameter_monitor['TemporalAverage'] + parameter_monitor['Bold'] + parameter_monitor['Ca']
    if 'Afferent_coupling' in parameter_monitor.keys() and parameter_monitor['Afferent_coupling']:
        nb_monitor+=1
    # initialise the variable for the saving the result
    save_result =[]
    for i in range(nb_monitor):
        save_result.append([])

    if print_connectome:
        plt.figure()
        plt.imshow(simulator.connectivity.weights, interpolation='nearest', cmap='binary')
        plt.colorbar()
        plt.show()
    # run the simulation
    count = 0
    finished = True
    try:
        for result in simulator(simulation_length=time):
            # result : (1,n_monitor+1,n_var), with result[0][0] being time
            # Check conditions
            for mon,var,min_v,max_v,t in simconfig.stop_conditions:
                if t > result[0][0]:
                    continue
                if (result[0][mon][var] < min_v).any() or (result[0][mon][var] > max_v).any():
                    Printer.print(f'Simulation stopped because of stop condition ({mon},{var},{min_v},{max_v},{t})', level=2)
                    with open(os.path.join(save_path, 'stop_condition_satisfied.txt'), 'w') as f:
                        f.write(f'{mon}, {var}, {min_v}, {max_v}, {t}, {result[0][mon][var]}')
                    raise Exception('Stop simulation')
                
            simconfig.apply_varying_params(simulator.model, t=result[0][0])
            for i in range(nb_monitor):
                if result[i] is not None:
                    save_result[i].append(result[i])
            #save the result in file
            if result[0][0] >= parameter_simulation['save_time']*(count+1): #check if the time for saving at some time step
                Printer.print('simulation time :'+str(result[0][0])+'\r')
                np.save(save_path + '/step_'+str(count)+'.npy',np.array(save_result, dtype='object'), allow_pickle = True)
                save_result = []
                for i in range(nb_monitor):
                    save_result.append([])
                count +=1
    except Exception as e:
        Printer.print('Error in run simulation loop', 2)
        Printer.print(e, 2)
        finished = False
        
    # save the last part
    np.save(save_path + '/step_'+str(count)+'.npy',np.array(save_result, dtype='object') , allow_pickle = True)
#    if count < int(time/parameter_simulation['save_time'])+1:
#        np.save(save_path + '/step_'+str(count+1)+'.npy',np.array([], dtype='object'))
        
    # clear_output(wait=True)
    if finished:
        Printer.print(f"Simulation Completed successfully")
    return finished
