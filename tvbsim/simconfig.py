import copy
import collections.abc
import numpy as np
import tvb.simulator.lab as lab
import numpy.random as rgn
import jsonpickle
import os
import builtins 

from tvbsim.common import print_dict_differences
from tvbsim.printer import Printer
from tvbsim.parameter import Parameter, ListParameter
from tvbsim.TVB.tvb_model_reference.simulation_file.parameter.parameter_M_Berlin_new import Parameters

class SimConfig:
    """
    Interface to modify a subset of parameters from the default file located at:
    ./TVB/tvb_model_reference/simulation_file/parameter/parameter_M_Berlin_new.py 
    and choose which parameters are reported in file names, plot titles etc
    """
    def __init__(self, general_parameters, run_sim, cut_transient, parameters=None, params_to_report=None,
                 stop_conditions=[], auto_report=False):
        """
            :param general_parameters: instance of Parameters from parameter_M_Berlin_new, which has the default parameters.
                                       A COPY OF IT IS CREATED FOR EACH SIMCONFIG OBJECT
            :param run_sim: duration of simulation in ms
            :param cut_transient: how much time should be cut from the results at the beginning of the simulation
                                  (data is still recorded so this parameter is only used by functions using the results of simulations)
            :param parameters: dictionary of params to change value compared to default
                Valid parameters :
                    - Any parameters from parameter_M_Berlin_new (no need to specify the dictionary where the parameter is defined)
                    - In particular for stimulation :
                        - stimregions (LIST)
                        - stimdur
                        - stimperiod
                        - stimtime (= onset)
                        ...
            :param params_to_report: names of the params that should appear in file names, plot titles etc
                                     NOTE : stuff that should be reported but are not parameters can be added by including 
                                            the name in params_to_report, and name:value in the parameters dictionary
            :param auto_report: add to params_to_report the params set in parameters (if they are not already in params_to_report)
            :param stop_conditions: stop simulations in case some var goes over/under thresholds (in KHz) after time t
                                    list of tuples : (monitor_id, var_id, min_value, max_value, t) (None if not specified)
                                    (for instance to stop when excitatory explosion : (1,0,None,150*1e-3,2000))
        """
        self.run_sim = run_sim
        self.cut_transient = cut_transient
        self.general_parameters = copy.deepcopy(general_parameters) # each have own parameter object in case of future parallel computing
        self.custom_parameters = parameters
        self.stop_conditions = stop_conditions
        # Parameter dictionaries from Parameters class
        parameter_dicts = [v for k,v in vars(self.general_parameters).items() if k.startswith('parameter')]

        for name,value in self.custom_parameters.items():
            # check if parameter exists in Parameters
            param_exists = False
            for parameter_dict in parameter_dicts:
                if name in parameter_dict:
                    param_exists=True
                    break
            if not param_exists:
                raise Exception(f'Parameter given with name {name} does not exist.')
            
            # parameters given as a list should be translated in ListParameter
            if not isinstance(value, str) and isinstance(value, collections.abc.Iterable):
                if isinstance(value, dict):
                    # not a parameter model
                    continue
                self.custom_parameters[name] = ListParameter(name, value, 'list')
                continue
            
            # numpy float cause serialization errors for some reason, so conversion here
            if isinstance(value, np.floating):
                Printer.print(f"Parameter {name}'s value {value} has type {type(value)} which creates serialization issues. Casting it to float.")
                self.custom_parameters[name] = value.item()
        # TODO: only time varying parameters
        self.special_parameters = [n for n,v in self.custom_parameters.items() if issubclass(type(v), Parameter)]
        
        self.params_to_report = params_to_report
        if auto_report:
            if self.params_to_report is None:
                self.params_to_report = []
            self.params_to_report.extend(parameters.keys())
            self.params_to_report = list(dict.fromkeys(self.params_to_report))
        # self.params_to_report.sort() # universal order of defining params
        
        # Also add to customized_parameters params_to_report's default values (clumsy, TODO)
        for name in self.params_to_report:
            if name not in self.custom_parameters: # otw only default values
                for parameter_dict in parameter_dicts:
                    try:
                        self.custom_parameters[name] = parameter_dict[name]
                        break
                    except Exception:
                        pass
        self._adjust_parameters()
    
    def get_sim_name(self, seed=''):
        """
            Returns the name and default folder of the simulation given the parameters to report (in self) and a seed
            This is used to name saved files for instance
        """
        sim_name = ''
        if self.custom_parameters.get('stimval', False):
            folder_root = './result/evoked'
            sim_name +=  f"stim_{self.custom_parameters['stimval']}_"
        else:
            folder_root = './result/synch'
        if self.custom_parameters.get('disconnect_regions',[]):
            sim_name += f"dr_{len(self.custom_parameters['disconnect_regions'])}_r_{self.custom_parameters['disconnect_regions'][0]}_" 
        sim_name += '_'.join([f'{name}_{str(self.custom_parameters[name])}' for name in self.params_to_report]) + '_' + str(seed)
        return folder_root, sim_name
    
    def get_plot_title(self):
        return ', '.join([f'{name} = {str(round(self.custom_parameters[name],2)) if isinstance(self.custom_parameters[name],float) else self.custom_parameters[name]}' for name in self.params_to_report])

    def _adjust_parameters(self):
        """
            Assigns the custom parameters chosen by user (held in self.parameters) to the (copied) default parameters (held in self.general_parameters)
        """
        # Find where each customized parameter can be set to the main parameter 
        parameter_dicts = [v for k,v in vars(self.general_parameters).items() if k.startswith('parameter')]
        # Add dictionaries that are values of parameters dictionaries (example : coupling_parameter)
        sub_parameter_dicts = []
        for parameter_dict in parameter_dicts:
            for param_name,param_value in parameter_dict.items():
                if isinstance(param_value, dict):
                    sub_parameter_dicts.append(param_value)
        parameter_dicts += sub_parameter_dicts
        for name, value in self.custom_parameters.items():
            for parameter_dict in parameter_dicts:
                if name in parameter_dict:
                    parameter_dict[name] = value
                    
    def __str__(self):
        s = f'run_sim = {self.run_sim}, cut_transient = {self.cut_transient}\n'
        s += f'custom parameters = {self.custom_parameters}\n'
        s += f'special parameters = {self.special_parameters}'
        return s
    
    
    """
        Functions used for parameters that change through time
    """
    def reset_varying_params(self):
        """
            Called at the end of running a simulation to reset parameters that change through time
        """
        for name in self.special_parameters:
            self.custom_parameters[name].reset()
    
    def apply_varying_params(self, model, t):
        """
            Called every step of a simulation ; applies updated parameters to the model
        """
        for name in self.special_parameters:
            setattr(model, name, np.array(self.custom_parameters[name].get(t=t)))
            
# TODO : write in a file parameters for each region, rather than having a super long folder name
# But then how to distinguish one folder from another ? Check manually the json parameter file ?

# TODO : special treatment of g_K g_Na etc ? Specify E_L_e, E_L_i (and g_L = 10) and they are automatically
#        determined ? easier for naming files + no round values in sim. Do this in sim_init if necessary
# TODO : should not be possible to set initial_conditions here I think, should remove it
def sim_init(simconfig, initial_condition=None, seed=10):
    """
        This uses a simconfig to produce a simulator with the parameters described in the simconfig's Parameter (parameter_M_berlin)
    
        :param simconfig: parameters for the simulation
        :param initial_condition: initial_condi
        :param seed: seed of the simulator
        
        :return: the simulator initialized
    """

    parameters = simconfig.general_parameters

    parameter_simulation  = parameters.parameter_simulation
    parameter_model = parameters.parameter_model
    parameter_connection_between_region = parameters.parameter_connection_between_region
    parameter_coupling = parameters.parameter_coupling
    parameter_integrator = parameters.parameter_integrator
    parameter_monitor = parameters.parameter_monitor
    parameter_stimulation = parameters.parameter_stimulus
    ## initialise the random generator
    parameter_simulation['seed'] = seed
    Printer.print(f"Setting seed to {parameter_simulation['seed']}")
    rgn.seed(parameter_simulation['seed'])

    if parameter_model['matteo']:
        if parameter_model['gK_gNa']:
            from .TVB.tvb_model_reference.src import Zerlaut_matteo_gK_gNa as model
        else:
            from .TVB.tvb_model_reference.src import Zerlaut_matteo as model
    else:
        if parameter_model['gK_gNa']:
            from .TVB.tvb_model_reference.src import Zerlaut_gK_gNa as model
        else:
            from .TVB.tvb_model_reference.src import Zerlaut as model

    ## Model
    if parameter_model['order'] == 1:
        model = model.Zerlaut_adaptation_first_order(variables_of_interest='E I W_e W_i noise'.split())
    elif parameter_model['order'] == 2:
        model = model.Zerlaut_adaptation_second_order(variables_of_interest='E I C_ee C_ei C_ii W_e W_i noise'.split())
    else:
        raise Exception('Bad order for the model')
    # ------- Changed by Maria 
    to_skip=['initial_condition', 'matteo', 'order', 'gK_gNa']
    for key, value in parameters.parameter_model.items():
        if key not in to_skip:
            if key not in simconfig.special_parameters:
                try:
                    setattr(model, key, np.array(value))
                    continue
                except:
                    try:            
                        value.reset()
                    except:
                        pass
                    setattr(model, key, np.array(value.get(t=0)))
            # Treat the case of varying parameters, which are not float and cannot be assigned directly
        
    for key,val in parameters.parameter_model['initial_condition'].items():
        model.state_variable_range[key] = val
    

    ## Connection
    if parameter_connection_between_region['default']:
        connection = lab.connectivity.Connectivity().from_file()
    elif parameter_connection_between_region['from_file']:
        path = parameter_connection_between_region['path']
        conn_name = parameter_connection_between_region['conn_name']
        connection = lab.connectivity.Connectivity().from_file(path+'/' + conn_name)
    elif parameter_connection_between_region['from_h5']:
        connection = lab.connectivity.Connectivity().from_file(parameter_connection_between_region['path'])
    elif parameter_connection_between_region['from_folder']:
        # mandatory file 
        tract_lengths = np.loadtxt(parameter_connection_between_region['path']+'/tract_lengths.txt')
        weights = np.loadtxt(parameter_connection_between_region['path']+'/weights.txt')
        # optional file
        if os.path.exists(parameter_connection_between_region['path']+'/region_labels.txt'):
            region_labels = np.loadtxt(parameter_connection_between_region['path']+'/region_labels.txt', dtype=str)
        else:
            region_labels = np.array([], dtype=np.dtype('<U128'))
        if os.path.exists(parameter_connection_between_region['path']+'/centres.txt'):
            centers = np.loadtxt(parameter_connection_between_region['path']+'/centres.txt')
        else:
            centers = np.array([])
        if os.path.exists(parameter_connection_between_region['path']+'/cortical.txt'):
            cortical = np.array(np.loadtxt(parameter_connection_between_region['path']+'/cortical.txt'),dtype=np.bool)
        else:
            cortical=None
        connection = lab.connectivity.Connectivity(
                                                   tract_lengths=tract_lengths,
                                                   weights=weights,
                                                   region_labels=region_labels,
                                                   centres=centers.T,
                                                   cortical=cortical)
    else:
        connection = lab.connectivity.Connectivity(
                                                number_of_regions=parameter_connection_between_region['number_of_regions'],
                                               tract_lengths=np.array(parameter_connection_between_region['tract_lengths']),
                                               weights=np.array(parameter_connection_between_region['weights']),
            region_labels=np.arange(0, parameter_connection_between_region['number_of_regions'], 1, dtype='U128'),#TODO need to replace by parameter
            centres=np.arange(0, parameter_connection_between_region['number_of_regions'], 1),#TODO need to replace by parameter
        )



    if parameter_connection_between_region['nullify_diagonals']:
        connection.weights[np.diag_indices(len(connection.weights))] = 0.

    if 'normalised' in parameter_connection_between_region.keys() and parameter_connection_between_region['normalised']:
        connection.weights = connection.weights/(np.sum(connection.weights,axis=0)+1e-12)
        Printer.print('Weights shape:', connection.weights.shape)

    connection.speed = np.array(parameter_connection_between_region['speed'])


    if parameter_connection_between_region['disconnect_regions']:
        connection.weights[parameter_connection_between_region['disconnect_regions']] = 0.
        connection.weights[:, parameter_connection_between_region['disconnect_regions']] = 0.


    ## Stimulus: added by TA and Jen
    if parameter_stimulation['stimval'] == 0.:
        stimulation = None
    else:
        eqn_t = lab.equations.PulseTrain()
        eqn_t.parameters["onset"] = np.array(parameter_stimulation["stimtime"]) # ms
        eqn_t.parameters["tau"]   = np.array(parameter_stimulation["stimdur"]) # ms
        eqn_t.parameters["T"]     = np.array(parameter_stimulation["stimperiod"]) # ms; # 0.02kHz repetition frequency
        weights = np.zeros(len(connection.weights))
        weights[list(parameter_stimulation['stimregion'])] = parameter_stimulation['stimval']
        stimulation = lab.patterns.StimuliRegion(temporal=eqn_t,
                                          connectivity=connection,
                                          weight=weights)
        model.stvar = parameter_stimulation['stimvariables']

    ## Coupling
    # error if unnecessary arguments in parameter_coupling
    try:
        coupling = getattr(lab.coupling, parameter_coupling['type'])(**{k:np.array(v) for k,v in parameter_coupling['coupling_parameter'].items()})
    except Exception as e:
        raise Exception(f'Coupling type not supported or not adequate parameters given (only specify the parameters used by the coupling type). Original error: {e}')
        
    ## Integrator
    if not parameter_integrator['stochastic']:
        if parameter_integrator['type'] == 'Heun':
            integrator = lab.integrators.HeunDeterministic(dt=np.array(parameter_integrator['dt']))
        elif parameter_integrator['type'] == 'Euler':
             integrator = lab.integrators.EulerDeterministic(dt=np.array(parameter_integrator['dt']))
        else:
            raise Exception('Bad type for the integrator')
    else:
        if parameter_integrator['noise_type'] == 'Additive':
            noise = lab.noise.Additive(nsig=np.array(parameter_integrator['noise_parameter']['nsig']),
                                        ntau=parameter_integrator['noise_parameter']['ntau'],)
            # print("type of noise: ", type(noise), "\nand noise: ", noise)
            
        else:
            raise Exception('Bad type for the noise')
        noise.random_stream.seed(parameter_simulation['seed'])

        if parameter_integrator['type'] == 'Heun':
            integrator = lab.integrators.HeunStochastic(noise=noise,dt=parameter_integrator['dt'])
        elif parameter_integrator['type'] == 'Euler':
             integrator = lab.integrators.EulerStochastic(noise=noise,dt=parameter_integrator['dt'])
        else:
            raise Exception('Bad type for the integrator')

    ## Monitors
    monitors =[]
    if parameter_monitor['Raw']:
        monitors.append(lab.monitors.RawVoi(
                variables_of_interest=np.array(parameter_monitor['parameter_Raw']['variables_of_interest'])))
    if parameter_monitor['TemporalAverage']:
        monitor_TAVG = lab.monitors.TemporalAverage(
            variables_of_interest=np.array(parameter_monitor['parameter_TemporalAverage']['variables_of_interest']),
            period=parameter_monitor['parameter_TemporalAverage']['period'])
        monitors.append(monitor_TAVG)
    if parameter_monitor['Bold']:
        monitor_Bold = lab.monitors.Bold(
            variables_of_interest=np.array(parameter_monitor['parameter_Bold']['variables_of_interest']),
            period=parameter_monitor['parameter_Bold']['period'])
        monitors.append(monitor_Bold)
    if 'Afferent_coupling' in parameter_monitor.keys() and parameter_monitor['Afferent_coupling']:
        monitor_Afferent_coupling = lab.monitors.AfferentCoupling(variables_of_interest=None)
        monitors.append(monitor_Afferent_coupling)
    if parameter_monitor['Ca']:
        monitor_Ca = lab.monitors.Ca(
            variables_of_interest=np.array(parameter_monitor['parameter_Ca']['variables_of_interest']),
            tau_rise=parameter_monitor['parameter_Ca']['tau_rise'],
            tau_decay=parameter_monitor['parameter_Ca']['tau_decay'])
        monitors.append(monitor_Ca)

    #initialize the simulator: edited by TA and Jen, added stimulation argument, try removing surface
    if initial_condition is None:
        simulator = lab.simulator.Simulator(model = model, connectivity = connection,
                          coupling = coupling, integrator = integrator, monitors = monitors,
                                            stimulus=stimulation)
    else:
        simulator = lab.simulator.Simulator(model = model, connectivity = connection,
                                            coupling = coupling, integrator = integrator,
                                            monitors = monitors, initial_conditions=initial_condition,
                                            stimulus=stimulation)
    simulator.configure()

    return simulator