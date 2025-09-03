import numpy as np
from abc import ABC
import collections.abc

# class instead of pure functions because need to store some stuff (like previous values), and also it needs to store a conversion to string

# Abstract since a parameter with no other behavior/attributes than name and value does not need a class
class Parameter(ABC):
    """
        Abstract class of a parameter, which has two attributes, a name and a value
        (the name should be one in one of parameter_M_berlin_new's dictionaries)
    """
    def __init__(self, name, value):
        """
            :param name: name of the parameter/key
            :param value: value of the parameter
        """
        self.name = name
        self.value = value
    def get(self, t):
        return self.value
    def __str__(self):
        v = str(self.value) if self.value is not None else ''
        return f'{v}'
    def __eq__(self, other):
        if isinstance(other, Parameter):
            return self.name == other.name and self.value == other.value
        return False

class ListParameter(Parameter):
    """
        Class for list of parameters. Custom name for displaying value can be used here as we wouldn't want
        to have a folder named ..._b_e_[5,5.1,6.3,5.8, etc] for instance
        If parameter name/key is b_e and custom_value_display is set to 'list', it will appear in name of folders as b_e_list
    """
    def __init__(self, name, value, custom_value_display='list'):
        assert(isinstance(value, list)), "If value is not a sequence, use an int/float/child class of Parameter instead of ParameterList. Also, give a list: np arrays have trouble serializing."
        super().__init__(name, value)
        self.custom_value_display = custom_value_display
        # varying parameters : varying through time
        self.varying_parameters_idxs = [i for i,v in enumerate(self.value) if issubclass(type(v), Parameter)]
    def __str__(self):
        return self.custom_value_display
    def __round__(self):
        """
            Round called to display shorter values, in this case the display of the value doesn't change
        """
        return self.custom_value_display
    def reset(self):
        for i in self.varying_parameters_idxs:
            self.value[i].reset()
    def get(self, t):
        v = self.value[:] # mixing floats and Parameter (create copy of list) (# TODO: very slow, copy every step)
        # Remove Parameter :
        for i in self.varying_parameters_idxs:
            v[i] = v[i].get(t=t)
        return np.array(v)[:, np.newaxis]
    def __getitem__(self, idx):
        return self.value[idx]
    def __list__(self):
        return self.value
    
# this is probably useless
#class ListParameterUniqueValues(ListParameter):
#    def __init__(self, name, value):
#        super().__init__(name, value)
#        self.unique_values = []
#        for v in self.value:
#            if v not in self.unique_values:
#                self.unique_values.append(v)
#    def __str__(self):
#        return '_'.join(list(map(str, self.unique_values)))

class VaryingParameter(Parameter, ABC):
    """
        Abstract class for parameters that vary though time
    """
    def __init__(self, name, value=None, log_values=False):
        """
            :param log_values: whether to record every values that the parameter has outputed in the simulation
        """
        super().__init__(name, value)
        self.log_values = log_values
        if self.log_values:
            self.prev_values = []
    def get(self, t):
        """
            Returns the parameter value given time t and records value if log_values true
        """
        v = self._compute_value(t)
        if self.log_values:
            self.prev_values.append(v)
        return v
    def _compute_value(self, t):
        pass
    def reset(self):
        pass
    def __str__(self):
        # returns the name of the function (for sim_name)
        return super().__str__()

class PureFunctionParameter(VaryingParameter):
    """
        Simple VaryingParameter where a function is called every timestep to output a parameter
    """
    def __init__(self, name, log_values, func):
        """
            :param func: function that takes as input PureFunctionParameter and time t
        """
        super().__init__(name, log_values=log_values)
        self._compute_value = func
    
class PeriodicallyVaryingParameter(VaryingParameter, ABC):
    """
        For VaryingParameters where the value is only changed every x ms
    """
    def __init__(self, name, period, log_values=False):
        super().__init__(name, log_values=log_values)
        self.period = period
        self.count = 0
    @staticmethod
    def periodic(func):
        def wrapper(self, *args, **kwargs):
            if kwargs['t'] >= (self.count + 1) * self.period or not hasattr(self, 'cur_value'):
                self.count += 1
                self.cur_value = func(self, *args, **kwargs)
                if self.log_values:
                    self.prev_values.append(self.cur_value)
            return self.cur_value
        return wrapper
    def __str__(self):
        return super().__str__() + f'period_{self.period}_'
    def reset(self):
        super().reset()
        self.count = 0
#    def __eq__(self, other):
#        if not isinstance(other, PeriodicallyVaryingParameter):
#            return False
#        return self.period == other.period

class GaussianParameter(PeriodicallyVaryingParameter):
    """
        Parameter sampled from gaussian distribution every x ms
        If force_positivity is on, it is not really Gaussian anymore, right-hand tail is bigger because density in the negative is reported to the positives
    """
    def __init__(self, name, period, mean, std, force_non_negative=False, log_values=False):
        super().__init__(name, period, log_values=log_values)
        self.mean = mean
        self.std = std
        self.force_non_negative = force_non_negative
    @PeriodicallyVaryingParameter.periodic
    def get(self, t):
        v = np.random.normal(loc=self.mean, scale=self.std)
        return np.abs(v) if self.force_non_negative else v
    def __str__(self):
        return super().__str__() + f'gaussian_{self.mean}_{self.std}'
    def __eq__(self, other):
        if not isinstance(other, GaussianParameter):
            return False
        return (self.mean, self.std, self.period) == (other.mean, other.std, other.period)
    
class UniformParameter(PeriodicallyVaryingParameter):
    """
        Parameter sampled from uniform distribution every x ms
    """
    def __init__(self, name, period, min_, max_, log_values=False):
        super().__init__(name, period, log_values=log_values)
        self.min = min_
        self.max = max_
    @PeriodicallyVaryingParameter.periodic
    def get(self, t):
        v = np.random.uniform(self.min, self.max)
        return v
    def __str__(self):
        return super().__str__() + f'uniform_{self.min}_{self.max}'
    def __eq__(self, other):
        if not isinstance(other, UniformParameter):
            return False
        return (self.period, self.min, self.max) == (other.period, other.min, other.max)
    
class OrnsteinUhlenbeckParameter(VaryingParameter):
    """
        Parameter follows an Ornstein-Uhlenbeck random process
    """
    def __init__(self, name, speed, mean, volatility, dt, x0=None, force_non_negative=False):
        super().__init__(name)
        self.speed = speed
        self.mean = mean
        self.volatility = volatility
        self.x0 = x0
        if self.x0 is None:
            self.x0 = self.mean
        self.x = self.x0
        self.dt = dt
        self.force_non_negative = force_non_negative
    def get(self, t):
        eps = np.random.normal(0,1)
        new_x = self.x + self.speed * (self.mean - self.x) * self.dt + self.volatility * np.sqrt(self.dt) * eps
        self.x = max(new_x, 0.) if self.force_non_negative else new_x
        return self.x
    def reset(self):
        super().reset()
        self.x = self.x0
    def __str__(self):
        return super().__str__() + f'ornuhl_{self.speed}_{self.mean}_{self.volatility}_{self.x0}'
    def __eq__(self, other):
        if not isinstance(other, OrnsteinUhlenbeckParameter):
            return False
        return (self.speed, self.mean, self.volatility, self.x0, self.dt) == (other.speed, other.mean, other.volatility, other.x0, other.dt)  
    



