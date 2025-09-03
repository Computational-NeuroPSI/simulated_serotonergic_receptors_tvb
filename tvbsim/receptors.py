import numpy as np

def conversion(E_Na, E_K, E_L, g_L=None, g_Na=None):
    """
        Returns leak conductance values (g_K and g_Na) given leak reversal potentials
        and either g_L or g_Na.
    
        This function applies algebraic rearrangements to ensure current balance
        in a conductance-based model.
    
        :param E_Na: Sodium leak reversal potential.
        :param E_K: Potassium leak reversal potential.
        :param E_L: Leak reversal potential.
        :param g_L: Leak conductance (optional, mutually exclusive with g_Na).
        :param g_Na: Sodium conductance (optional, mutually exclusive with g_L).
        :return: (g_K, g_Na) tuple of potassium and sodium conductances.
    """
    if g_L is not None:
        g_K = g_L * (E_L - E_Na) / (E_K - E_Na)
        g_Na = g_L - g_K
        return g_K, g_Na
    g_L = g_Na * (E_Na - E_K) / (E_L - E_K)
    g_K = g_L - g_Na
    return g_K, g_Na

def get_g_K_values(g_K_max, g_K_min, receptors):
    """
        Computes region-specific leak potassium conductance (g_K) values based on receptor densities.
    
        The function linearly interpolates g_K values between `g_K_max` and `g_K_min`
        according to the density of 5-HT2a receptors provided (or mixture of receptors).
    
        :param g_K_max: Maximum potassium conductance (float).
        :param g_K_min: Minimum potassium conductance (float).
        :param receptors: Array of receptor densities.
        :return: List of interpolated g_K values across regions.
    """
    receptors = receptors.flatten()
    g_Ks = np.interp(x=receptors, xp=[0, np.max(receptors)], fp=[g_K_max, g_K_min])
    return g_Ks.tolist()