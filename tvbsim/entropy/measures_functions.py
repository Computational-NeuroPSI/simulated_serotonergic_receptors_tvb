import random
import numpy as np
from scipy.signal import hilbert
from einops import rearrange
from lempel_ziv_complexity import lempel_ziv_complexity

from tvbsim.entropy.utils import preprocess, phase_diff, int_reduction, entropy, binarize_signal

def calculate_SCE(data):
    """
        Compute Synchrony Coalition Entropy (SCE) normalized by a random baseline.
    
        :param data: Array of shape (time, n_regions).
        :returns: Scalar SCE value.
    """
    np.random.seed(0)
    time,n_regions = data.shape
    data = preprocess(data)
    angles = np.angle(hilbert(data, axis=0))
    # TODO: maybe this could be vectorized
    synchronies = np.zeros((n_regions, time, n_regions-1), dtype=bool)
    for r1 in range(n_regions):
        l = 0
        for r2 in range(n_regions):
            if r1 != r2:
                synchronies[r1,:,l] = phase_diff(angles[:,r1],angles[:,r2])
                l += 1
    ce = np.zeros(n_regions)
    for r in range(n_regions):
        ce[r] = entropy(int_reduction(synchronies[r]))
    
    norm = entropy(int_reduction(
            np.random.randint(2, size=(time, n_regions))))
    return np.mean(ce)/norm
    
def calculate_ACE(data):
    """
        Compute Amplitude Coalition Entropy (ACE) normalized by a shuffled baseline.
    
        :param data: Array of shape (time, n_regions).
        :returns: Scalar ACE value.
    """
    np.random.seed(0)
    time,n_regions = data.shape
    data = preprocess(data)
    binarized_data = binarize_signal(data)
    data_entropy = entropy(int_reduction(binarized_data))
    for i in range(n_regions):
        random.shuffle(binarized_data[:,i])
    norm_entropy = entropy(int_reduction(binarized_data))
    return data_entropy / norm_entropy

def array2str(data):
    return ''.join(map(str, data.tolist()))

def calculate_LempelZiv(data):
    """
    Compute multidimensional Lempel–Ziv complexity (LZc) normalized by a shuffled baseline.

    :param data: Array of shape (time, n_regions).
    :returns: Scalar normalized LZc value.
    """
    np.random.seed(0)
    data = preprocess(data)
    bin_data = rearrange(binarize_signal(data), 't r -> (t r)').astype(int)
    s = array2str(bin_data)
    lzc = lempel_ziv_complexity(s)
    np.random.shuffle(bin_data)
    s_shuf = array2str(bin_data)
    lzc_shuf = lempel_ziv_complexity(s_shuf)
    return lzc / lzc_shuf

def calculate_LempelZiv_single(data):
    """
        Compute the average single-channel Lempel–Ziv complexity across regions.
    
        :param data: Array of shape (time, n_regions).
        :returns: Scalar mean LZc across regions.
    """
    return np.mean(np.array([calculate_LempelZiv(data[:,i:i+1]) for i in range(data.shape[1])]))