from scipy.signal import detrend, hilbert
import numpy as np

from einops import reduce, repeat

def preprocess(data):
    """
        Normalize each region by subtracting its temporal mean and detrend along time.
    
        :param data: Array of shape (time, n_regions).
        :returns: Detrended array of shape (time, n_regions).
    """
    return detrend(data - reduce(data, 't r -> r', np.mean), axis=0)
    
def binarize_signal(data, return_type=int):
    """
        Binarize each region based on the mean amplitude of its Hilbert transform.
    
        :param data: Array of shape (time, n_regions).
        :param return_type: Output dtype (e.g., bool or int).
        :returns: Binarized array of shape (time, n_regions).
    """
    hilbert_data = np.abs(hilbert(data, axis=0))
    thresholds = reduce(hilbert_data, 't r -> r', np.mean)
    return (hilbert_data > thresholds).astype(return_type)

def int_reduction(data):
    """
        Convert each time slice of a binary matrix to an integer via bit-weighted sum.
    
        :param data: Binary array of shape (time, n_regions).
        :returns: 1D array of length time with integer encodings.
    """
    time, n_regions = data.shape
    weights = repeat(2 ** np.arange(n_regions), 'r -> t r', t=time)
    return reduce(data * weights, 't r -> t', np.sum)

def entropy(string):
    """
        Compute Shannon entropy (base 2) of a sequence.
    
        :param string: Iterable of hashable symbols.
        :returns: Entropy in bits.
    """
    s = list(string)
    probs = [s.count(c) / len(s) for c in dict.fromkeys(s)]
    return -sum(p * np.log2(p) for p in probs if p > 0)

def phase_diff(p1, p2, thresh=0.8):
    """
        Compute binary synchrony time series from two phase time series.
    
        :param p1: 1D array of phases (radians).
        :param p2: 1D array of phases (radians).
        :param thresh: Threshold in radians for defining synchrony.
        :returns: Boolean array of shape (time,) indicating synchrony.
    """
    diffs = np.abs(p1 - p2)
    diffs = np.where(diffs > np.pi, 2 * np.pi - diffs, diffs)
    return diffs < thresh