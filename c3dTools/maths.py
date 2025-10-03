from math import gcd
from scipy.signal import butter, filtfilt, resample_poly
import numpy as np

def matlab_resample(x, resample_rate, orig_sample_rate):
    """
    Resample a signal by a rational factor (p/q) to match MATLAB's `resample` function.
    copied from here : https://franciscormendes.github.io/2024/12/17/matching-matlabs-resample/

    Parameters:
        x (array-like): Input signal.
        p (int): Upsampling factor.
        q (int): Downsampling factor.

    Returns:
        array-like: Resampled signal.
    """
    p = resample_rate
    q = orig_sample_rate
    factor_gcd = gcd(int(p), int(q))
    p = int(p // factor_gcd)
    q = int(q // factor_gcd)

    # Ensure input is a numpy array
    x = np.asarray(x)

    # Use resample_poly to perform efficient polyphase filtering
    y = resample_poly(x, p, q, window=('kaiser', 1.0))

    # Match MATLAB's output length behavior
    output_length = int(np.ceil(len(x) * p / q))
    y = y[:output_length]

    return y

def filter_data(data, f, f_mocap):
    Wn = f / (f_mocap / 2)  # Normalized cut-off frequency
    b, a = butter(4, Wn, 'low', analog=False)  # 4th order Butterworth filter

    for i in range(data.shape[1]):
        init = data[0, i]
        data_inter = data[:, i] - init
        data[:, i] = filtfilt(b, a, data_inter)
    return data

def rotation_matrix(seq : str, q:list[float]):
    def rx(q):
        return np.array([[1, 0, 0],
                       [0, np.cos(np.radians(q)), -np.sin(np.radians(q))],
                       [0, np.sin(np.radians(q)), np.cos(np.radians(q))]])

    def ry(q):
        return np.array([[np.cos(np.radians(q)), 0, np.sin(np.radians(q))],
                       [0, 1, 0],
                       [-np.sin(np.radians(q)), 0, np.cos(np.radians(q))]])

    def rz(q):
        return np.array([[np.cos(np.radians(q)), -np.sin(np.radians(q)), 0],
                       [np.sin(np.radians(q)), np.cos(np.radians(q)), 0],
                       [0, 0, 1]])

    r = np.identity(3)
    for i, ax in enumerate(seq):
        if ax == 'x':
            r = r@rx(q[i])
        elif ax == 'y':
            r = r @ ry(q[i])
        elif ax == 'z':
            r = r @ rz(q[i])
    return r