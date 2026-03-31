import numpy as np
from scipy import signal, optimize
import libys.plot as plt

from numpy import pi, exp, conj


def get_mode(
    waveform: np.ndarray,  # shape = (n_samples)
    t0: int,
    len_mode: int,
    f_sideband: float,
    filter_order: int,
    filter_cutoff: float,
    sampling_int: float,
    rotate: bool = False,
    plot: bool = False,
) -> np.ndarray:  # shape = (len_mode), dtype = complex

    waveform0 = np.zeros(3*len_mode)
    waveform0[len_mode:2*len_mode] = waveform[t0:t0+len_mode]
    t = np.arange(-len_mode, 2*len_mode) * sampling_int
    unfiltered = waveform0 * exp(2j*pi*f_sideband*t)
    b, a = signal.butter(filter_order, filter_cutoff, fs=1/sampling_int)
    filtered = signal.filtfilt(b, a, unfiltered)
    mode = 2 * filtered[len_mode:2*len_mode]

    if rotate:
        def imag_power(phase):
            rotated_mode = mode * exp(-1j*phase)
            return sum(rotated_mode.imag**2)

        phase = optimize.minimize_scalar(imag_power).x
        mode *= exp(-1j*phase)

        if mode.real.mean() < 0:
            mode = -mode

    if plot:
        plt.semilogy(*signal.periodogram(unfiltered, fs=1/sampling_int, return_onesided=False), '.')
        plt.semilogy(*signal.periodogram(filtered, fs=1/sampling_int, return_onesided=False), '.')

    return mode


def get_iq(
    waveforms: np.ndarray,  # shape = (n_shots, n_samples)
    t0s: np.ndarray,
    mode: np.ndarray,  # dtype = complex
    f_sideband: float,
    sampling_int: float,
) -> np.ndarray:  # shape = (len(t0s), n_shots), dtype = complex

    n_shots, n_samples = waveforms.shape
    iqs = np.empty((len(t0s), n_shots), dtype=complex)

    for i, t0 in enumerate(t0s):
        v = waveforms[:, t0:t0+len(mode)]
        t = np.arange(len(mode)) * sampling_int
        iqs[i, :] = np.inner(conj(mode)*exp(2j*pi*f_sideband*t), v) / np.inner(conj(mode), mode)

    return iqs


class PulseMode:

    def __init__(
        self,
        mode: np.ndarray,
        f_sideband: float,
        sampling_int: float,
    ):
        self.mode = mode
        self.f_sideband = f_sideband
        self.sampling_int = sampling_int

    def get_iq(self, waveforms, t0s):
        return get_iq(waveforms, t0s, self.mode, self.f_sideband, self.sampling_int)
