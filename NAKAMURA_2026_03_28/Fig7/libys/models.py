import math
import numpy as np
import lmfit as lmf

from numpy import exp, log, sqrt, pi, sinh, cosh
from scipy.signal import find_peaks, periodogram
from scipy.constants import h
from lmfit.models import LorentzianModel
from lmfit.lineshapes import lorentzian


def decay(x, decay, amplitude, offset):
    return amplitude * exp(-x / decay) + offset


class Decay(lmf.Model):

    def __init__(self):
        super().__init__(decay)

    def guess(self, y: np.ndarray, x: np.ndarray) -> lmf.Parameters:
        if y[0] > y[-1]:
            offset = min(y)
            sign = 1
        else:
            offset = max(y)
            sign = -1

        x = x[y != offset]
        y = y[y != offset]
        slope, intercept = np.polyfit(x, log(sign * (y - offset)), deg=1)

        return self.make_params(
            decay=-1 / slope,
            amplitude=sign * exp(intercept),
            offset=offset,
        )


def cos(x, period, lo, hi, shift):
    amplitude = (hi - lo)/2
    offset = (lo + hi)/2
    return amplitude * np.cos(2*pi * (x - shift) / period) + offset


class Cos(lmf.Model):

    def __init__(self):
        super().__init__(cos)

    def guess(self, y: np.ndarray, x: np.ndarray) -> lmf.Parameters:
        dx = x[1] - x[0]
        f, psd = periodogram(y, fs=1 / dx)
        period = 1 / f[psd.argmax()]

        return self.make_params(
            period=period,
            lo=min(y),
            hi=max(y),
            shift=x[y.argmax()],
        )


def cos_decay(x, decay, period, amplitude, shift, offset):
    exp_decay = amplitude * exp(-x / decay)
    cosine = np.cos(2*pi * (x - shift) / period)
    return exp_decay * cosine + offset


class CosDecay(lmf.Model):

    def __init__(self):
        super().__init__(cos_decay)

    def guess(self, y: np.ndarray, x: np.ndarray) -> lmf.Parameters:
        dx = x[1] - x[0]
        f, psd = periodogram(y, fs=1 / dx)
        period = 1 / f[psd.argmax()]
        peaks = find_peaks(y, distance=math.ceil(period / dx / 2))[0]

        return self.make_params(
            decay=x[-1] - x[0],
            period=period,
            amplitude=(max(y) - min(y)) / 2,
            shift=x[peaks[0]],
            offset=y.mean(),
        )


def lorentzian_peak(x, amplitude, center, sigma, offset):
    return lorentzian(x, amplitude, center, sigma) + offset


class LorentzianPeak(lmf.Model):

    def __init__(self):
        super().__init__(lorentzian_peak)

    def guess(self, y: np.ndarray, x: np.ndarray) -> lmf.Parameters:
        y0 = y - y.mean()
        negative = max(y0) < -min(y0)
        lorentzian_guess = LorentzianModel().guess(y, x, negative).valuesdict()

        return self.make_params(
            amplitude=lorentzian_guess['amplitude'],
            center=lorentzian_guess['center'],
            sigma=lorentzian_guess['sigma'],
            offset=min(y),
        )


def resonator_s11(f, f_r, k_ex_2pi, k_in_2pi, magnitude, slope, offset):
    s11 = k_ex_2pi / ((k_ex_2pi + k_in_2pi)/2 + 1j*(f - f_r)) - 1
    return magnitude * exp(1j*(slope * (f - f_r) + offset)) * s11


class ResonatorS11(lmf.Model):

    def __init__(self):
        super().__init__(resonator_s11)

    def guess(self, data: np.ndarray, f: np.ndarray) -> lmf.Parameters:
        limit = (data[0] + data[-1])/2
        magnitude = abs(limit)
        distance = abs(data - limit)
        diameter = max(distance)
        f_dip = f[np.nonzero(distance > diameter/sqrt(2))[0]]

        if len(f_dip) > 1:
            k = abs(f_dip[-1] - f_dip[0])
        else:
            k = abs(f[1] - f[0])

        dk = (diameter - magnitude) / magnitude * k

        return self.make_params(
            f_r=f[np.argmax(distance)],
            k_ex_2pi=(k + dk)/2,
            k_in_2pi=(k - dk)/2,
            magnitude=magnitude,
            slope=0,
            offset=np.angle(limit) + pi,
        )


def resonator_s11_ge(f, f_g, chi_pi, k_ex_2pi, k_in_2pi, ratio_e, magnitude, slope, offset):
    f_e = f_g - chi_pi
    g = resonator_s11(f, f_g, k_ex_2pi, k_in_2pi, magnitude, slope, offset)
    e = resonator_s11(f, f_e, k_ex_2pi, k_in_2pi, magnitude, slope, offset)
    return (1 - ratio_e) * g + ratio_e * e


class ResonatorS11Ge(lmf.Model):

    def __init__(self):
        super().__init__(resonator_s11_ge)

    def guess(self, data: np.ndarray, f: np.ndarray, chi_pi: float,
              ratio_e: float = 0.1) -> lmf.Parameters:
        limit = (data[0] + data[-1])/2
        magnitude = abs(limit)
        distance = abs(data - limit)
        diameter = max(distance)
        f_dip = f[np.nonzero(distance > diameter/sqrt(2))[0]]
        k = abs(f_dip[-1] - f_dip[0])
        dk = (diameter - magnitude) / magnitude * k

        return self.make_params(
            f_g=f[np.argmax(distance)],
            chi_pi=chi_pi,
            k_ex_2pi=(k + dk)/2,
            k_in_2pi=(k - dk)/2,
            ratio_e=ratio_e,
            magnitude=magnitude,
            slope=0,
            offset=np.angle(limit) + pi,
        )


def resonator_s11_gef(f, f_g, chi_pi, chi_ef_pi, k_ex_2pi, k_in_2pi, p_e, p_f, magnitude, slope, offset):
    f_e = f_g - chi_pi
    f_f = f_e - chi_ef_pi
    g = resonator_s11(f, f_g, k_ex_2pi, k_in_2pi, magnitude, slope, offset)
    e = resonator_s11(f, f_e, k_ex_2pi, k_in_2pi, magnitude, slope, offset)
    f = resonator_s11(f, f_f, k_ex_2pi, k_in_2pi, magnitude, slope, offset)
    return (1 - p_e - p_f) * g + p_e * e + p_f * f


class ResonatorS11Gef(lmf.Model):

    def __init__(self):
        super().__init__(resonator_s11_gef)

    def guess(self, data: np.ndarray, f: np.ndarray, chi_pi: float, chi_ef_pi: float,
              p_e: float, p_f: float) -> lmf.Parameters:
        limit = (data[0] + data[-1])/2
        magnitude = abs(limit)
        distance = abs(data - limit)
        diameter = max(distance)
        f_dip = f[np.nonzero(distance > diameter/sqrt(2))[0]]
        k = abs(f_dip[-1] - f_dip[0])
        dk = (diameter - magnitude) / magnitude * k

        return self.make_params(
            f_g=f[np.argmax(distance)],
            chi_pi=chi_pi,
            chi_ef_pi=chi_ef_pi,
            k_ex_2pi=(k + dk)/2,
            k_in_2pi=(k - dk)/2,
            p_e=p_e,
            p_f=p_f,
            magnitude=magnitude,
            slope=0,
            offset=np.angle(limit) + pi,
        )


def resonator_s11_ge_ratio(f, f_g, chi_pi, k_ex_2pi, k_in_2pi, p_ge):
    g = resonator_s11(f, f_g, k_ex_2pi, k_in_2pi, 1, 0, 0)
    e = resonator_s11_ge(f, f_g, chi_pi, k_ex_2pi, k_in_2pi, 1-p_ge, 1, 0, 0)
    return g / e


class ResonatorS11GeRatio(lmf.Model):

    def __init__(self):
        super().__init__(resonator_s11_ge_ratio)


def resonator_s11_gef_ratio(f, f_g, chi_pi, k_ex_2pi, k_in_2pi, p_eg):
    g = resonator_s11_ge(f, f_g, chi_pi, k_ex_2pi, k_in_2pi, p_eg, 1, 0, 0)
    e = resonator_s11_ge(f, f_g, chi_pi, k_ex_2pi, k_in_2pi, 1, 1, 0, 0)
    return g / e


class ResonatorS11GefRatio(lmf.Model):

    def __init__(self):
        super().__init__(resonator_s11_gef_ratio)


def qubit_s11(f, f_q, n, gamma_ex_2pi, gamma_in_2pi, gamma_phi_2pi, n_th, magnitude, slope, offset):
    n_eff = gamma_in_2pi * n_th / (gamma_ex_2pi + gamma_in_2pi)
    gamma1_2pi = gamma_ex_2pi + (2*n_th + 1) * gamma_in_2pi
    gamma2_2pi = gamma1_2pi/2 + gamma_phi_2pi
    factor = gamma_ex_2pi / (2*n_eff + 1) / gamma2_2pi
    numerator = 1 - 1j*(f - f_q)/gamma2_2pi
    denominator = 1 + ((f - f_q)/gamma2_2pi)**2 + 2*gamma_ex_2pi*n/gamma1_2pi/gamma2_2pi/pi
    s11 = 1 - factor * numerator / denominator
    return magnitude * exp(1j*(slope * (f - f_q) + offset)) * s11


class QubitS11(lmf.Model):

    def __init__(self):
        super().__init__(qubit_s11)


def qubit_s11_power(f, f_q, power, attenuation, gamma_ex_2pi, gamma_in_2pi, gamma_phi_2pi, n_th, magnitude, slope, offset):
    n = power / h / f / attenuation
    n_eff = gamma_in_2pi * n_th / (gamma_ex_2pi + gamma_in_2pi)
    gamma1_2pi = gamma_ex_2pi + (2*n_th + 1) * gamma_in_2pi
    gamma2_2pi = gamma1_2pi/2 + gamma_phi_2pi
    factor = gamma_ex_2pi / (2*n_eff + 1) / gamma2_2pi
    numerator = 1 - 1j*(f - f_q)/gamma2_2pi
    denominator = 1 + ((f - f_q)/gamma2_2pi)**2 + 2*gamma_ex_2pi*n/gamma1_2pi/gamma2_2pi/pi
    s11 = 1 - factor * numerator / denominator
    return magnitude * exp(1j*(slope * (f - f_q) + offset)) * s11


class QubitS11Power(lmf.Model):

    def __init__(self):
        super().__init__(qubit_s11_power, independent_vars=['f', 'power'])


def fogi_rabi(t, kappa_2pi, gamma_2pi, g_2pi, t0, factor, offset):
    kappa = 2*pi * kappa_2pi
    gamma = 2*pi * gamma_2pi
    g = 2*pi * g_2pi

    def p_f(t):
        omega = sqrt(-4*g**2 + (kappa-gamma)**2/4 + 0j)
        factor = exp(-(kappa + gamma)/2*t)
        term0 = cosh(omega*t/2)
        term1 = (kappa - gamma)/2/omega * sinh(omega*t/2)
        return factor * abs(term0 + term1)**2

    return factor * p_f(t - t0) + offset


class FogiRabi(lmf.Model):

    def __init__(self):
        super().__init__(fogi_rabi)