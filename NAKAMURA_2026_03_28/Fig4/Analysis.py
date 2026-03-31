import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import lmfit as lmf
from lmfit.models import (
    LorentzianModel, GaussianModel, SineModel, LinearModel, ConstantModel
)
import scipy.signal as sg
from scipy import interpolate, optimize
from scipy.ndimage import shift as nd_shift
from sklearn.decomposition import PCA
from datataking import search_datadict_miyamura
from scipy.integrate import quad, cumulative_trapezoid
from scipy.special import gamma, zeta, eval_hermite, factorial
from mpmath import *
import time as T
import math
from plottr.data.datadict_storage import DataDict, DDH5Writer, datadict_from_hdf5
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift, ifft

class SpatiotemporalAnalysis:
    def __init__(self, header, data, ctrl_path, ctrl_pulse_tx_path, ctrl_pulse_rx_path, num_of_ph_amp, num_of_fogi_timing, f_if):
        self.passing_band=0.01
        self.header = header
        self.data = data
        self.num_of_ph_amp = num_of_ph_amp
        self.num_of_fogi_timing = num_of_fogi_timing
        self.f_if = f_if
        self.duration = ctrl_pulse_tx_path["duration"]
        _, self.ctrl_pulse_tx_dd = search_datadict_miyamura(ctrl_path, ctrl_pulse_tx_path["date"], acquire_time=ctrl_pulse_tx_path["acquire_time"])
        _, self.ctrl_pulse_rx_dd = search_datadict_miyamura(ctrl_path, ctrl_pulse_rx_path["date"], acquire_time=ctrl_pulse_rx_path["acquire_time"])
        
        self.fogi_delay = []
        self.time = []
        self.waveform = []
        self.waveform_zero_fogi = []
        self.energys = []
        self.energy_bases = []
        self.energy_base_imags = []
        # self.rates = []
        self.y_absorbed_signals = []
        self.y_base_signals = []

        self.ctrl_pulse_tx_time = []
        self.ctrl_pulse_tx_waveform = []
        self.ctrl_pulse_rx_time = []
        self.ctrl_pulse_rx_waveform = []
        
        self._load_data()
        self._process_ctrl_pulse(self.ctrl_pulse_tx_dd, self.ctrl_pulse_tx_time, self.ctrl_pulse_tx_waveform)
        self._process_ctrl_pulse(self.ctrl_pulse_rx_dd, self.ctrl_pulse_rx_time, self.ctrl_pulse_rx_waveform)

    
    def _load_data(self):
        dd = datadict_from_hdf5(self.header + self.data + "/data")
        for p in range(self.num_of_ph_amp):
            k = p * self.num_of_fogi_timing
            d = dd['delay']['values'][k:k + self.num_of_fogi_timing]
            t = dd['time']['values'][k:k + self.num_of_fogi_timing]
            w = dd['waveform']['values'][k:k + self.num_of_fogi_timing]
            wf = dd['waveform_wo_fogi_rx']['values'][k:k + self.num_of_fogi_timing]
            
            self.fogi_delay.append(d)
            self.time.append(t)
            self.waveform.append(w)
            self.waveform_zero_fogi.append(wf)
            
        self.process_data()

    def _process_ctrl_pulse(self, datadict, times, signals):
        time = datadict["time"]["values"].ravel()
        waveform = datadict["waveform"]["values"].ravel()
        time, waveform = start_stop(time, waveform, 0, self.duration)
        times.append(time)

        lpf = 2 * lowpass(
                time, waveform * np.exp(2j * np.pi * self.f_if*1e-9 * time),
                self.passing_band, 0.03, 0.1, 90
                )
        index_max = np.argmax(np.abs(lpf))
        phase = np.angle(lpf[index_max])
        signal = 2 * lowpass(time, 
                            waveform * np.exp(2j*np.pi*self.f_if*1e-9*time),
                            0.01, 0.03, 0.1, 90) * np.exp(-1j*phase)
        norm=np.sum(2*np.abs(signal)**2)**0.5

        signals.append(signal/norm)
        
    def process_data(self):
        for p in range(self.num_of_ph_amp):
            rate = []
            energy = []
            energy_base = []
            energy_base_imag = []
            y_absorbed_signal = []
            y_base_signal = []
            
            for n in range(len(self.fogi_delay[0])):
                x = self.time[p][n]
                y_absorbed = self.waveform[p][n]
                lpf_absorbed = 2 * lowpass(
                x, y_absorbed * np.exp(2j * np.pi * self.f_if  * (x * 1e-9)),
                self.passing_band, 0.03, 0.1, 90
                )
                index_max_absorbed = np.argmax(np.abs(lpf_absorbed))
                phase_absorbed = np.angle(lpf_absorbed[index_max_absorbed])
                signal_absorbed = lpf_absorbed * np.exp(-1j * phase_absorbed)
                y_absorbed_signal.append(signal_absorbed)
                
                y_base = self.waveform_zero_fogi[p][n]
                lpf_base= 2 * lowpass(
                x, y_base * np.exp(2j * np.pi * self.f_if  * (x * 1e-9)),
                self.passing_band, 0.03, 0.1, 90
                )
                index_max_base = np.argmax(np.abs(lpf_base))
                phase_base = np.angle(lpf_base[index_max_base])
                signal_base = lpf_base * np.exp(-1j * phase_base)
                y_base_signal.append(signal_base)
                
                E = np.sum(np.abs(signal_absorbed) ** 2) * 2
                E1 = np.sum(np.abs(signal_base) ** 2) * 2
                E_imag = np.sum(np.abs(signal_base.imag) ** 2) * 2
                
                energy.append(E)
                energy_base.append(E1)
                energy_base_imag.append(E_imag)
                # rate.append((1 - E / E1) * 100)
            
            self.energys.append(energy)
            self.energy_bases.append(energy_base)
            self.energy_base_imags.append(energy_base_imag)
            # self.rates.append(rate)
            self.y_absorbed_signals.append(y_absorbed_signal)
            self.y_base_signals.append(y_base_signal)
        
        self.results = self.get_results()
        energy_base_avg = np.average(self.results['energy_bases'])
        energy_base_std = np.std(self.results['energy_bases'])

        # self.rates = (1 - np.ravel(self.results['energys']) / energy_base_avg) * 100
        # self.results['rates_err'] = (np.ravel(self.results['energys']) / energy_base_avg**2) * energy_base_std * 100
        self.rates = (1 - np.ravel(self.results['energys']) / np.ravel(self.results['energy_bases'])) * 100
        self.results['rates_err'] = (np.ravel(self.results['energys']) / np.ravel(self.results['energy_bases'])**2) * energy_base_std * 100
        self.results['energy_imag'] = np.ravel(self.energy_base_imags) 
        
    def get_results(self):
        return {
            "fogi_delay": self.fogi_delay,
            "time": self.time,
            "waveform": self.waveform,
            "waveform_zero_fogi": self.waveform_zero_fogi,
            "energys": self.energys,
            "energy_bases": self.energy_bases,
            "y_absorbed_signals": self.y_absorbed_signals,
            "y_base_signals": self.y_base_signals
        }
    
    def get_rates(self):
        return {
            "rates": self.rates,
            "rates_err": self.results['rates_err']
        }
    
    def roll_with_interpolation(self, array, shift):
        return nd_shift(array, shift=shift, mode='constant', cval=0.0)
    
    def overlap_comm(self, data_tx, data_rx, taus):
        ys = []
        
        for tau in taus:
            shift = tau/2
            shifted_data_rx = self.roll_with_interpolation(data_rx, shift=shift)
            new_func = -data_tx * np.conjugate(shifted_data_rx)
            y = np.abs(np.sum(new_func) * 2)**2
            ys.append(y)

        return np.array(ys) * 100
    
    def overlap_comm_delay_dep(self, data_rx, taus, delay_shift, eight=False):
        ys = []

        taus_exp = np.ravel(self.results['fogi_delay'])
        tx_waveforms = self.results['y_base_signals'][0]

        for tau in taus:
            idx = np.argmin(np.abs(taus_exp - tau))
            data_tx = tx_waveforms[idx][100:-149]
            norm=np.sum(2*np.abs(data_tx)**2)**0.5  # shape: [N_delay, N_time]
            
            data_tx_norm = data_tx/norm

            shift = (tau + delay_shift) / 2
            shifted_data_rx = self.roll_with_interpolation(data_rx, shift=shift)
            new_func = -data_tx_norm * np.conjugate(shifted_data_rx)
            y = np.abs(np.sum(new_func) * 2)**2
            ys.append(y)

        return np.array(ys) * 100

    def overlap_curve_fitting(self, data, signal_ph_tx, signal_ph_rx, taus):
        def overlap_curve(taus, delay):
            def new_func(taus, delay):
                delay = int(round(delay))
                return self.overlap_comm(signal_ph_tx, signal_ph_rx, taus + delay)
            return new_func(taus, delay)

        par_ini = {'delay':18}
        par_min = {'delay': -np.inf}
        par_max = {'delay': np.inf}
        par_vary = {'delay': True}

        model = lmf.Model(overlap_curve)

        params = model.make_params()
        for name in model.param_names:
            params[name].set(value = par_ini[name],min = par_min[name],max = par_max[name],vary = par_vary[name])
        result = model.fit(data = data, params = params, taus = taus, method='nelder-mead')
        return result
    
    def overlap_curve_fitting1(self, data, signal_ph_tx, signal_ph_rx, taus, eight=False):
        def overlap_curve(taus, delay):
            def new_func(taus, delay):
                delay = int(round(delay))
                return self.overlap_comm_delay_dep(signal_ph_rx, taus, delay, eight)
            return new_func(taus, delay)

        par_ini = {'delay':18}
        par_min = {'delay': -np.inf}
        par_max = {'delay': np.inf}
        par_vary = {'delay': True}

        model = lmf.Model(overlap_curve)

        params = model.make_params()
        for name in model.param_names:
            params[name].set(value = par_ini[name],min = par_min[name],max = par_max[name],vary = par_vary[name])
        result = model.fit(data = data, params = params, taus = taus, method='nelder-mead')
        print("delay", result.params['delay'].value)
        return result
    
    def plot_results(self, ax=None, label=None, color=None, fitting=True, fitting1=True, init_fit=False, err_bar=False, x_target=None, eight=False):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel(r'F0g1 delay $\tau$ (ns)', size="large")
            ax.set_ylabel('Absorption rate R', size="large")
            
            plt.tick_params(pad=10, top=True, bottom=True, left=True, right=True)
            plt.rcParams["xtick.direction"] = 'in'
            plt.rcParams["ytick.direction"] = 'in'
            plt.tight_layout()
        
        # ax.plot(np.ravel(self.results['fogi_delay']), np.ravel(self.results['rates']), 
        #         "o", label=label, color=color)
        ax.plot(np.ravel(self.results['fogi_delay']), self.rates*0.01, 
                "o", label=label, color=color)
        if err_bar:
            ax.errorbar(np.ravel(self.results['fogi_delay']), self.rates*0.01, 
                yerr=self.results['rates_err']*0.01, fmt='o', 
                label=label, color=color, capsize=5)

        if fitting:
            fit_result = self.overlap_curve_fitting(
                np.ravel(self.rates), 
                np.asarray(self.ctrl_pulse_tx_waveform[0]),
                # np.asarray(self.results['waveform_zero_fogi'][0][0][50:551]), 
                np.asarray(self.ctrl_pulse_rx_waveform[0])[::-1], 
                np.ravel(self.results['fogi_delay'])
            )
            
            if init_fit:
                ax.plot(np.ravel(self.results['fogi_delay']), fit_result.init_fit*0.01, "-", label=label, color=color)
                if x_target is not None:
                    f_interp = interpolate.interp1d(np.ravel(self.results['fogi_delay']), fit_result.init_fit*0.01, kind="linear")
                    y_fit_target = float(f_interp(x_target))
                    print(f"Fit curve at x={x_target:.3f} -> y={y_fit_target:.3f}")
                idx_18 = np.argmin(np.abs(np.ravel(self.results['fogi_delay']) - 18))
                print("ideal efficiency", fit_result.init_fit[idx_18]*0.01)
            else:
                ax.plot(np.ravel(self.results['fogi_delay']), fit_result.best_fit*0.01, "-", label=label, color=color)

        if fitting1:
            fit_result = self.overlap_curve_fitting1(
                np.ravel(self.rates), 
                np.asarray(self.ctrl_pulse_tx_waveform[0]),
                # np.asarray(self.results['waveform_zero_fogi'][0][0][50:551]), 
                np.asarray(self.ctrl_pulse_rx_waveform[0])[::-1], 
                np.ravel(self.results['fogi_delay']),
                eight
            )
            if init_fit:
                ax.plot(np.ravel(self.results['fogi_delay']), fit_result.init_fit*0.01, "--", label=label, color=color)
                # print(fit_result.init_fit*0.01)
                if x_target is not None:
                    f_interp = interpolate.interp1d(np.ravel(self.results['fogi_delay']), fit_result.init_fit*0.01, kind="linear")
                    y_fit_target = float(f_interp(x_target))
                    print(f"Fit curve at x={x_target:.3f} -> y={y_fit_target:.3f}")
            else:
                ax.plot(np.ravel(self.results['fogi_delay']), fit_result.best_fit*0.01, "-", label=label, color=color)
        return ax
    
    
    @staticmethod
    def plot_absorption_rate_matrix(data, err, ax):
        extent = (0, 3, 0, 3)
        data=data*0.01
        im = ax.imshow(data, extent=extent, filternorm=False, vmin=0, vmax=1)

        colorbar = plt.colorbar(im, ax=ax, label="Absorption efficiency $R_{mn}$")
        colorbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_ylabel(r"Sender mode $m$")
        ax.set_xlabel(r"Receiver mode $n$")        

        ax.tick_params(top=True, bottom=True, left=True, right=True)

        num_x_ticks = data.shape[1]
        num_y_ticks = data.shape[0]
        x_ticks = np.linspace(extent[0] + 0.37, extent[1] - 0.37, num_x_ticks)
        y_ticks = np.linspace(extent[2] + 0.37, extent[3] - 0.37, num_y_ticks)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(range(num_x_ticks))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(range(num_y_ticks))

        for j in range(num_y_ticks):
            for k in range(num_x_ticks):
                value = (rf"{data[k, j]:.2f}" )
                        # + "\n" 
                        # + rf"$\pm${err[k, j]:.2f}")
                if data[k, j] > 0.5:
                    ax.text(x_ticks[j], 3 - y_ticks[k], value, ha='center', va='center', color="red", fontsize=8)
                else:
                    ax.text(x_ticks[j], 3 - y_ticks[k], value, ha='center', va='center', color="white", fontsize=8)
    


def load_datadict(data_path, date, acquire_time, name=None):
    _, datadict = search_datadict_miyamura(data_path, date, acquire_time=acquire_time, name=name)
    return datadict

def fourier_tr(x, y):
    off_ini = np.mean(y)
    y_mod = y - off_ini
    N = len(y)
    x_fft = np.fft.fftfreq(N,d=x[1]-x[0])                                                                                                                                                   
    y_fft = np.fft.fft(y_mod)
    sorted_idx = np.argsort(x_fft)
    x_fft = x_fft[sorted_idx]
    y_fft = y_fft[sorted_idx]
    return x_fft, y_fft

def fourier_tr_padding(x, y):
    N = len(x)
    offset_init = np.mean(y)
    y_n = y - offset_init
    n_padding = 100
    freq_fft = np.fft.fftfreq(N*n_padding, (x[1]-x[0]))
    data_fft = np.fft.fft(np.concatenate([y_n, [0]*N*(n_padding-1)]))
    sorted_idx = np.argsort(freq_fft)
    freq_fft = freq_fft[sorted_idx]
    data_fft = data_fft[sorted_idx]
    return freq_fft, data_fft

def fourier_tr_padding_centered(x, y, center_x, n_padding=100):
    N = len(y)
    
    center_idx = np.argmin(np.abs(x - center_x))
    
    half_len = min(center_idx, N - center_idx - 1)
    y_centered = y[center_idx - half_len : center_idx + half_len + 1]
    
    pad_len = len(y_centered) * (n_padding - 1)
    pad_left = pad_len // 2
    pad_right = pad_len - pad_left
    y_pad = np.concatenate([np.zeros(pad_left, dtype=complex),
                            y_centered,
                            np.zeros(pad_right, dtype=complex)])
    y_pad_shifted = np.fft.ifftshift(y_pad)
    data_fft = np.fft.fft(y_pad_shifted)
    freq_fft = np.fft.fftfreq(len(y_pad), x[1] - x[0])
    
    data_fft = np.fft.fftshift(data_fft)
    freq_fft = np.fft.fftshift(freq_fft)
    
    return freq_fft, data_fft



def fourier_no_padding(x, y):
    y_n = y - np.mean(y)
    N = len(x)
    dt = x[1] - x[0]
    freq_fft = np.fft.fftfreq(N, dt)
    data_fft = np.fft.fft(y_n)
    sorted_idx = np.argsort(freq_fft)
    freq_fft = freq_fft[sorted_idx]
    data_fft = data_fft[sorted_idx]
    return freq_fft, data_fft

def start_stop(x, y, xstart, xstop):
    x_list = np.array([el for el in x if el>=xstart and el<=xstop])
    x_idx = [i for i, el in enumerate(x) if el>=xstart and el<=xstop]
    y_list = y[x_idx]
    return x_list, y_list


def lowpass(t, x, fp, fs, gpass, gstop):
    samplerate = 1/(t[1]-t[0])
    fn = samplerate / 2                      
    wp = fp / fn                             
    ws = fs / fn                             
    N, Wn = sg.buttord(wp, ws, gpass, gstop) 
    b, a = sg.butter(N, Wn, "low")           
    y = sg.filtfilt(b, a, x)                 
    return y

def demodulate(t, data, demodulation_if = 0.125):
    return (data * np.exp(2j * np.pi * demodulation_if * t)).mean(axis=-1)

def FFT(x, y):
    off_ini = np.mean(y)
    y_mod = y - off_ini
    Y = np.abs(np.fft.fft(y_mod))
    dt = 2e-9
    N = len(y)
    t = x
    f = np.fft.fftfreq(N, dt)
    sorted_idx=np.argsort(f)
    x_fft = f[sorted_idx]
    y_fft = Y[sorted_idx]
    return x_fft, y_fft

def LPF(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate/2
    wp = fp/fn
    ws = fs/fn
    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(N, Wn, "low")
    y = signal.filtfilt(b, a, x)
    return y

def overlap(func1, func2):
    i = 0
    for t in range(len(func1)):
        i += func1[t] * np.conjugate(func2[t])
    return np.abs(i)

######### sech
def sech(x):
    return 1 / np.cosh(x)

_kappa_cache = {}

def _N_m(m, kappa):
    numerator = 8 * (1 - 2**(1 - 2*m)) * gamma(2*m + 1) * zeta(2*m)
    denominator = kappa**(2*m + 1)
    return 1.0 / (numerator / denominator)

def _compute_Z_m(m, A_coeffs, kappa):
    Z = 0.0
    for k in range(m // 2 + 1):
        deg_k = m - 2 * k
        A_k = A_coeffs[deg_k]
        inner_sum = 0.0
        for l in range(m // 2 + 1):
            deg_l = m - 2 * l
            A_l = A_coeffs[deg_l]
            Nj_index = m - (k + l)
            Nj = _N_m(Nj_index, kappa)
            inner_sum += A_l / Nj
        Z += A_k * inner_sum
    return 1 / Z

def _get_A_m(m, kappa):
    if kappa not in _kappa_cache:
        _kappa_cache[kappa] = {
            "A_list": {0: [1.0]},
            "Z_dict": {0: kappa / 4},
        }

    A_list = _kappa_cache[kappa]["A_list"]
    Z_dict = _kappa_cache[kappa]["Z_dict"]

    if m in A_list:
        return A_list[m]

    A_m_coeffs = [0.0] * (m + 1)
    A_m_coeffs[m] = 1.0

    for i in range(1, m // 2 + 1):
        lower_m = m - 2 * i
        A_lower = _get_A_m(lower_m, kappa)

        inner_sum = 0.0
        for j in range(0, (m // 2 - i) + 1):
            index_in_A = m - 2 * (i + j)
            if 0 <= index_in_A < len(A_lower):
                A_coeff = A_lower[index_in_A]
                Nj = _N_m(m - (i + j), kappa)
                inner_sum += A_coeff / Nj

        Z = Z_dict.get(lower_m, 1.0)
        for power, coeff in enumerate(A_lower):
            if power < len(A_m_coeffs):
                A_m_coeffs[power] -= inner_sum * Z * coeff

    Z_m = _compute_Z_m(m, A_m_coeffs, kappa)
    Z_dict[m] = Z_m
    A_list[m] = A_m_coeffs
    return A_m_coeffs

def sech_m(t, m, kappa):
    A_coeffs = _get_A_m(m, kappa)
    A_coeffs = np.array([float(c) for c in A_coeffs], dtype=float)
    Zm = float(_kappa_cache[kappa]["Z_dict"][m])
    A_val =sum(c * t**i for i, c in enumerate(A_coeffs))  
    return np.sqrt(Zm) * sech(kappa * t / 2) * A_val


#### hermite
def hermite(x, n, kappa):
    x_scaled = np.sqrt(kappa) * x
    norm = np.sqrt(np.sqrt(kappa / np.pi)) / np.sqrt(2**n * factorial(n))
    psi = norm * eval_hermite(n, x_scaled) * np.exp(-x_scaled**2 / 2)
    return np.asarray(psi, dtype=np.float64) 

