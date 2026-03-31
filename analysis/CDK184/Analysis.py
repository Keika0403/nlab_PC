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


class Analysis:
    def __init__(self, data:np.ndarray, v1:np.ndarray, v2:np.ndarray|None=None) -> None:
        self.v1 = np.unique(v1)
        if v2 is not None:
            self.v2 = np.unique(v2)
            self.data = data.reshape(len(self.v2), len(self.v1))
        else:
            self.data = data
            self.v2 = None

    def pca(self):
        data_IQ = np.array([self.data.real, self.data.imag]).T
        pca = PCA(n_components=2)
        pca.fit(data_IQ)
        data_pca = pca.transform(data_IQ)
        return data_pca[: ,0].real
    
    def pca_2d(self):
        assert self.v2 is not None
        data = self.data.ravel()
        data_IQ = np.array([data.real, data.imag]).T
        pca = PCA(n_components=2)
        pca.fit(data_IQ)
        data_pca = pca.transform(data_IQ)
        return data_pca[: ,0].real.reshape(len(self.v2), len(self.v1))
    
    def plot_adjust(self, title=None, labels=None, lims=None):
        if title is not None:
            plt.title(title)
        if labels is not None:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
        if lims is not None:
            plt.xlim(lims[0])
            if len(lims) == 2:
                plt.ylim(lims[1])
    
    def plot(self, title=None, labels=None, lims=None, figsize=(4, 3), legend=False, grid=False, N=1, show=True):
        """
        title:str \n
        labels = [xlabel, ylabel] \n
        lims = [(),()]
        """
        fig = plt.figure(figsize=figsize)
        if self.v2 is None:
            data = self.pca()
            plt.plot(self.v1, data * N)
        else:
            if self.data.dtype == "complex128":
                data = self.pca_2d()
                assert data.dtype != "complex128", data.dtype
            else: data = self.data

            c = plt.pcolormesh(self.v1, self.v2, data * N, shading="auto")
            cbar = fig.colorbar(c)
            cbar.ax.tick_params(labelsize=14)
        self.plot_adjust(title=title, labels=labels, lims=lims)
        plt.grid(grid)
        if legend: plt.legend()
        if show:plt.show()

    def fit_1d(self, kind, plot=False):
        if kind == "resonator":
            F = ResonatorFit(self.data, self.v1)
            return F.fit()
        data_pca = self.pca()
        F = Fit(data_pca, self.v1)
        if kind == "ramsey": res = F.ramsey_fit()
        elif kind == "rabi": res = F.rabi_fit()
        elif kind == "gaussian": res = F.gaussian_fit()
        elif kind == "decay": res = F.decay_fit()
        # if plot:
        #     plt.plot(self.v1, data_pca)
        #     plt.plot(self.v1, res.best_fit)
        #     plt.plot(self.v1, res.init_fit)
        return res
    # def fit_1d(self):

    def fit_successive(self, kind, extract_parameter=None):
        assert self.v2 is not None
        results = []
        for i, v in enumerate(self.v1):
            data = self.data[i]
            results.append(self.fit_1d(kind=kind))
        return results

class Fit:
    def __init__(self, data, v,) -> None:
        assert len(data) == len(v), f"{len(data)} != {len(v)}"
        self.data = data 
        self.v = v
        self.envelope = np.abs(sg.hilbert(data))
        # plt.figure(figsize=(4,2));plt.plot(self.v, self.data)
        # plt.show()
    
    def find_osc_init(self):
        x_fft, y_fft = fourier_tr(self.v, self.data)
        x_fft, y_fft = start_stop(x_fft, y_fft, 0, x_fft[-1])
        A = max(self.data) - min(self.data)
        off = np.mean(self.data)
        freq = abs(x_fft[np.argmax(np.abs(y_fft))])
        return A, off, freq, np.angle(y_fft[np.argmax(np.abs(y_fft))])
    
    def find_decay_init(self, hilbert=False):
        if hilbert: data = self.envelope
        else: data = self.data
        off_ini = np.mean(data[-5:])
        moving_average = np.convolve(data, np.ones(25) / 25, mode="valid")
        A_ini = moving_average[0]
        gamma_ini = abs((moving_average[0] - moving_average[5]) / (self.v[0] - self.v[5])) / A_ini
        return A_ini, gamma_ini, off_ini

    def find_decay_init2(self, hilbert=False):
        if hilbert: data = self.envelope
        else: data = self.data
        off_ini = np.mean(data[-5:])
        moving_average = np.convolve(data, np.ones(25) / 25, mode="valid")
        A_ini = self.data[0]-off_ini
        gamma_ini = abs((moving_average[0] - moving_average[5]) / (self.v[0] - self.v[5])) / A_ini
        return A_ini, gamma_ini, off_ini

    def quadra_fit(self):
        def modelf(x, a, b, c): return a*x**2 + b*x + c
        par_ini = dict(a=-0.5, b=0, c=5.65)
        par_max = dict(a=np.inf, b=np.inf, c=np.inf)
        par_min = dict(a=-np.inf, b=-np.inf, c=-np.inf)
        par_var = dict(a=True, b=False, c=True)
        model = lmf.Model(modelf)
        params = model.make_params()
        for name in params:
            params[name].set(
                value=par_ini[name],  # 初期値
                max=par_max[name],
                min=par_min[name],
                vary=par_var[name] # パラメータを動かすかどうか
            )
        return model.fit(self.data, x=self.v, params=params, method='leastsq')

    def ramsey_fit(self):
        def modelf(t, T2, freq, A, phi, off):
            return A * np.exp(-t / T2) * np.cos(2 * np.pi * freq * t + phi) + off
        A, off, freq, phi = self.find_osc_init()
        T2_ini = 1 / self.find_decay_init(hilbert=True)[1]
        par_ini = dict(T2=T2_ini, freq=freq, A=A/2, phi=phi, off=off)
        par_max = dict(T2=10 * T2_ini, freq=freq * 2, A=A*2, phi=np.inf, off=np.inf)
        par_min = dict(T2=0, freq=0, A=A / 1e5, phi=-np.inf, off=-np.inf)
        par_var = dict(T2=True, freq=True, A=True, phi=True, off=True)
        model = lmf.Model(modelf)
        params = model.make_params()
        for name in params:
            params[name].set(
                value=par_ini[name],  # 初期値
                max=par_max[name],
                min=par_min[name],
                vary=par_var[name] # パラメータを動かすかどうか
            )
        return model.fit(self.data, t=self.v, params=params, method='leastsq')
    
    def decay_fit(self, hilbert=False):
        def modelf(t, T1, A, off): return A * np.exp(-(t-t[0]) / T1) + off
        A, gamma, off = self.find_decay_init2(hilbert=hilbert)
        gamma *= 2
        par_ini = dict(T1=1/abs(gamma), A=A, off=off)
        par_max = dict(T1=10 / abs(gamma), A=A*2, off=abs(off) * 5)
        par_min = dict(T1=0, A=A / 1e5, off=-abs(off) * 5)
        par_var = dict(T1=True, A=True, phi=True, off=True)
        model = lmf.Model(modelf)
        params = model.make_params()
        # print(par_ini["off"])
        for name in params:
            params[name].set(
                value=par_ini[name],  # 初期値
                max=par_max[name],
                min=par_min[name],
                vary=par_var[name] # パラメータを動かすかどうか
            )
        return model.fit(self.data, t=self.v, params=params, method='leastsq')
    
    def sin_fit(self):
        def modelf(t, freq, A, phi, off): 
            return A * np.cos(2 * np.pi * freq * t + phi) + off
        A, off, freq, phi = self.find_osc_init()
        par_ini = dict(freq=freq, A=A, phi=phi, off=off)
        par_max = dict(freq=freq * 2, A=A*2, phi=np.pi, off=abs(off) * 5)
        par_min = dict(freq=0, A=A / 1e5, phi=-np.pi, off=-abs(off) * 5)
        par_var = dict(freq=True, A=True, phi=True, off=True)
        model = lmf.Model(modelf)
        params = model.make_params()
        for name in params:
            params[name].set(
                value=par_ini[name],  # 初期値
                max=par_max[name],
                min=par_min[name],
                vary=par_var[name] # パラメータを動かすかどうか
            )
        return model.fit(self.data, t=self.v, params=params, method='leastsq')

    def linear_fit(self):
        model = LinearModel()
        par = model.guess(self.data, x=self.v)
        return model.fit(self.data, par, x=self.v)

    def rabi_fit(self):
        model = SineModel()
        par = model.guess(self.data, x=self.v)
        res = model.fit(self.data, par, x=self.v)
        omega = res.params.valuesdict()["frequency"]
        phi = res.params.valuesdict()["shift"]
        delta = np.pi / omega
        n_list = np.arange(self.v[0] // delta, self.v[-1] // delta + 6, 1)
        print([((n-0.5)*np.pi-phi)/omega for n in n_list if (n*np.pi-phi)/omega >= self.v[0] and (n*np.pi-phi)/omega<self.v[-1]])
        return res

    def lorentz_fit(self):
        model = LorentzianModel() + ConstantModel()
        par = LorentzianModel().guess(self.data, x=self.v)
        par += ConstantModel().make_params()
        return model.fit(self.data, par, x=self.v)
    
    def gaussian_fit(self):
        model = GaussianModel()
        par = model.guess(self.data, x=self.v)
        return model.fit(self.data, par, x=self.v)

class ResonatorFit:
    def __init__(self, data, f) -> None:
        assert len(data) == len(f), f"{len(data)} != {len(f)}"
        self.data = data 
        self.f = f
        self.B = np.array([[0,0,0,-2], [0,1,0,0], [0,0,1,0], [-2,0,0,0]])

    def plot(self, with_fit=False, init=False):
        res = self.fit()
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(self.data.real, self.data.imag, "ko")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(self.f, np.unwrap(np.angle(self.data)), "ko")
        ax2.set_xlabel("Frequency (GHz)")
        ax2.set_ylabel("Phase (rad)")
        if with_fit:
            s11 = res.best_fit
            ax.plot(s11.real, s11.imag, "r--")
            ax2.plot(self.f, np.unwrap(np.angle(s11)))
            if init:
                ax.plot(res.init_fit.real, res.init_fit.imag, "g--")
                ax2.plot(self.f, np.unwrap(np.angle(res.init_fit)))
        plt.show()

    def return_M(self):
        x, y = self.data.real, self.data.imag
        z = x**2 + y**2
        M_11, M_12, M_13, M_14 = np.sum(z**2), np.sum(x*z), np.sum(y*z), np.sum(z)
        M_22, M_23, M_24 = np.sum(x**2), np.sum(x*y), np.sum(x)
        M_33, M_34 = np.sum(y**2), np.sum(y)
        M_44 = len(x)
        M = np.array(
            [[M_11,M_12,M_13,M_14],
            [M_12,M_22,M_23,M_24],
            [M_13,M_23,M_33,M_34],
            [M_14,M_24,M_34,M_44]]
        )
        return M
    
    def center_of_circle(self):
        def determinant(eta, M): return np.linalg.det(M - eta * self.B)
        from scipy.optimize import newton
        M = self.return_M()
        B_inv = np.linalg.inv(self.B)
        try: eta = newton(determinant, 0, args=(M,), tol=1e-13)
        except: eta=0.1
        X = np.linalg.inv(np.dot(B_inv,M)-eta*np.eye(4)-1e-5*np.eye(4))
        a = np.array([1,0,0,0])
        for _ in range(1000):
            a = np.dot(X,a)
            a = a / np.sqrt((a[1]**2 + a[2]**2 -4*a[0]*a[3]))
        x_c = -a[1] / 2 / a[0]
        y_c = -a[2] / 2 / a[0]
        r_0 = 1 / 2 / np.abs(a[0])
        return x_c, y_c, r_0
    
    def guess_phase(self, phase):
        deriv = np.diff(phase)
        F = Fit(deriv, self.f[1:])
        lor_fit = F.lorentz_fit()
        f_r_ini = lor_fit.params.valuesdict()["center"]
        kappa = lor_fit.params.valuesdict()["sigma"]
        return f_r_ini, f_r_ini / kappa
    
    def phase_fit(self, phase):
        def modelf(x, f_r, theta_0, Q_l, C):
            return theta_0 + 2 * np.arctan(2 * Q_l * (1 - (x / f_r))) - C * (x - f_r) * 2 * np.pi
        model = lmf.Model(modelf)
        f_r_ini, Q_l_ini = self.guess_phase(phase)
        par_ini = dict(f_r=f_r_ini, theta_0=phase[0]-np.pi, Q_l=Q_l_ini, C=0,)
        par_max = dict(f_r=f_r_ini*2, theta_0=np.inf, Q_l=np.inf, C=np.inf,)
        par_min = dict(f_r=f_r_ini/2, theta_0=-np.inf, Q_l=0, C=-np.inf,)
        par_var = dict(f_r=True, theta_0=True, Q_l=True, C=True,)
        model = lmf.Model(modelf)
        params = model.make_params()
        for name in params:
            params[name].set(
                value=par_ini[name],  # 初期値
                max=par_max[name],
                min=par_min[name],
                vary=par_var[name] # パラメータを動かすかどうか
            )
        res = model.fit(phase, params, x=self.f)
        # fig = plt.figure()
        # plt.plot(self.f, phase)
        # plt.plot(self.f, res.best_fit)
        # plt.plot(self.f, res.init_fit)
        # plt.show()
        return res
    
    def guess(self):
        x_c, y_c, A = self.center_of_circle()
        r_0 = np.sqrt(x_c**2 + y_c**2)
        # assert A > r_0
        data_mod = self.data - x_c - 1j * y_c
        phase = np.unwrap(np.angle(data_mod))
        res_phase_fit = self.phase_fit(phase)
        f_r_ini = res_phase_fit.params.valuesdict()['f_r']
        theta_0 = res_phase_fit.params.valuesdict()['theta_0']
        Q_l = res_phase_fit.params.valuesdict()['Q_l']
        Ed = res_phase_fit.params.valuesdict()['C']
        beta = (theta_0 + np.pi) % (2 * np.pi)
        P_tilda = (x_c + r_0 * np.cos(beta)) + 1j*(y_c + r_0 * np.sin(beta))
        alpha_ini = np.angle(P_tilda)
        phi_ini = beta - alpha_ini
        par_ini=dict(
            f_r=f_r_ini, k_ex=(A-r_0)/A*f_r_ini/Q_l, k_in=r_0/A*f_r_ini/Q_l,
            phi=0, A=A, C=0, Ed=Ed, alpha=theta_0,
        )
        # par_ini=dict(
        #     f_r=10.51, k_ex=40e-3, k_in=1e-3,
        #     phi=0, A=A, C=0, Ed=0, alpha=theta_0,
        # )
        par_max=dict(
            f_r=14, k_ex=1, k_in=1,
            phi=np.inf, A=np.inf, C=np.inf, Ed=10, alpha=2*np.pi,
        )
        par_min=dict(
            f_r=6, k_ex=0, k_in=0,
            phi=-np.inf, A=0, C=-np.inf, Ed=-10, alpha=-2*np.pi,
        )
        par_var=dict(
            f_r=True, k_ex=True, k_in=True,
            phi=True, A=True, C=True, Ed=True, alpha=True,
        )
        return par_ini, par_max, par_min, par_var
    
    def fit(self):
        par_ini, par_max, par_min, par_var = self.guess()
        model = lmf.Model(s11)
        params = model.make_params()
        for name in params:
            params[name].set(
                value=par_ini[name],  # 初期値
                max=par_max[name],
                min=par_min[name],
                vary=par_var[name] # パラメータを動かすかどうか
            )
        return model.fit(self.data, f=self.f, params=params, method='leastsq')

class WaveformAnalysis:
    def __init__(self, waveforms, time, fogi_freqs, lo_freq, freq_relation="RF=LO-IF") -> None:
        self.time = np.unique(time)
        self.fogi_freqs = np.unique(fogi_freqs)
        self.waveforms = waveforms.reshape(len(self.fogi_freqs), len(self.time))
        self.lo_freq = lo_freq
        if freq_relation == "RF=LO-IF": self._to_rf = lambda if_freq:self.lo_freq + if_freq
        elif freq_relation == "RF=LO+IF": self._to_rf = lambda if_freq:self.lo_freq - if_freq
        else: raise AssertionError
        self.freq_relation = freq_relation
    
    def plot(self):
        Analysis(self.waveforms, self.time, self.fogi_freqs).plot()

    def extract_frequency(self, plot=False):
        photon_frequencies, arg_max_frequencies, fourier_amps = [], [], []
        for waveform in self.waveforms:
            freq_fft, fourier = fourier_tr_padding(self.time, waveform)
            freq_fft, fourier = start_stop(freq_fft, fourier, freq_fft[0], 0)
            photon_frequencies.append(self._to_rf(freq_fft))
            arg_max_frequencies.append(photon_frequencies[-1][np.argmax(np.abs(fourier))])
            fourier_amps.append(fourier)
        photon_frequencies, arg_max_frequencies = np.array(photon_frequencies), np.array(arg_max_frequencies)
        fourier_amps = np.array(fourier_amps)
        if plot: Analysis(np.abs(fourier_amps), self.fogi_freqs, photon_frequencies).plot()
        return photon_frequencies, arg_max_frequencies
        
    def extract_decay_rate(self, fit_start=30, fp=0.02, fs=0.05, gpass=1, gstop=90):
        decay_rates, stderrs, _ = [], [], []
        for i, waveform in enumerate(self.waveforms):
            freq_fft, fourier = fourier_tr_padding(self.time, waveform)
            freq_fft, fourier = start_stop(freq_fft, fourier, 0, freq_fft[-1])
            photon_frequency = self._to_rf(freq_fft)
            if self.freq_relation=="RF=LO-IF": photon_if_freq = self.lo_freq - photon_frequency[np.argmax(np.abs(fourier))] #### changed by Sunada
            else: photon_if_freq = self.lo_freq + photon_frequency[np.argmax(np.abs(fourier))]
            signal = 2 * lowpass(self.time, 
                                 waveform * np.exp(2j * np.pi * photon_if_freq * self.time), 
                                 fp, fs, gpass, gstop)
            envelope = np.abs(signal)
            t, env = start_stop(self.time, envelope, fit_start, self.time[-1])
            result = decay_fit(env, t)
            decay_rates.append(result.params.valuesdict()["gamma"] /2/np.pi)
            stderrs.append(result.result.params["gamma"].stderr / 2 / np.pi)
        return decay_rates, stderrs  

    def extract_decay_rate_and_freq(self, fit_start=30, fp=0.02, fs=0.05, gpass=1, gstop=90, plot=False, skip=[]):
        """
        return: decay rate, photon_freq, (stderrs, photon_frequencies)
        """
        photon_frequencies, arg_max_frequencies, fourier_amps = [], [], []
        decay_rates, stderrs, _ = [], [], []
        if plot:
            plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
            plt.rcParams['ytick.direction'] = 'in'
            plt.rcParams['xtick.major.width'] = 0.9#x軸主目盛り線の線幅
            plt.rcParams['ytick.major.width'] = 0.9#y軸主目盛り線の線幅
            plt.rcParams['font.size'] = 8 #フォントの大きさ
            plt.rcParams['axes.linewidth'] = 0.9# 軸の線幅edge linewidth。囲みの太さ
            plt.rcParams['xtick.top'] = True
            plt.rcParams['xtick.bottom'] = True
            plt.rcParams['ytick.left'] = True
            plt.rcParams['ytick.right'] = True
            plt.rcParams['lines.linewidth'] = 1
            plt.rcParams['lines.markersize'] = 2.
            ncols = 3
            nrows = (len(self.waveforms) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 1 * nrows), constrained_layout=True)
            axes = axes.flatten()

            ncols_f = 3
            nrows_f = (len(self.waveforms) + ncols_f - 1) // ncols_f
            fig_freq, axes_freq = plt.subplots(nrows=nrows_f, ncols=ncols_f, figsize=(12, 3 * nrows_f))
            axes_freq = axes_freq.flatten()
        for i, waveform in enumerate(self.waveforms):
            if i in skip:
                continue
            freq_fft, fourier = fourier_tr_padding(self.time, waveform)
            freq_fft, fourier = start_stop(freq_fft, fourier, freq_fft[0], 0)
            photon_frequency = self._to_rf(freq_fft)
            photon_frequencies.append(photon_frequency)
            arg_max_frequencies.append(photon_frequencies[-1][np.argmax(np.abs(fourier))])
            fourier_amps.append(fourier)
            if self.freq_relation=="RF=LO-IF": photon_if_freq = self.lo_freq - photon_frequency[np.argmax(np.abs(fourier))]
            else: photon_if_freq = self.lo_freq + photon_frequency[np.argmax(np.abs(fourier))]
            signal = 2 * lowpass(self.time, 
                                 waveform * np.exp(2j * np.pi * photon_if_freq * self.time), 
                                 fp, fs, gpass, gstop)
            envelope = np.abs(signal)
            t, env = start_stop(self.time, envelope, fit_start, self.time[-1])
            result = decay_fit(env, t)
            decay_rates.append(result.params.valuesdict()["gamma"] /2/np.pi)
            try: stderrs.append(result.result.params["gamma"].stderr/ 2 / np.pi)
            except: stderrs.append(None)
            if plot:
                axes[i].plot(self.time, waveform*1e3, label="Waveform", alpha=0.5)
                # axes[i].plot(t, env, label="Envelope")
                axes[i].plot(t, result.best_fit*1e3, color="red", lw=2.5,  alpha=0.7, label="Fit")
                # axes[i].legend()
                # axes[i].set_title(f"Freq{i}: {photon_frequency[np.argmax(np.abs(fourier))]}")
                axes[i].set_xlabel("Time (ns)")
                axes[i].set_xlim(0, 1000)
                axes[i].set_ylabel("Amplitude (mV)")

                axes_freq[i].plot(photon_frequency, np.abs(fourier), label="|FFT|", lw=2)
                axes_freq[i].set_title(f"Freq{i}: {photon_frequency[np.argmax(np.abs(fourier))]}")
                axes_freq[i].set_xlim(9.3, 9.5)
                axes_freq[i].set_xlabel("Freq (GHz)")
                axes_freq[i].set_ylabel("Amplitude")
                axes_freq[i].grid(True)
        if plot:
            plt.tight_layout()
            plt.show()
        photon_frequencies, arg_max_frequencies = np.array(photon_frequencies), np.array(arg_max_frequencies)
        fourier_amps = np.array(fourier_amps)
        return np.array(decay_rates), arg_max_frequencies, (stderrs, photon_frequencies)

    def extract_decay_rate_and_freq_thesis(self, fit_start=30, fp=0.02, fs=0.05, gpass=1, gstop=90, plot=False):
        """
        return: decay rate, photon_freq, (stderrs, photon_frequencies)
        """
        photon_frequencies, arg_max_frequencies, fourier_amps = [], [], []
        decay_rates, stderrs, _ = [], [], []
        print("aaaaaaa")
        if plot:
            plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
            plt.rcParams['ytick.direction'] = 'in'
            plt.rcParams['xtick.major.width'] = 0.9#x軸主目盛り線の線幅
            plt.rcParams['ytick.major.width'] = 0.9#y軸主目盛り線の線幅
            plt.rcParams['font.size'] = 8 #フォントの大きさ
            plt.rcParams['axes.linewidth'] = 0.9# 軸の線幅edge linewidth。囲みの太さ
            plt.rcParams['xtick.top'] = True
            plt.rcParams['xtick.bottom'] = True
            plt.rcParams['ytick.left'] = True
            plt.rcParams['ytick.right'] = True
            plt.rcParams['lines.linewidth'] = 1
            plt.rcParams['lines.markersize'] = 2.
            ncols = 3
            nrows = (len(self.waveforms) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 2 * nrows))
            axes = axes.flatten()

            ncols_f = 3
            nrows_f = (len(self.waveforms) + ncols_f - 1) // ncols_f
            fig_freq, axes_freq = plt.subplots(nrows=nrows_f, ncols=ncols_f, figsize=(12, 3 * nrows_f))
            axes_freq = axes_freq.flatten()
        for i, waveform in enumerate(self.waveforms):
            freq_fft, fourier = fourier_tr_padding(self.time, waveform)
            freq_fft, fourier = start_stop(freq_fft, fourier, freq_fft[0], 0)
            photon_frequency = self._to_rf(freq_fft)
            photon_frequencies.append(photon_frequency)
            arg_max_frequencies.append(photon_frequencies[-1][np.argmax(np.abs(fourier))])
            fourier_amps.append(fourier)
            if self.freq_relation=="RF=LO-IF": photon_if_freq = self.lo_freq - photon_frequency[np.argmax(np.abs(fourier))]
            else: photon_if_freq = self.lo_freq + photon_frequency[np.argmax(np.abs(fourier))]
            signal = 2 * lowpass(self.time, 
                                 waveform * np.exp(2j * np.pi * photon_if_freq * self.time), 
                                 fp, fs, gpass, gstop)
            envelope = np.abs(signal)
            t, env = start_stop(self.time, envelope, fit_start, self.time[-1])
            result = decay_fit(env, t)
            decay_rates.append(result.params.valuesdict()["gamma"] /2/np.pi)
            try: stderrs.append(result.result.params["gamma"].stderr/ 2 / np.pi)
            except: stderrs.append(None)
            if plot:
                axes[i].plot(self.time, waveform*1e3, label="Waveform")
                # axes[i].plot(t, env, label="Envelope")
                axes[i].plot(t, result.best_fit*1e3, color="red", lw=2, label="Fit")
                # axes[i].legend()
                # axes[i].set_title(f"Freq{i}: {photon_frequency[np.argmax(np.abs(fourier))]}")
                axes[i].set_xlabel("Time (ns)")
                axes[i].set_ylabel("Amplitude (mV)")

                axes_freq[i].plot(photon_frequency, np.abs(fourier), label="|FFT|", lw=2)
                axes_freq[i].set_title(f"Freq{i}: {photon_frequency[np.argmax(np.abs(fourier))]}")
                axes_freq[i].set_xlim(9.3, 9.5)
                axes_freq[i].set_xlabel("Freq (GHz)")
                axes_freq[i].set_ylabel("Amplitude")
                axes_freq[i].grid(True)
        if plot:
            plt.tight_layout()
            plt.show()
        photon_frequencies, arg_max_frequencies = np.array(photon_frequencies), np.array(arg_max_frequencies)
        fourier_amps = np.array(fourier_amps)
        return np.array(decay_rates), arg_max_frequencies, (stderrs, photon_frequencies)


class WaveformAnalysisMulti:
    def __init__(self, name_dict:dict, data_path:str, T2e=None, name=None) -> None:
        amps_dict, datadicts, params_dict, results_dict = dict(), dict(), dict(), dict()
        funcs_dict=dict()
        for k, v in name_dict.items():
            print(v["amp"], v["date"], v["acquire_time"])
            amps_dict[k] = v["amp"]
            datadicts[k] = load_datadict(data_path, v["date"], v["acquire_time"], name=name)
            params_dict[k] = v["params"]
            results_dict[k], funcs_dict[k] = dict(), dict()
        self.amps_dict = amps_dict
        self.datadicts = datadicts
        self.params = params_dict
        self.results_dict = results_dict
        self.funcs_dict = funcs_dict
        self.T2e = T2e # ns

    def gamma_t(self, time, const, plot=0, form="sech",):
        center = (time[0] + time[-1]) / 2
        def squared_sech(x):
            def func(var):
                return 1/np.cosh(const*var)**2
            norm_func = func(x)/quad(func, time[0], time[-1])[0]
            return norm_func
        def squared_anti_sech(x):
            def func(var):
                return (var/np.cosh(const*var))**2
            norm_func = func(x)/quad(func, time[0], time[-1])[0]
            return norm_func
        def Li_2(values):
            vectorized_polylog = np.vectorize(lambda t: polylog(2, -np.exp(-2 * t)))
            return vectorized_polylog(values)
        def Li_3(values):
            vectorized_polylog = np.vectorize(lambda t: polylog(3, -np.exp(-2 * t)))
            return vectorized_polylog(values)
        def Li_4(values):
            vectorized_polylog = np.vectorize(lambda t: polylog(4, -np.exp(-2 * t)))
            return vectorized_polylog(values)
        def Li_5(values):
            vectorized_polylog = np.vectorize(lambda t: polylog(5, -np.exp(-2 * t)))
            return vectorized_polylog(values)
        def Li_6(values):
            vectorized_polylog = np.vectorize(lambda t: polylog(6, -np.exp(-2 * t)))
            return vectorized_polylog(values)
        
        if form == "sech_0_nu":
            top = np.array([squared_sech(t) for t in time - center])
            bottom = 1 - (np.array([quad(squared_sech, center, t)[0] for t in time])) ** 2
        elif form=="sech_0":
            top = (const/2)/(np.cosh(const*(time-center)))**2
            bottom = 1-(1/2)*(np.tanh(const*(time-center))-np.tanh(const*(time[0]-center)))
            print(bottom)
        elif form=="sech_1":                                                      
            top = (6*const**3/np.pi**2)*(time-center)**2/(np.cosh(const*(time-center)))**2
            def f(t):
                return 6 * (Li_2(t) + t * (-t - 2 * np.log(np.exp(-2 * t) + 1) + t * np.tanh(t))) / np.pi**2
            bottom = 1-(f(const*(time-center))-f(const*(time[0]-center)))
        elif form=="sech_2":                                                    
            top = (45*const**5/2/np.pi**4) * ((time-center)**2 - (np.pi/const)**2/12)**2/ (np.cosh(const * (time - center)))**2
            def f(t):
                f_2 = 6*t**2*Li_2(t) + 6*t*Li_3(t) + 3*Li_4(t) - t**4*(1 - np.tanh(t)) - 4*t**3*np.log(np.exp(-2 * t) + 1)
                f_1 = -np.pi**2 * (Li_2(t) + t * (-t - 2 * np.log(np.exp(-2 * t) + 1) + t * np.tanh(t))) /6
                f_0 = np.pi**4*np.tanh(t)/144
                return 45*(f_2 + f_1 + f_0)/2/np.pi**4
            bottom = 1 - (f(const * (time - center)) - f(const * (time[0] - center)))
        elif form=="sech_3":
            top = (350*const**7/9/np.pi**6) * ((time-center)**3 - 7*(np.pi/const)**2/20*(time-center))**2/ (np.cosh(const * (time - center)))**2
            def f(t):
                f_3 = 15*t**4*Li_2(t) + 30*t**3*Li_3(t) + 45*t**2*Li_4(t) + 45*t*Li_5(t) + 45/2*Li_6(t) - t**6*(1 - np.tanh(t)) - 6*t**5*np.log(np.exp(-2 * t) + 1)
                f_2 = -7*np.pi**2*(6*t**2*Li_2(t) + 6*t*Li_3(t) + 3*Li_4(t) - t**4*(1 - np.tanh(t)) - 4*t**3*np.log(np.exp(-2 * t) + 1))/10
                f_1 =  (7*np.pi**2/20)**2* (Li_2(t) + t * (-t - 2 * np.log(np.exp(-2 * t) + 1) + t * np.tanh(t)))
                return 350*(f_3 + f_2 + f_1)/9/np.pi**6
            bottom = 1 - (f(const * (time - center)) - f(const * (time[0] - center)))
        if plot:
            plt.figure(figsize=(6,2))
            plt.plot(time, top/bottom/2/np.pi)
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.show()
        gammat = top/bottom/2/np.pi
        if form=="sech_0":
            gammat = gammat
        else:
            float_list = [float(val) for val in gammat]
            gammat = np.array(float_list)
        return gammat
    
    def gamma_t_num(self, time, target, plot=0):
        abs_sq = np.abs(target)**2
        integral = cumulative_trapezoid(abs_sq, time, initial=0)
        denominator = 1 - integral
        gammat = abs_sq / denominator / (2 * np.pi)
        return gammat

    def get_values(self, datadict):
        time = np.unique(datadict["time"]["values"])
        fogi_freqs = np.unique(datadict["fogi_frequency"]["values"])
        waveforms = datadict["waveform"]["values"].reshape(len(fogi_freqs), len(time))
        return waveforms, time, fogi_freqs

    def generate_results(self, lo_freq, thesis=False):
        if thesis:
            for k, datadict in self.datadicts.items():
                waveforms, time, fogi_freqs = self.get_values(datadict)
                obj = WaveformAnalysis(waveforms, time, fogi_freqs, lo_freq=lo_freq, freq_relation="RF=LO-IF") 
                decay_rate, photon_freqs, _ = obj.extract_decay_rate_and_freq_thesis(self.params[k]["fit_start"], 
                                                self.params[k]["fp"], 
                                                self.params[k]["fs"],
                                                self.params[k]["gpass"],
                                                self.params[k]["gstop"],
                                                self.params[k]["plot"])
                if self.T2e is not None: self.results_dict[k]["decay_rates"] = decay_rate * 2  - 1 / self.T2e
                else: self.results_dict[k]["decay_rates"] = decay_rate * 2
                self.results_dict[k]["stderrs"] = _[0]
                skip_indices = self.params[k].get("skip", [])
                fogi_freqs = [f for i, f in enumerate(fogi_freqs) if i not in skip_indices]
                photon_freqs = [f for i, f in enumerate(photon_freqs) if i not in skip_indices]
                decay_rates = [f*2 for i, f in enumerate(decay_rate) if i not in skip_indices]
                self.results_dict[k]["fogi_freqs"] = fogi_freqs
                self.results_dict[k]["photon_freqs"] = photon_freqs
                self.results_dict[k]["decay_rates"] = decay_rates
        else:
            for k, datadict in self.datadicts.items():
                waveforms, time, fogi_freqs = self.get_values(datadict)
                obj = WaveformAnalysis(waveforms, time, fogi_freqs, lo_freq=lo_freq, freq_relation="RF=LO-IF") 
                decay_rate, photon_freqs, _ = obj.extract_decay_rate_and_freq(self.params[k]["fit_start"], 
                                                self.params[k]["fp"], 
                                                self.params[k]["fs"],
                                                self.params[k]["gpass"],
                                                self.params[k]["gstop"],
                                                self.params[k]["plot"],
                                                self.params[k].get("skip", []))
                if self.T2e is not None: self.results_dict[k]["decay_rates"] = decay_rate * 2  - 1 / self.T2e
                else: self.results_dict[k]["decay_rates"] = decay_rate * 2
                self.results_dict[k]["stderrs"] = _[0]
                skip_indices = self.params[k].get("skip", [])
                fogi_freqs = [f for i, f in enumerate(fogi_freqs) if i not in skip_indices]
                photon_freqs = [f for i, f in enumerate(photon_freqs) if i not in skip_indices]
                self.results_dict[k]["fogi_freqs"] = fogi_freqs
                self.results_dict[k]["photon_freqs"] = photon_freqs

    def plot_decayrates(self):
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        for k, v in self.results_dict.items():
            ax.plot(v["fogi_freqs"], v["decay_rates"], '.-',label=f"amp {self.amps_dict[k]}")
            ax2.plot(v["photon_freqs"], v["decay_rates"], '.-',label=f"amp {self.amps_dict[k]}")
        ax.set_xlabel('fogi freq (GHz)')
        ax.set_ylabel('$\Gamma_f $ (MHz)')
        ax2.set_xlabel('photon freq (GHz)')
        ax2.set_ylabel('$\Gamma_f $ (MHz)')
        ax2.legend(bbox_to_anchor=(1.6, 1.1), loc='upper right')
        ax.grid()
        ax2.grid()
        plt.tight_layout()
        plt.show()

    def generate_control_pulse(self, fogi_lo, target_freq, duration, kappa, form="sech", mode_id=0, plot=True, tx=False, target_waveform=[]):
        time = np.arange(duration)
        Gammas = [0]
        fogi_freq_shifted = []
        amps = list(self.amps_dict.values())
        amps.insert(0, 0)
        for k, v in self.results_dict.items():
            photon_freqs = v["photon_freqs"]
            fogi_freqs = v["fogi_freqs"]
            decay_rates = v["decay_rates"]
            photonfreq_to_decayrate = interpolate.interp1d(photon_freqs, decay_rates, bounds_error=False, fill_value=(decay_rates[-1], decay_rates[0]))
            Gammas.append(photonfreq_to_decayrate(target_freq))
            photonfreq_to_fogifreq = interpolate.interp1d(photon_freqs, fogi_freqs, bounds_error=False, fill_value=(fogi_freqs[-1], fogi_freqs[0]))
            fogi_freq_shifted.append(photonfreq_to_fogifreq(target_freq))
        # envelope
        fogi_freq_shifted.insert(0, fogi_freq_shifted[0])
        res = polynomial_even_fit(x=amps, data=Gammas)
        c = tuple(res.params.valuesdict().values())
        x = np.linspace(0, amps[-1], 10001)
        Gamma_to_amp = interpolate.interp1d(polynomial_even(x, c[0], c[1], c[2], c[3], c[4], c[5]),
                                             x, bounds_error=False, fill_value=(0, amps[-1]))
        if form == "sech":
            target = sech_m(time-(time[-1]-time[0])/2, mode_id, kappa)
        elif form == "hermite":
            target = hermite(time-(time[-1]-time[0])/2, mode_id, kappa)
        elif form == "arbitrary":
            target = target_waveform[mode_id]
        AWGamp_of_time = lambda time:Gamma_to_amp(self.gamma_t_num(time, target=target))

        if tx:
            envelope = AWGamp_of_time(time)*np.exp(-1j*np.angle(target))
        else:
            envelope = AWGamp_of_time(time)

        if plot:
            fig = plt.figure(figsize = (6.5, 3))
            ax = fig.add_subplot(1, 2, 1)
            ax.plot(amps, res.best_fit * 1e3, )
            ax.plot(amps, np.array(Gammas)*1e3,'ro')
            ax.set_xlabel('Drive amplitude (V)')
            ax.set_ylabel('$\Gamma_{\mathrm{f}}$ (MHz)')
            virtual_Gamma = np.linspace(0, 20e-3, 10001)
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(virtual_Gamma, Gamma_to_amp(virtual_Gamma),'r-', label='fit')
            ax2.plot(Gammas, amps, 'ko')
            ax2.set_ylabel('Drive amplitude (V)')
            ax2.set_xlabel('$\Gamma_{\mathrm{f}}$ (GHz)')
            plt.show()

            fig = plt.figure(figsize = (6.5, 2))
            ax = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            ax.plot(time, envelope.real)
            ax.plot(time, envelope.imag)
            ax.set_title('AWGamplitude vs time')
            ax2.plot(time, target.real)
            ax2.plot(time, target.imag)
            # print(time, const, duration)
            ax2.set_title('target shape')
            plt.show()

        # fogi frequency
        fogi_freq_of_AWGamp = interpolate.interp1d(amps, fogi_freq_shifted, bounds_error=False, 
                                                   fill_value=(fogi_freq_shifted[0], fogi_freq_shifted[-1]))
        

        def fogi_freq_of_time(time):
            f_freq = fogi_freq_of_AWGamp(AWGamp_of_time(time))
            return f_freq
        fogi_frequency = fogi_freq_of_time(time) 
        
        if plot:
            fig = plt.figure(figsize=(5, 2))
            ax = fig.add_subplot(1, 2, 1)
            ax.plot(amps, fogi_freq_of_AWGamp(amps), "ro", ls="--")
            ax.set_xlabel('Drive amplitude (V)')
            ax.set_ylabel('$\Gamma_{\mathrm{ph}}$')
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(time, fogi_frequency) 
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Momentary frequency')
            plt.show()

        # generation
        phase_of_time=[0]
        for i in range(1,len(time)):
            next_phi = phase_of_time[i-1]+(fogi_frequency[i] - fogi_lo) 
            phase_of_time.append(next_phi)
        phase_of_time = np.array(phase_of_time)
        waveform_at_AWG = envelope*np.exp(2j * np.pi * phase_of_time) 
        if plot:
            plt.figure(figsize=(4, 3))
            plt.plot(time,waveform_at_AWG.real)
            plt.plot(time,waveform_at_AWG.imag)
            plt.plot(time,envelope)
            plt.xlabel('Time (ns)')
            plt.ylabel('Amplitude (V)')
            plt.show()
        return waveform_at_AWG, envelope

    def amp_Gamma(self, target_freq):
        Gammas = [0]
        fogi_freq_shifted = []
        amps = list(self.amps_dict.values())
        amps.insert(0, 0)
        for k, v in self.results_dict.items():
            photon_freqs = v["photon_freqs"]
            fogi_freqs = v["fogi_freqs"]
            decay_rates = v["decay_rates"]
            photonfreq_to_decayrate = interpolate.interp1d(photon_freqs, decay_rates, bounds_error=False, fill_value=(decay_rates[-1], decay_rates[0]))
            Gammas.append(photonfreq_to_decayrate(target_freq))
            photonfreq_to_fogifreq = interpolate.interp1d(photon_freqs, fogi_freqs, bounds_error=False, fill_value=(fogi_freqs[-1], fogi_freqs[0]))
            fogi_freq_shifted.append(photonfreq_to_fogifreq(target_freq))
        # envelope
        fogi_freq_shifted.insert(0, fogi_freq_shifted[0])
        res = polynomial_even_fit(x=amps, data=Gammas)
        return amps, res.best_fit, np.array(Gammas), np.array(fogi_freq_shifted)

class SechPhotonAnalysis:
    def __init__(self, data_path, result_dict, duration=1000, kappa=2.5e-3 * 2 * np.pi*2, readout_lo_freq=9.47):
        self.data_path = data_path
        self.result_dict = result_dict
        self.duration = duration
        self.kappa = kappa
        self.readout_lo_freq = readout_lo_freq
        self.time = []
        self.waveform_envelope = []
        self.signals = []
        self.opt_signals = []

    def fit_waveform(self, time, data, kappa, x0=[2.5e-3, 450], form='sech', mode_id=0, target_waveform=[], padded=False):
        def func(x, time, kappa, mode_id):
            t_shift = time - x[1]
            if form == "sech":
                y = x[0]*sech_m(t_shift, mode_id, kappa)
            if form == "hermite":
                y = x[0]*hermite(t_shift, mode_id, kappa)
            if form == "arbitrary":
                y = x[0]*self.roll_with_interpolation(target_waveform[mode_id], shift=- x[1])
            return y
        kappa = self.kappa
        cost_func = lambda x: np.linalg.norm(np.abs(func(x, time, kappa, mode_id)) - np.abs(data))
        best_x = optimize.minimize(cost_func, x0=x0, method='Nelder-Mead').x
        print("best_x:", best_x)
        fitted = func(best_x, time, kappa, mode_id)
        if padded:
            extra_time_front = 200
            dt = time[1] - time[0]
            n_extra = int(extra_time_front/dt)
            time_extended = np.linspace(time[0]-extra_time_front, time[-1], len(time)+n_extra)
            fitted = func(best_x, time_extended, kappa, mode_id)
            return fitted, best_x[1], time_extended
        else:
            fitted = func(best_x, time, kappa, mode_id)
            return fitted, best_x[1]
    
    def analyze_waveforms_data(self, passing_band, legend=False, savefig=False, mode_opt=False, target_waveform=[]):
        times = []
        signals = []
        ress = []
        fouriers = []
        fourier_tars = []
        photon_freqs = []
        photon_freq_tars = []


        result_items = list(self.result_dict.items())
        n = len(result_items)

        for i, (k, v) in enumerate(self.result_dict.items()):
            target_freq = v["target_freq"]
            form = v["form"]
            mode_id = v["mode_id"]
            x0 = v["x0"]

            _, datadict = search_datadict_miyamura(self.data_path, v["date"], acquire_time=v["acquire_time"])
            time, waveform = start_stop(datadict["time"]["values"].ravel(), datadict["waveform"]["values"].ravel(), 0, self.duration)
            times.append(time)
           
            # phase = np.angle(demodulate(time, waveform, self.readout_lo_freq - target_freq))
            lpf = 2 * lowpass(
                time, waveform * np.exp(2j * np.pi * (self.readout_lo_freq - target_freq) * time),
                passing_band, 0.03, 0.1, 90
                )
            index_max = np.argmax(np.abs(lpf))
            phase = np.angle(lpf[index_max])
            print("phase:", phase)
            signal = lpf * np.exp(-1j * phase)
            norm = np.linalg.norm(signal)
            signal = signal/norm
            waveform = waveform/norm
            res, best_x = self.fit_waveform(time, signal, self.kappa, x0, form, mode_id, target_waveform)
            
            sign = np.sign(np.vdot(res, signal.real))
            print("sign:", sign)
            signals.append(signal*sign)
            ress.append(res)


            xfft, fourier = fourier_tr_padding_centered(time, signal, best_x)
            # xfft, fourier = start_stop(xfft, fourier, xfft[0], 0)
            photon_freq = target_freq + xfft
            photon_freqs.append(photon_freq)
            if i == 0:
                scale_t = np.max(np.abs(waveform))
            # signals.append(signal)
            # ax_t.plot(time,  res/scale_t, color="tab:gray", lw=2.5,  alpha=0.5, label="Target")
            # # ax_t.plot(time, waveform, "tab:blue", alpha=0.3, lw=0.5, label="Data")
            # ax_t.plot(time, sign*signal.real/scale_t, color=colors[i], lw=1.2,linestyle="-", label="Real") 
            # # ax_t.plot(time, sign*signal.imag, "tab:blue", lw=1.2,linestyle=(0, (2, 1)),  label="Imag")
            # # ax_t.plot(time, 2000*signal.real, "tab:blue", lw=1.2,linestyle="-", label="Real") 
            # # ax_t.plot(time, 2000*signal.imag, "tab:blue", lw=1.2,linestyle=(0, (2, 1)),  label="Imag")
            # ax_t.label_outer()
            # ax_t.set_xlim(0, self.duration)
            # ax_t.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            xfft_tar, fourier_tar = fourier_tr_padding_centered(time,  res, best_x)
            # xfft_tar, fourier_tar = start_stop(xfft_tar, fourier_tar, xfft_tar[0], 0)
            photon_freq_tar = target_freq + xfft_tar
            photon_freq_tars.append(photon_freq_tar)
            fouriers.append(fourier)
            fourier_tars.append(fourier_tar)
            if i == 0:
                scale_f = np.max(np.abs(fourier.real))
            # if mode_id % 2 == 0:
            #     sign = np.sign(np.vdot(fourier.real, fourier_tar.real))
            #     fourier     *= sign
            #     ax_f[0].plot(photon_freq_tar,  fourier_tar.real/scale_f, color="tab:gray", lw=2.5, alpha=0.7, label="Target")
            #     ax_f[0].plot(photon_freq,     fourier.real/scale_f, color=colors[i], lw=1.2,  label="Real")
            #     ax_f[1].plot(photon_freq,     fourier.imag/scale_f, color=colors[i], lw=1.2, label="Imag")
            # else:
            #     sign = np.sign(np.vdot(fourier.imag, fourier_tar.imag))
            #     fourier     *= sign
            #     ax_f[1].plot(photon_freq_tar, fourier_tar.imag/scale_f, color="tab:gray", lw=2.5, alpha=0.7, label="Target")
            #     ax_f[0].plot(photon_freq,      fourier.real/scale_f, color=colors[i], lw=1.2,  label="Real")
            #     ax_f[1].plot(photon_freq,      fourier.imag/scale_f, color=colors[i], lw=1.2, label="Imag")
            # ax_f[0].set_xlim(target_freq-0.02, target_freq+0.02)
            # ax_f[0].label_outer()
            # ax_t.set_xlabel("Time (ns)")
            # ax_f[0].set_xlabel("Frequency (GHz)")
            # ax_f[1].set_xlim(target_freq-0.02, target_freq+0.02)
            # ax_f[1].label_outer()
            # ax_f[1].set_xlabel("Frequency (GHz)")
        return time, signals, ress, photon_freqs, fouriers, fourier_tars, photon_freq_tars


    def analyze_waveforms_overlap(self, passing_band, legend=False, savefig=False, mode_opt=False, target_waveform=[]):
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.tight_layout(rect=[0.05, 0.02, 1, 1])
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 0.9#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 0.9#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 8 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 0.9# 軸の線幅edge linewidth。囲みの太さ
        plt.rcParams['xtick.top'] = True
        plt.rcParams['xtick.bottom'] = True
        plt.rcParams['ytick.left'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams['lines.markersize'] = 2.
        times = []
        signals = []
        transmissivity = []
        fouriers = []
        fourier_tars = []
        freq_axes = []

        result_items = list(self.result_dict.items())
        n = len(result_items)
        
        fig_t, ax_t = plt.subplots(1, 1, figsize=(1.6 * 2, 0.9 * 2))
        fig_f, ax_f = plt.subplots(1, 2, figsize=(1.6 * 2, 0.9 * 2))


        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        for i, (k, v) in enumerate(self.result_dict.items()):
            target_freq = v["target_freq"]
            form = v["form"]
            mode_id = v["mode_id"]
            x0 = v["x0"]

            _, datadict = search_datadict_miyamura(self.data_path, v["date"], acquire_time=v["acquire_time"])
            time, waveform = start_stop(datadict["time"]["values"].ravel(), datadict["waveform"]["values"].ravel(), 0, self.duration)
            times.append(time)
           
            # phase = np.angle(demodulate(time, waveform, self.readout_lo_freq - target_freq))
            lpf = 2 * lowpass(
                time, waveform * np.exp(2j * np.pi * (self.readout_lo_freq - target_freq) * time),
                passing_band, 0.03, 0.1, 90
                )
            index_max = np.argmax(np.abs(lpf))
            phase = np.angle(lpf[index_max])
            print("phase:", phase)
            signal = lpf * np.exp(-1j * phase)
            norm = np.linalg.norm(signal)
            signal = signal/norm
            waveform = waveform/norm
            res, best_x = self.fit_waveform(time, signal, self.kappa, x0, form, mode_id, target_waveform)
            
            sign = np.sign(np.vdot(res, signal.real))
            print("sign:", sign)
            signals.append(signal*sign)

            xfft, fourier = fourier_tr_padding_centered(time, signal, best_x)
            # xfft, fourier = start_stop(xfft, fourier, xfft[0], 0)
            photon_freq = target_freq + xfft
            if i == 0:
                scale_t = np.max(np.abs(waveform))
            # signals.append(signal)
            ax_t.plot(time,  res/scale_t, color="tab:gray", lw=2.5,  alpha=0.5, label="Target")
            # ax_t.plot(time, waveform, "tab:blue", alpha=0.3, lw=0.5, label="Data")
            ax_t.plot(time, sign*signal.real/scale_t, color=colors[i], lw=1.2,linestyle="-", label="Real") 
            # ax_t.plot(time, sign*signal.imag, "tab:blue", lw=1.2,linestyle=(0, (2, 1)),  label="Imag")
            # ax_t.plot(time, 2000*signal.real, "tab:blue", lw=1.2,linestyle="-", label="Real") 
            # ax_t.plot(time, 2000*signal.imag, "tab:blue", lw=1.2,linestyle=(0, (2, 1)),  label="Imag")
            ax_t.label_outer()
            ax_t.set_xlim(0, self.duration)
            ax_t.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            xfft_tar, fourier_tar = fourier_tr_padding_centered(time,  res, best_x)
            # xfft_tar, fourier_tar = start_stop(xfft_tar, fourier_tar, xfft_tar[0], 0)
            photon_freq_tar = target_freq + xfft_tar
            
            xfft_no_pad, fourier_no_pad = fourier_no_padding(time, waveform)
            freq_abs = self.readout_lo_freq + xfft_no_pad

            mask_neg = (xfft_no_pad <= 0)
            fourier_neg = fourier_no_pad.copy()
            fourier_neg[~mask_neg] = 0
            xfft_neg = xfft_no_pad  # ?????????

            xfft_tar_no_pad, fourier_tar_no_pad = fourier_no_padding(time, res)
            freq_abs_tar = self.readout_lo_freq + xfft_tar_no_pad

            mask_neg_tar = (xfft_tar_no_pad <= 0)
            fourier_neg_tar = fourier_tar_no_pad.copy()
            fourier_neg_tar[~mask_neg_tar] = 0
            xfft_neg_tar =  self.readout_lo_freq + xfft_tar_no_pad      

            fouriers.append(fourier_neg)
            fourier_tars.append(fourier_neg_tar)
            freq_axes.append(xfft_neg_tar)
            if i == 0:
                scale_f = np.max(np.abs(fourier.real))
            if mode_id % 2 == 0:
                sign = np.sign(np.vdot(fourier.real, fourier_tar.real))
                fourier     *= sign
                ax_f[0].plot(photon_freq_tar,  fourier_tar.real/scale_f, color="tab:gray", lw=2.5, alpha=0.7, label="Target")
                ax_f[0].plot(photon_freq,     fourier.real/scale_f, color=colors[i], lw=1.2,  label="Real")
                ax_f[1].plot(photon_freq,     fourier.imag/scale_f, color=colors[i], lw=1.2, label="Imag")
            else:
                sign = np.sign(np.vdot(fourier.imag, fourier_tar.imag))
                fourier     *= sign
                ax_f[1].plot(photon_freq_tar, fourier_tar.imag/scale_f, color="tab:gray", lw=2.5, alpha=0.7, label="Target")
                ax_f[0].plot(photon_freq,      fourier.real/scale_f, color=colors[i], lw=1.2,  label="Real")
                ax_f[1].plot(photon_freq,      fourier.imag/scale_f, color=colors[i], lw=1.2, label="Imag")
            ax_f[0].set_xlim(target_freq-0.02, target_freq+0.02)
            ax_f[0].label_outer()
            ax_t.set_xlabel("Time (ns)")
            ax_f[0].set_xlabel("Frequency (GHz)")
            ax_f[1].set_xlim(target_freq-0.02, target_freq+0.02)
            ax_f[1].label_outer()
            ax_f[1].set_xlabel("Frequency (GHz)")
            transmissivity.append(fourier/(fourier_tar/2))

        fig_t.text(0.0, 0.5, "Amplitude (arb. unit)", va="center", rotation="vertical")
        fig_f.text(0.0, 0.55, "Amplitude (arb. unit)", va="center", rotation="vertical")
        self.times = times
        self.signals = signals
        if savefig:
            plt.tight_layout()
            fig_t.savefig("waveforms.pdf",bbox_inches="tight")
            fig_f.savefig("spectra.pdf",bbox_inches="tight")
        plt.tight_layout()
        plt.show()



    def analyze_waveforms(self, passing_band, legend=False, savefig=False, mode_opt=False, target_waveform=[]):
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.tight_layout(rect=[0.05, 0.02, 1, 1])
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 0.9#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 0.9#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 8 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 0.9# 軸の線幅edge linewidth。囲みの太さ
        plt.rcParams['xtick.top'] = True
        plt.rcParams['xtick.bottom'] = True
        plt.rcParams['ytick.left'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams['lines.markersize'] = 2.
        times = []
        signals = []
        transmissivity = []
        fouriers = []
        fourier_tars = []
        freq_axes = []

        result_items = list(self.result_dict.items())
        n = len(result_items)
        ncols = 2
        nrows = math.ceil(n / ncols)
        
        fig_t, axes_t = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 0.9 * nrows), sharex=True, sharey=True)
        fig_f, axes_f = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 0.9 * nrows), sharex=True, sharey=True)
        
        axes_t = axes_t.flatten()
        axes_f = axes_f.flatten()

        for i, (k, v) in enumerate(self.result_dict.items()):
            ax_t = axes_t[i]
            ax_f = axes_f[i]
            target_freq = v["target_freq"]
            form = v["form"]
            mode_id = v["mode_id"]
            x0 = v["x0"]

            _, datadict = search_datadict_miyamura(self.data_path, v["date"], acquire_time=v["acquire_time"])
            time, waveform = start_stop(datadict["time"]["values"].ravel(), datadict["waveform"]["values"].ravel(), 0, self.duration+500)
            times.append(time)
           
            # phase = np.angle(demodulate(time, waveform, self.readout_lo_freq - target_freq))
            lpf = 2 * lowpass(
                time, waveform * np.exp(2j * np.pi * (self.readout_lo_freq - target_freq) * time),
                passing_band, 0.03, 0.1, 90
                )
            index_max = np.argmax(np.abs(lpf))
            phase = np.angle(lpf[index_max])
            print("phase:", phase)
            signal = lpf * np.exp(-1j * phase)
            norm = np.linalg.norm(signal)
            signal = signal/norm
            waveform = waveform/norm
            res, best_x = self.fit_waveform(time, signal, self.kappa, x0, form, mode_id, target_waveform, padded=False)

            sign = np.sign(np.vdot(res, signal.real))
            print("sign:", sign)
            signals.append(signal*sign)
            xfft, fourier = fourier_tr_padding_centered(time, signal, best_x)
            # xfft, fourier = start_stop(xfft, fourier, xfft[0], 0)
            photon_freq = target_freq + xfft

            # signals.append(signal)
            ax_t.plot(time,  res, "r", lw=2.5,  alpha=0.7, label="Target")
            ax_t.plot(time, waveform, "tab:blue", alpha=0.3, lw=0.5, label="Data")
            ax_t.plot(time, sign*signal.real, "tab:blue", lw=1.2,linestyle="-", label="Real") 
            ax_t.plot(time, sign*signal.imag, "tab:blue", lw=1.2,linestyle=(0, (2, 1)),  label="Imag")
            # ax_t.plot(time, 2000*signal.real, "tab:blue", lw=1.2,linestyle="-", label="Real") 
            # ax_t.plot(time, 2000*signal.imag, "tab:blue", lw=1.2,linestyle=(0, (2, 1)),  label="Imag")
            ax_t.label_outer()
            ax_t.set_xlim(0, self.duration)
            ax_t.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


            xfft_tar, fourier_tar = fourier_tr_padding_centered(time,  res, best_x)
            # xfft_tar, fourier_tar = start_stop(xfft_tar, fourier_tar, xfft_tar[0], 0)
            photon_freq_tar = target_freq + xfft_tar
            
            xfft_no_pad, fourier_no_pad = fourier_no_padding(time, waveform)
            freq_abs = self.readout_lo_freq + xfft_no_pad

            mask_neg = (xfft_no_pad <= 0)
            fourier_neg = fourier_no_pad.copy()
            fourier_neg[~mask_neg] = 0
            xfft_neg = xfft_no_pad  # ?????????

            xfft_tar_no_pad, fourier_tar_no_pad = fourier_no_padding(time, res)
            freq_abs_tar = self.readout_lo_freq + xfft_tar_no_pad

            mask_neg_tar = (xfft_tar_no_pad <= 0)
            fourier_neg_tar = fourier_tar_no_pad.copy()
            fourier_neg_tar[~mask_neg_tar] = 0
            xfft_neg_tar =  self.readout_lo_freq + xfft_tar_no_pad      

            fouriers.append(fourier_neg)
            fourier_tars.append(fourier_neg_tar)
            freq_axes.append(xfft_neg_tar)
            
            if mode_id % 2 == 0:
                sign = np.sign(np.vdot(fourier.real, fourier_tar.real))
                fourier     *= sign
                ax_f.plot(photon_freq_tar,  fourier_tar.real, "r", lw=2.5, alpha=0.7, label="Target")
                ax_f.plot(photon_freq,     fourier.real,     "tab:blue", lw=1.2, linestyle="-",  label="Real")
                ax_f.plot(photon_freq,     fourier.imag,     "tab:blue", lw=1.2, linestyle=(0, (2, 1)), label="Imag")
            else:
                sign = np.sign(np.vdot(fourier.imag, fourier_tar.imag))
                fourier     *= sign
                ax_f.plot(photon_freq_tar, fourier_tar.imag, "r", lw=2.5, alpha=0.7, label="Target")
                ax_f.plot(photon_freq,      fourier.real,     "tab:blue", lw=1.2, linestyle="-",  label="Real")
                ax_f.plot(photon_freq,      fourier.imag,     "tab:blue", lw=1.2, linestyle=(0, (2, 1)), label="Imag")
            ax_f.set_xlim(target_freq-0.02, target_freq+0.02)
            ax_f.label_outer()
            if i // ncols == nrows - 1:
                ax_t.set_xlabel("Time (ns)")
                ax_f.set_xlabel("Frequency (GHz)")
            # transmissivity.append(fourier/(fourier_tar/2))

        if legend:
            handles, labels = [], []
            for ax in axes_t:
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)
            unique = dict(zip(labels, handles))

            fig_t.legend(unique.values(), unique.keys(),
                        loc='upper center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.05),
                        handlelength=1.5)
        fig_t.text(0.0, 0.5, "Amplitude (arb. unit)", va="center", rotation="vertical")
        fig_f.text(0.0, 0.55, "Amplitude (arb. unit)", va="center", rotation="vertical")
        self.times = times
        self.signals = signals
        if savefig:
            plt.tight_layout()
            fig_t.savefig("waveforms.pdf",bbox_inches="tight")
            fig_f.savefig("spectra.pdf",bbox_inches="tight")
        plt.tight_layout()
        plt.show()

        nrows = n 
        ncols = 2 
        fig, axes = plt.subplots(nrows, ncols,  figsize=(1.6 * ncols, 0.9 * nrows), sharex=False, sharey=False)
        if nrows == 1:
            axes = axes.reshape(1, 2)
        scale_t = None
        scale_f = None
        for i, (k, v) in enumerate(result_items):
            ax_t = axes[i, 0] 
            ax_f = axes[i, 1]  
            target_freq = v["target_freq"]
            form = v["form"]
            mode_id = v["mode_id"]
            x0 = v["x0"]
            ax_t.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax_f.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

            _, datadict = search_datadict_miyamura(self.data_path, v["date"], acquire_time=v["acquire_time"])
            time, waveform = start_stop(datadict["time"]["values"].ravel(), datadict["waveform"]["values"].ravel(), 0, self.duration)
    
            # phase = np.angle(demodulate(time, waveform, self.readout_lo_freq - target_freq))
            lpf = 2 * lowpass(
                time, waveform * np.exp(2j * np.pi * (self.readout_lo_freq - target_freq) * time),
                passing_band, 0.03, 0.1, 90
                )
            index_max = np.argmax(np.abs(lpf))
            phase = np.angle(lpf[index_max])
            print("phase:", phase)
            signal = lpf * np.exp(-1j * phase)
            norm = np.linalg.norm(signal)
            signal = signal/norm
            waveform = waveform/norm
            res, best_x = self.fit_waveform(time, signal, self.kappa, x0, form, mode_id, target_waveform)
            
            sign = np.sign(np.vdot(res, signal.real))
            print("sign:", sign)

            xfft, fourier = fourier_tr_padding_centered(time, signal, best_x)
            # xfft, fourier = start_stop(xfft, fourier, xfft[0], 0)
            photon_freq = target_freq + xfft
            if i == 0:
                scale_t = np.max(np.abs(waveform))

            ax_t.plot(time, res/scale_t, "r", lw=2.0, alpha=0.7, label="Target")
            ax_t.plot(time, waveform/scale_t, "tab:blue", alpha=0.3, lw=0.5, label="Data")
            ax_t.plot(time, sign*signal.real/scale_t, "tab:blue", lw=1.0, linestyle="-", label="Real")
            ax_t.plot(time, sign*signal.imag/scale_t, "tab:blue", lw=1.0, linestyle="--", label="Imag")
            ax_t.set_xlim(0, self.duration)
            ax_t.set_ylim(-1.2, 1.2)
            ax_t.set_yticks([-1, 0, 1])
            ax_t.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            if i == nrows - 1:
                ax_t.set_xlabel("Time (ns)")

            xfft_tar, fourier_tar = fourier_tr_padding_centered(time,  res, best_x)
            # xfft_tar, fourier_tar = start_stop(xfft_tar, fourier_tar, xfft_tar[0], 0)
            photon_freq_tar = target_freq + xfft_tar
            if i == 0:
                scale_f = np.max(np.abs(fourier.real))
            if mode_id % 2 == 0:
                sign = np.sign(np.vdot(fourier.real, fourier_tar.real))
                fourier *= sign
                ax_f.plot(photon_freq_tar, fourier_tar.real/scale_f, "r", lw=2.0, alpha=0.7, label="Target")
                ax_f.plot(photon_freq, fourier.real/scale_f, "tab:blue", lw=1.0, linestyle="-", label="Real")
                ax_f.plot(photon_freq, fourier.imag/scale_f, "tab:blue", lw=1.0, linestyle="--", label="Imag")
            else:
                sign = np.sign(np.vdot(fourier.imag, fourier_tar.imag))
                fourier *= sign
                ax_f.plot(photon_freq_tar, fourier_tar.imag/scale_f, "r", lw=2.0, alpha=0.7, label="Target")
                ax_f.plot(photon_freq, fourier.real/scale_f, "tab:blue", lw=1.0, linestyle="-", label="Real")
                ax_f.plot(photon_freq, fourier.imag/scale_f, "tab:blue", lw=1.0, linestyle="--", label="Imag")

            ax_f.set_xlim(target_freq - 0.02, target_freq + 0.02)
            ax_f.set_ylim(-1.2, 1.2)
            ax_f.set_yticks([-1, 0, 1])
            ax_f.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            if i == nrows - 1:
                ax_f.set_xlabel("Frequency (GHz)")

        if legend:
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.02))

        fig.text(0.0, 0.5, "Amplitude (arb. unit)", va="center", rotation="vertical")
        fig.text(0.49, 0.5, "Amplitude (arb. unit)", va="center", rotation="vertical")
        plt.tight_layout()
        if savefig:
            fig.savefig("waveform_and_spectrum.pdf", bbox_inches="tight")
        plt.show()

            
        if mode_opt:
            return self.optimize_modes(
                target_freq, photon_freq, freq_axes, fouriers, fourier_tars, transmissivity, passing_band
            )
        else:
            return signals

    def analyze_waveforms_padded(self, passing_band, legend=False, savefig=False, mode_opt=False, target_waveform=[]):
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.tight_layout(rect=[0.05, 0.02, 1, 1])
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 0.9#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 0.9#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 8 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 0.9# 軸の線幅edge linewidth。囲みの太さ
        plt.rcParams['xtick.top'] = True
        plt.rcParams['xtick.bottom'] = True
        plt.rcParams['ytick.left'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams['lines.markersize'] = 2.
        times = []
        signals = []
        transmissivity = []
        fouriers = []
        fourier_tars = []
        freq_axes = []

        result_items = list(self.result_dict.items())
        n = len(result_items)
        ncols = 2
        nrows = math.ceil(n / ncols)
        
        fig_t, axes_t = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 0.9 * nrows), sharex=True, sharey=True)
        fig_f, axes_f = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 0.9 * nrows), sharex=True, sharey=True)
        
        axes_t = axes_t.flatten()
        axes_f = axes_f.flatten()

        for i, (k, v) in enumerate(self.result_dict.items()):
            ax_t = axes_t[i]
            ax_f = axes_f[i]
            target_freq = v["target_freq"]
            form = v["form"]
            mode_id = v["mode_id"]
            x0 = v["x0"]

            _, datadict = search_datadict_miyamura(self.data_path, v["date"], acquire_time=v["acquire_time"])
            time, waveform = start_stop(datadict["time"]["values"].ravel(), datadict["waveform"]["values"].ravel(), 0, self.duration+500)
            times.append(time)
           
            # phase = np.angle(demodulate(time, waveform, self.readout_lo_freq - target_freq))
            lpf = 2 * lowpass(
                time, waveform * np.exp(2j * np.pi * (self.readout_lo_freq - target_freq) * time),
                passing_band, 0.03, 0.1, 90
                )
            index_max = np.argmax(np.abs(lpf))
            phase = np.angle(lpf[index_max])
            print("phase:", phase)
            signal = lpf * np.exp(-1j * phase)
            norm = np.linalg.norm(signal)
            signal = signal/norm
            waveform = waveform/norm
            res, best_x, time_tar = self.fit_waveform(time, signal, self.kappa, x0, form, mode_id, target_waveform, padded=True)

            sign = np.sign(np.vdot(res[len(res)-len(time)-1:-1], signal.real))
            print("sign:", sign)
            signals.append(signal*sign)
            xfft, fourier = fourier_tr_padding_centered(time, signal, best_x)
            # xfft, fourier = start_stop(xfft, fourier, xfft[0], 0)
            photon_freq = target_freq + xfft

            # signals.append(signal)
            ax_t.plot(time_tar,  res, "r", lw=2.5,  alpha=0.7, label="Target")
            ax_t.plot(time, waveform, "tab:blue", alpha=0.3, lw=0.5, label="Data")
            ax_t.plot(time, sign*signal.real, "tab:blue", lw=1.2,linestyle="-", label="Real") 
            ax_t.plot(time, sign*signal.imag, "tab:blue", lw=1.2,linestyle=(0, (2, 1)),  label="Imag")
            # ax_t.plot(time, 2000*signal.real, "tab:blue", lw=1.2,linestyle="-", label="Real") 
            # ax_t.plot(time, 2000*signal.imag, "tab:blue", lw=1.2,linestyle=(0, (2, 1)),  label="Imag")
            ax_t.label_outer()
            ax_t.set_xlim(0, self.duration)
            ax_t.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            xfft_tar, fourier_tar = fourier_tr_padding_centered(time_tar,  res, best_x)
            # xfft_tar, fourier_tar = start_stop(xfft_tar, fourier_tar, xfft_tar[0], 0)
            photon_freq_tar = target_freq + xfft_tar
            
            xfft_no_pad, fourier_no_pad = fourier_no_padding(time, waveform)
            freq_abs = self.readout_lo_freq + xfft_no_pad

            mask_neg = (xfft_no_pad <= 0)
            fourier_neg = fourier_no_pad.copy()
            fourier_neg[~mask_neg] = 0
            xfft_neg = xfft_no_pad  # ?????????

            xfft_tar_no_pad, fourier_tar_no_pad = fourier_no_padding(time, res)
            freq_abs_tar = self.readout_lo_freq + xfft_tar_no_pad

            mask_neg_tar = (xfft_tar_no_pad <= 0)
            fourier_neg_tar = fourier_tar_no_pad.copy()
            fourier_neg_tar[~mask_neg_tar] = 0
            xfft_neg_tar =  self.readout_lo_freq + xfft_tar_no_pad      

            fouriers.append(fourier_neg)
            fourier_tars.append(fourier_neg_tar)
            freq_axes.append(xfft_neg_tar)
            
            if mode_id % 2 == 0:
                sign = 1#np.sign(np.vdot(fourier.real, fourier_tar.real))
                fourier     *= sign
                ax_f.plot(photon_freq_tar,  fourier_tar.real, "r", lw=2.5, alpha=0.7, label="Target")
                ax_f.plot(photon_freq,     fourier.real,     "tab:blue", lw=1.2, linestyle="-",  label="Real")
                ax_f.plot(photon_freq,     fourier.imag,     "tab:blue", lw=1.2, linestyle=(0, (2, 1)), label="Imag")
            else:
                sign = -1#np.sign(np.vdot(fourier.imag, fourier_tar.imag))
                fourier     *= sign
                ax_f.plot(photon_freq_tar, fourier_tar.imag, "r", lw=2.5, alpha=0.7, label="Target")
                ax_f.plot(photon_freq,      fourier.real,     "tab:blue", lw=1.2, linestyle="-",  label="Real")
                ax_f.plot(photon_freq,      fourier.imag,     "tab:blue", lw=1.2, linestyle=(0, (2, 1)), label="Imag")
            ax_f.set_xlim(target_freq-0.02, target_freq+0.02)
            ax_f.label_outer()
            if i // ncols == nrows - 1:
                ax_t.set_xlabel("Time (ns)")
                ax_f.set_xlabel("Frequency (GHz)")
            # transmissivity.append(fourier/(fourier_tar/2))

        if legend:
            handles, labels = [], []
            for ax in axes_t:
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)
            unique = dict(zip(labels, handles))

            fig_t.legend(unique.values(), unique.keys(),
                        loc='upper center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.05),
                        handlelength=1.5)
        fig_t.text(0.0, 0.5, "Amplitude (arb. unit)", va="center", rotation="vertical")
        fig_f.text(0.0, 0.55, "Amplitude (arb. unit)", va="center", rotation="vertical")
        self.times = times
        self.signals = signals
        if savefig:
            plt.tight_layout()
            fig_t.savefig("waveforms.pdf",bbox_inches="tight")
            fig_f.savefig("spectra.pdf",bbox_inches="tight")
        plt.tight_layout()
        plt.show()

        nrows = n 
        ncols = 2 
        fig, axes = plt.subplots(nrows, ncols,  figsize=(1.6 * ncols, 0.9 * nrows), sharex=False, sharey=False)
        if nrows == 1:
            axes = axes.reshape(1, 2)
        scale_t = None
        scale_f = None
        for i, (k, v) in enumerate(result_items):
            ax_t = axes[i, 0] 
            ax_f = axes[i, 1]  
            target_freq = v["target_freq"]
            form = v["form"]
            mode_id = v["mode_id"]
            x0 = v["x0"]
            ax_t.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax_f.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

            _, datadict = search_datadict_miyamura(self.data_path, v["date"], acquire_time=v["acquire_time"])
            time, waveform = start_stop(datadict["time"]["values"].ravel(), datadict["waveform"]["values"].ravel(), 0, self.duration+200)
    
            # phase = np.angle(demodulate(time, waveform, self.readout_lo_freq - target_freq))
            lpf = 2 * lowpass(
                time, waveform * np.exp(2j * np.pi * (self.readout_lo_freq - target_freq) * time),
                passing_band, 0.03, 0.1, 90
                )
            index_max = np.argmax(np.abs(lpf))
            phase = np.angle(lpf[index_max])
            print("phase:", phase)
            signal = lpf * np.exp(-1j * phase)
            norm = np.linalg.norm(signal)
            signal = signal/norm
            waveform = waveform/norm
            res, best_x, time_tar = self.fit_waveform(time, signal, self.kappa, x0, form, mode_id, target_waveform, padded=True)
            
            sign = np.sign(np.vdot(res[len(res)-len(time)-1:-1], signal.real))
            print("sign:", sign)

            xfft, fourier = fourier_tr_padding_centered(time, signal, best_x)
            # xfft, fourier = start_stop(xfft, fourier, xfft[0], 0)
            photon_freq = target_freq + xfft
            if i == 0:
                scale_t = np.max(np.abs(waveform))

            ax_t.plot(time_tar, res/scale_t, "r", lw=2.0, alpha=0.7, label="Target")
            ax_t.plot(time, waveform/scale_t, "tab:blue", alpha=0.3, lw=0.5, label="Data")
            ax_t.plot(time, sign*signal.real/scale_t, "tab:blue", lw=1.0, linestyle="-", label="Real")
            ax_t.plot(time, sign*signal.imag/scale_t, "tab:blue", lw=1.0, linestyle="--", label="Imag")
            ax_t.set_xlim(0, self.duration)
            ax_t.set_ylim(-1.2, 1.2)
            ax_t.set_yticks([-1, 0, 1])
            ax_t.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            if i == nrows - 1:
                ax_t.set_xlabel("Time (ns)")

            xfft_tar, fourier_tar = fourier_tr_padding_centered(time_tar,  res, best_x)
            # xfft_tar, fourier_tar = start_stop(xfft_tar, fourier_tar, xfft_tar[0], 0)
            photon_freq_tar = target_freq + xfft_tar
            if i == 0:
                scale_f = np.max(np.abs(fourier.real))
            if mode_id % 2 == 0:
                sign = 1#np.sign(np.vdot(fourier.real, fourier_tar.real))
                fourier *= sign
                ax_f.plot(photon_freq_tar, fourier_tar.real/scale_f, "r", lw=2.0, alpha=0.7, label="Target")
                ax_f.plot(photon_freq, fourier.real/scale_f, "tab:blue", lw=1.0, linestyle="-", label="Real")
                ax_f.plot(photon_freq, fourier.imag/scale_f, "tab:blue", lw=1.0, linestyle="--", label="Imag")
            else:
                sign = -1#np.sign(np.vdot(fourier.imag, fourier_tar.imag))
                fourier *= sign
                ax_f.plot(photon_freq_tar, fourier_tar.imag/scale_f, "r", lw=2.0, alpha=0.7, label="Target")
                ax_f.plot(photon_freq, fourier.real/scale_f, "tab:blue", lw=1.0, linestyle="-", label="Real")
                ax_f.plot(photon_freq, fourier.imag/scale_f, "tab:blue", lw=1.0, linestyle="--", label="Imag")

            ax_f.set_xlim(target_freq - 0.02, target_freq + 0.02)
            ax_f.set_ylim(-1.2, 1.2)
            ax_f.set_yticks([-1, 0, 1])
            ax_f.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            if i == nrows - 1:
                ax_f.set_xlabel("Frequency (GHz)")

        if legend:
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.02))

        fig.text(0.0, 0.5, "Amplitude (arb. unit)", va="center", rotation="vertical")
        fig.text(0.49, 0.5, "Amplitude (arb. unit)", va="center", rotation="vertical")
        plt.tight_layout()
        if savefig:
            fig.savefig("waveform_and_spectrum.pdf", bbox_inches="tight")
        plt.show()

            
        if mode_opt:
            return self.optimize_modes(
                target_freq, photon_freq, freq_axes, fouriers, fourier_tars, transmissivity, passing_band
            )
        else:
            return signals


    def optimize_modes(self, target_freq, photon_freq, freq_axes, fouriers, fourier_tars, transmissivity, passing_band):
        trans_array = np.array(np.abs(transmissivity))
        trans_array_masked = np.where(trans_array > 1.5, np.nan, trans_array)

        fig_trans, ax = plt.subplots(figsize=(4.2, 2.5))
        for i in range(trans_array_masked.shape[0]):
            ax.plot(photon_freq, trans_array_masked[i], alpha=0.6, label=f"Mode {i}")
        mean_trans = np.nanmean(trans_array_masked, axis=0)
        ax.plot(photon_freq, mean_trans, color='tab:green', lw=1.5)
        ax.set_xlim(target_freq - 0.01, target_freq + 0.01)
        ax.set_ylim(-0.1, 1.5)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Mag[data/ target]")
        # ax.set_title("Transmissivity of Each Mode")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


        n = len(fouriers)
        domega = freq_axes[0][1] - freq_axes[0][0]

        # Scattering matrix S
        S = np.zeros((n, n), dtype=complex)
        for l in range(n):
            for k in range(n):
                integrand = np.conj(fouriers[k]) * fouriers[l]
                S[l, k] = np.sum(integrand) * domega

        fig, ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)
        im2 = ax.imshow(np.abs(S)*1e-5, cmap='viridis', origin='lower')
        ax.set_xlabel(r"mode $k$")
        ax.set_ylabel(r"mode $l$")
        fig.colorbar(im2, ax=ax, label=r"$|S_{lk}|$")
        plt.show()

        # Diagonalization
        eigvals, U = np.linalg.eigh(S)
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        U = U[:, sorted_indices]

        plt.figure(figsize=(4, 3))
        plt.plot(np.arange(n), eigvals, 'o-')
        plt.xlabel("Mode")
        plt.ylabel("Eigenvalue")
        plt.title("Transmissivity eigenvalues of scattering matrix")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Construct optimal modes
        optimal_f_w_list = []
        for m in range(n):
            optimal_f_w = sum(U[m, k] * fourier_tars[k] for k in range(n))
            optimal_f_w_list.append(optimal_f_w)

        optimal_xi_t_list = [ifft(fftshift(f_w)) for f_w in optimal_f_w_list]

        fig, axs = plt.subplots(n, 2, figsize=(6, 2*n), constrained_layout=True)
        dt = self.times[0][1] - self.times[0][0]
        new_signals = []

        for m in range(n):
            N = len(optimal_xi_t_list[m])
            t_axis = np.arange(N) * dt
            xi_t = optimal_xi_t_list[m]
            xi_w = optimal_f_w_list[m]

            lpf = lowpass(
                t_axis, xi_t * np.exp(2j * np.pi * (self.readout_lo_freq - target_freq) * t_axis),
                passing_band, 0.03, 0.1, 90
            )
            index_max = np.argmax(np.abs(lpf))
            phase = np.angle(lpf[index_max])
            signal = lpf * np.exp(-1j * phase)
            norm = np.sqrt(np.sum(np.abs(signal)**2) * dt)
            signal = signal / norm
            xi_t = xi_t / norm
            new_signals.append(signal)

            ax_time = axs[m, 0]
            ax_time.plot(t_axis, xi_t, "tab:blue", alpha=0.3, lw=0.5, label="Data")
            ax_time.plot(t_axis, signal.real, "tab:blue", lw=1.2, linestyle="-", label="Real")
            ax_time.plot(t_axis, signal.imag, "tab:blue", lw=1.2, linestyle=(0, (2, 1)), label="Imag")
            ax_time.set_title(f"Mode {m}")
            ax_time.set_xlabel("Time (s)")
            ax_time.set_ylabel("Amplitude (arb. unit)")
            ax_time.grid(True)

            ax_freq = axs[m, 1]
            ax_freq.plot(freq_axes[m], 10 * np.abs(xi_w), "tab:blue", lw=1.2, linestyle="-", label=f"data")
            ax_freq.set_xlabel("Frequency (GHz)")
            ax_freq.set_ylabel("Amplitude (arb. unit)")
            ax_freq.set_xlim(target_freq - 0.01, target_freq + 0.01)
            ax_freq.grid(True)
        self.opt_signals = new_signals
        return new_signals


    def plot_squaredI_matrix(self, mode_num, value=True, opt_mode=False):
        expI_matrix = np.zeros((mode_num, mode_num))
        I_normalized = np.zeros((mode_num, mode_num))
        for m in range(mode_num):
            for m_prime in range(mode_num):
                if opt_mode:
                    wf_m = self.opt_signals[m]
                    wf_m_prime = self.opt_signals[m_prime]
                else:
                    wf_m = self.signals[m]
                    wf_m_prime = self.signals[m_prime]
                expI_matrix[m, m_prime] = overlap(wf_m, wf_m_prime)
                norm_product = np.linalg.norm(wf_m) * np.linalg.norm(wf_m_prime)
                I_normalized[m, m_prime] = expI_matrix[m, m_prime]# / norm_product
        
        squaredI_matrix = I_normalized**2
        plt.figure(figsize=(4, 2))
        plt.imshow(squaredI_matrix, cmap='viridis', origin='lower', vmin=0, vmax=1)
        for i in range(mode_num):
            for j in range(mode_num):
                if i != j and value:
                    if squaredI_matrix[j, i]>0.5:
                        plt.text(i, j, f"{squaredI_matrix[j, i]:.2f}", ha='center', va='center', color="red")
                    else:
                        plt.text(i, j, f"{squaredI_matrix[j, i]:.2f}", ha='center', va='center', color="white")

        plt.colorbar(label=r'Squared overlap $|I_{mm^{\prime}}|^2$')
        plt.rcParams['xtick.major.width'] = 0.9#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 0.9#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 8 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 0.9# 軸の線幅edge linewidth。囲みの太さ
        plt.rcParams['xtick.top'] = True
        plt.rcParams['xtick.bottom'] = True
        plt.rcParams['ytick.left'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams['lines.markersize'] = 2.
        plt.xlabel(r"Photon mode $m^{\prime}$")
        plt.ylabel(r"Photon mode $m$")
        plt.xticks(range(len(squaredI_matrix)))
        plt.yticks(range(len(squaredI_matrix)))
        plt.tight_layout()
        # plt.savefig('Fig2_c.pdf', bbox_inches='tight')
        # print("basis_fidelity:", basis_fidelity(expI_matrix))
        print("separability:", separability(expI_matrix))
        return separability(expI_matrix)[0]
    
    def plot_squaredI_matrix_8(self, mode_num, value=True, opt_mode=False):
        expI_matrix = np.zeros((mode_num, mode_num))
        I_normalized = np.zeros((mode_num, mode_num))
        for m in range(mode_num):
            for m_prime in range(mode_num):
                if opt_mode:
                    wf_m = self.opt_signals[m]
                    wf_m_prime = self.opt_signals[m_prime]
                else:
                    wf_m = self.signals[m]
                    wf_m_prime = self.signals[m_prime]
                expI_matrix[m, m_prime] = overlap(wf_m, wf_m_prime)
                norm_product = np.linalg.norm(wf_m) * np.linalg.norm(wf_m_prime)
                I_normalized[m, m_prime] = expI_matrix[m, m_prime]# / norm_product
        
        squaredI_matrix = I_normalized**2
        plt.figure(figsize=(5, 3.5))
        plt.imshow(squaredI_matrix, cmap='viridis', origin='lower', vmin=0, vmax=1)
        for i in range(mode_num):
            for j in range(mode_num):
                if i < j and value:  # ←ここを変更
                    if squaredI_matrix[j, i] > 0.5:
                        plt.text(i, j, f"{squaredI_matrix[j, i]:.2f}",
                                ha='center', va='center', color="red")
                    else:
                        plt.text(i, j, f"{squaredI_matrix[j, i]:.2f}",
                                ha='center', va='center', color="white")
        plt.colorbar(label=r'Squared overlap $|I_{mm^{\prime}}|^2$')
        plt.rcParams['xtick.major.width'] = 0.9#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 0.9#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 8 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 0.9# 軸の線幅edge linewidth。囲みの太さ
        plt.rcParams['xtick.top'] = True
        plt.rcParams['xtick.bottom'] = True
        plt.rcParams['ytick.left'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams['lines.markersize'] = 2.
        plt.xlabel(r"Photon mode $m^{\prime}$")
        plt.ylabel(r"Photon mode $m$")
        plt.xticks(range(len(squaredI_matrix)))
        plt.yticks(range(len(squaredI_matrix)))
        plt.tight_layout()
        # plt.savefig('Fig2_c.pdf', bbox_inches='tight')
        # print("basis_fidelity:", basis_fidelity(expI_matrix))
        print("separability:", separability(expI_matrix))
        return separability(expI_matrix)[0]
    
    ## for tx
    def _unwrap_phase_downwards(self, phase):
        unwrapped = [phase[0]]
        for i in range(1, len(phase)):
            d = phase[i] - phase[i-1]
            if d > 0:
                unwrapped.append(unwrapped[-1] + d - 2 * np.pi)
            else:
                unwrapped.append(unwrapped[-1] + d)
        # unwrapped = np.array(unwrapped)
        # window_size = 5
        # smooth_unwrapped = np.convolve(unwrapped, np.ones(window_size)/window_size, mode='same')
        return np.array(unwrapped)
    
    def phase_subtraction(self, ctrl_pulse_path, save=False, target_waveform=[]):
        fig, axes = plt.subplots(len(self.result_dict), 3, figsize=(9, 2 * len(self.result_dict)))

        for i, (k, v) in enumerate(self.result_dict.items()):
            ax, ax2, ax3 = axes[i]
            target_freq = v["target_freq"]
            form = v["form"]
            mode_id = v["mode_id"]
            x0 = v["x0"]
            time = self.times[i]
            signal = self.signals[i]

            _, datadict = search_datadict_miyamura(self.data_path, v["date"], acquire_time=v["acquire_time"])
            print("sign should be the same through the mode")
            target_signal = self.fit_waveform(time, signal, self.kappa, x0, form, mode_id, target_waveform)[0]
            ax.plot(time, np.abs(signal), color="tab:blue", lw=3, label="|signal|")
            ax.plot(time, signal.real, color="tab:blue", lw=2, ls="--", alpha=0.7, label="Re(signal)")
            ax.plot(time, signal.imag, color="tab:blue", lw=2, ls=":", alpha=0.7, label="Im(signal)")

            ax.plot(time, np.abs(target_signal), color="tab:orange", lw=3, label="|target|")
            ax.plot(time, target_signal.real, color="tab:orange", lw=2, ls="--", alpha=0.7, label="Re(target)")
            ax.plot(time, target_signal.imag, color="tab:orange", lw=2, ls=":", alpha=0.7, label="Im(target)")
            ax.set_xlabel("Time [ns]")
            ax.set_ylabel("Amplitude")

            phase = np.unwrap(np.angle(signal))
            target_phase = np.unwrap(np.angle(target_signal))
            target_phase_downwards = self._unwrap_phase_downwards(target_phase)
            phase_of_time = interpolate.interp1d(time, phase-target_phase_downwards)
            
            ax2.plot(time, phase, color="tab:blue", lw=2, label="phase (measured)")
            ax2.plot(time, target_phase_downwards, color="tab:orange", lw=2, label="target phase")
            ax2.plot(time, phase - target_phase_downwards, color="tab:green", lw=2, ls="--", label="correction phase")
            ax2.set_xlabel("Time [ns]")
            ax2.set_ylabel("Phase [rad]")
            ax2.grid(True, ls=":", alpha=0.3)

            target_shape_note = load_note(self.data_path, v["date"], f"{_}\\target_shape.md")
            idx = target_shape_note.find("path : ")
            ctrl_pulse_filename = target_shape_note[idx + 7:]

            _, datadict = search_datadict_miyamura(
                ctrl_pulse_path,
                ctrl_pulse_filename[:11],
                name=ctrl_pulse_filename,
            )
            
            ctrl_pulse = datadict["control_pulse"]["values"]
            ctrl_envelope = datadict["control_envelope"]["values"]
            phase_for_correction = phase_of_time(np.arange(len(ctrl_pulse)))
            new_ctrl_pulse = ctrl_pulse * np.exp(1j * phase_for_correction)

            ax3.plot(new_ctrl_pulse.real, color="tab:blue", lw=2, label="New control pulse (real)")
            ax3.plot(ctrl_envelope, color="tab:orange", lw=2, label="Control envelope")
            ax3.set_xlabel("Time [ns]")
            ax3.set_ylabel("Amplitude")
            ax3.grid(True, ls=":", alpha=0.3)
            if i == 0:
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=2, fontsize=9) 
                ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.5), fontsize=9) 
                ax3.legend(loc="upper center", bbox_to_anchor=(0.5, 1.5), fontsize=9) 

            if save:
                data = DataDict(
                    time=dict(unit="ns"),
                    control_pulse=dict(axes=["time"]),
                    control_envelope=dict(axes=["time"])
                )
                data.validate()

                with DDH5Writer(data, ctrl_pulse_path, name=f"Control_Pulse_{k}") as writer:
                    writer.add_tag(["control_pulse", "corrected"])
                    writer.save_text("target_shape.md", "corrected \n" + target_shape_note)
                    writer.add_data(
                        time=np.arange(len(ctrl_pulse)),
                        control_pulse=new_ctrl_pulse,
                        control_envelope=ctrl_envelope
                    )
                T.sleep(1.0)

        plt.tight_layout()
        plt.show()
        return phase_for_correction
    
    ## for rx
    def phase_subtraction_and_pulse_flip(self, ctrl_pulse_path, save=False):
        fig, axes = plt.subplots(len(self.result_dict), 3, figsize=(9, 2 * len(self.result_dict)))

        for i, (k, v) in enumerate(self.result_dict.items()):
            ax, ax2, ax3 = axes[i]
            target_freq = v["target_freq"]
            form = v["form"]
            mode_id = v["mode_id"]
            x0 = v["x0"]
            time = self.times[i]
            signal = self.signals[i]

            _, datadict = search_datadict_miyamura(self.data_path, v["date"], acquire_time=v["acquire_time"])
            
            target_signal = self.fit_waveform(time, signal, self.kappa, x0, form, mode_id)[0]
            ax.plot(time, np.abs(signal), color="tab:blue", lw=3, label="|signal|")
            ax.plot(time, signal.real, color="tab:blue", lw=2, ls="--", alpha=0.7, label="Re(signal)")
            ax.plot(time, signal.imag, color="tab:blue", lw=2, ls=":", alpha=0.7, label="Im(signal)")

            ax.plot(time, np.abs(target_signal), color="tab:orange", lw=3, label="|target|")
            ax.plot(time, target_signal.real, color="tab:orange", lw=2, ls="--", alpha=0.7, label="Re(target)")
            ax.plot(time, target_signal.imag, color="tab:orange", lw=2, ls=":", alpha=0.7, label="Im(target)")
            ax.set_xlabel("Time [ns]")
            ax.set_ylabel("Amplitude")

            phase = np.unwrap(np.angle(signal))
            phase_of_time = interpolate.interp1d(time, phase)

            ax2.plot(time, phase, color="tab:blue", lw=2, label="phase (measured)")
            ax2.plot(time, np.zeros(len(time)), color="tab:orange", lw=2, label="target phase")
            ax2.plot(time, phase, color="tab:green", lw=2, ls="--", label="correction phase")
            ax2.set_xlabel("Time [ns]")
            ax2.set_ylabel("Phase [rad]")
            ax2.grid(True, ls=":", alpha=0.3)

            target_shape_note = load_note(self.data_path, v["date"], f"{_}\\target_shape.md")
            idx = target_shape_note.find("path : ")
            ctrl_pulse_filename = target_shape_note[idx + 7:]

            _, datadict = search_datadict_miyamura(
                ctrl_pulse_path,
                ctrl_pulse_filename[:11],
                name=ctrl_pulse_filename,
            )
            
            ctrl_pulse = datadict["control_pulse"]["values"]
            ctrl_envelope = datadict["control_envelope"]["values"]
            zero_points = self.find_zero_point(ctrl_envelope)
            phase_for_correction = phase_of_time(np.arange(len(ctrl_pulse)))
            new_ctrl_pulse = ctrl_pulse * np.exp(1j * phase_for_correction)
            flipped_envelope = self.flip_control_envelope(ctrl_envelope, zero_points)
            flipped_ctrl_pulse = self.flip_control_envelope(new_ctrl_pulse, zero_points)

            ax3.plot(new_ctrl_pulse.real, color="tab:blue", lw=2, label="New control pulse (real)")
            ax3.plot(ctrl_envelope, color="tab:orange", lw=2, label="Control envelope")
            ax3.set_xlabel("Time [ns]")
            ax3.set_ylabel("Amplitude")
            ax3.grid(True, ls=":", alpha=0.3)

            if i == 0:
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=2, fontsize=9) 
                ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.5), fontsize=9) 
                ax3.legend(loc="upper center", bbox_to_anchor=(0.5, 1.5), fontsize=9) 

            if save:
                data = DataDict(
                    time=dict(unit="ns"),
                    control_pulse=dict(axes=["time"]),
                    control_pulse_unflip=dict(axes=["time"]),
                    control_envelope=dict(axes=["time"]),
                    control_envelope_unflip=dict(axes=["time"]),
                )
                data.validate()

                with DDH5Writer(data, ctrl_pulse_path, name=f"Control_Pulse_{k}") as writer:
                    writer.add_tag(["control_pulse", "corrected"])
                    writer.save_text("target_shape.md", "corrected \n" + target_shape_note)
                    writer.add_data(
                        time=np.arange(len(flipped_ctrl_pulse)),
                        control_pulse=flipped_ctrl_pulse,
                        control_pulse_unflip=new_ctrl_pulse,
                        control_envelope=flipped_envelope,
                        control_envelope_unflip=np.abs(new_ctrl_pulse)
                    )
                T.sleep(1.0)

        plt.tight_layout()
        plt.show()

    def find_zero_point(self, envelope):
        diff_envelope = np.diff(envelope)
        sign_change = np.diff(np.sign(diff_envelope))

        inflection_points = np.where(sign_change == 2)[0] 
        
        return inflection_points

    def flip_control_envelope(self, control_envelope, zero_points):
        flipped_envelope = control_envelope.copy()
        
        for i in range(len(zero_points)):  
            if i % 2 == 0:
                start = zero_points[i] + 1
                end = zero_points[i+1] + 1 if i+1 < len(zero_points) else len(control_envelope)
                flipped_envelope[start:end] *= -1  
        
        return flipped_envelope

    def generate_control_pulse_rx(self, ctrl_pulse_path, save=False):
        fig, axes = plt.subplots(len(self.result_dict), 1, figsize=(4, 2 * len(self.result_dict)))

        for i, (key, params) in enumerate(self.result_dict.items()):
            ax = axes[i]
            date = params["date"]
            acquire_time = params["acquire_time"]
            form = params["form"]

            _, datadict = search_datadict_miyamura(self.data_path, date, acquire_time=acquire_time)
   
            target_shape_note = load_note(self.data_path,date, f"{_}\\target_shape.md")
            idx = target_shape_note.find("path : ")
            ctrl_pulse_filename = target_shape_note[idx + 7:]

            _, datadict = search_datadict_miyamura(
                ctrl_pulse_path,
                ctrl_pulse_filename[:11],
                name=ctrl_pulse_filename,
            )
            
            ctrl_pulse = datadict["control_pulse_unflip"]["values"]
            env = datadict["control_envelope_unflip"]["values"]
            phase = np.unwrap(np.angle(ctrl_pulse))
            env_rvs = np.flip(env)
            if i == 0:
                zero_points = []
            else:
                zero_points = self.find_zero_point(env_rvs)
            flipped_envelope = self.flip_control_envelope(env_rvs, zero_points)

            phase_rvs = np.flip(phase)
            inst_omega_minus = -np.diff(phase_rvs)
            phase_recieve = np.zeros(phase.shape)
            for i in range(1, len(phase_recieve)):
                phase_recieve[i] = phase_recieve[i-1] + inst_omega_minus[i-1]

            recieve_pulse = flipped_envelope * np.exp(1j * phase_recieve)

            target_shape_note += f"\nRecieveVersion of {key}"
            
            ax.plot(np.arange(len(recieve_pulse)), recieve_pulse.real, label=f"Control pulse rx")
            ax.plot(np.arange(len(recieve_pulse)), flipped_envelope, label=f"Control envelope rx")
            ax.legend()

            if save:
                data = DataDict(
                    time=dict(unit="ns"),
                    control_pulse=dict(axes=["time"]),
                    control_envelope=dict(axes=["time"])
                )
                data.validate()
                
                with DDH5Writer(data, ctrl_pulse_path, name="Recieve_Pulse") as writer:
                    writer.add_tag(["recieve_pulse", form])
                    writer.save_text("target_shape.md", target_shape_note)
                    writer.add_data(
                        time=np.arange(len(recieve_pulse)),
                        control_pulse=recieve_pulse,
                        control_envelope=flipped_envelope
                    )
                T.sleep(1)
                print(f"Saved Recieve_Pulse for {key} at {ctrl_pulse_path}")
        plt.tight_layout()
        plt.show()
    
    def roll_with_interpolation(self, array, shift):
        return nd_shift(array, shift=shift, mode='constant', cval=0.0)


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
            # plt.plot(data_tx_norm)
            # plt.plot(np.conjugate(shifted_data_rx))
            # plt.show()
            new_func = -data_tx_norm * np.conjugate(shifted_data_rx)
            y = np.abs(np.sum(new_func) * 2)**2
            ys.append(y)

        return np.array(ys) * 100

    def overlap_curve_fitting(self, data, signal_ph_tx, signal_ph_rx, taus):
        def overlap_curve(taus, delay):
            def new_func(taus, delay):
                delay = int(round(delay))
                return self.overlap_comm(signal_ph_tx, signal_ph_rx, taus + delay)
                # return self.overlap_comm_delay_dep(signal_ph_rx, taus, delay)
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
                # return self.overlap_comm(signal_ph_tx, signal_ph_rx, taus + delay)
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
    
    def plot_er_ea(self, ax=None, label=None, color=None):
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5)) 
            ax.set_xlabel(r'F0g1 delay $\tau$ (ns)', size="large")
            ax.set_ylabel('Energy', size="large")
            
            plt.tick_params(pad=10, top=True, bottom=True, left=True, right=True)
            plt.rcParams["xtick.direction"] = 'in'
            plt.rcParams["ytick.direction"] = 'in'
            plt.tight_layout()
        ax[0].plot(np.ravel(self.results['fogi_delay']), np.ravel(self.results['energy_bases']), 
                "o", label=label, color=color)
        ax[1].plot(np.ravel(self.results['fogi_delay']), np.ravel(self.results['energys']), 
                "o", label=label, color=color)
        return ax
    
    def plot_energy_imag(self, ax=None, label=None, color=None):
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(6 , 5)) 
            ax.set_xlabel(r'F0g1 delay $\tau$ (ns)', size="large")
            ax.set_ylabel('Energy imag', size="large")
            
            plt.tick_params(pad=10, top=True, bottom=True, left=True, right=True)
            plt.rcParams["xtick.direction"] = 'in'
            plt.rcParams["ytick.direction"] = 'in'
            plt.tight_layout()
        ax.plot(np.ravel(self.results['fogi_delay']), np.ravel(self.results['energy_imag']), 
                "o", label=label, color=color)
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
    
    def plot_waveform(self, ax=None, title=None, ph_amp_idx=None, fogi_freq_idx=None, show_legend=False, show_title=False, color="tab:blue", mode=0, show_absorbed=True, norm =False):
        if ax is None:
            fig, ax = plt.subplots()
        
        time_data = self.results['time'][ph_amp_idx][fogi_freq_idx][0:501]
        y_reflect = self.results['y_base_signals'][ph_amp_idx][fogi_freq_idx][100:601] * 1e3
        y_absorb = self.results['y_absorbed_signals'][ph_amp_idx][fogi_freq_idx][100:601] * 1e3
        if mode%2==0:
            sign = 1
        else:
            sign = -1
        norm_waveform = 1
        # if norm:
        #     norm_waveform = np.max(np.abs(y_reflect))
        ax.plot(time_data, sign*y_reflect.real/norm_waveform, lw=2, alpha=0.7, color=color, label="w/o fogi")
        ax.plot(time_data, sign*y_reflect.imag/norm_waveform, lw=2, alpha=0.7, color=color, linestyle="--",label="w/ fogi")
        # ax.plot(time_data, sign*y_absorb/norm_waveform, lw=1.5, color="black", linestyle=":",label="w/ fogi")
        if show_absorbed:
            sign_abs = np.sign(np.real(np.vdot(y_reflect, y_absorb)))
            ax.plot(time_data, sign_abs*sign*y_absorb/norm_waveform, lw=1.5, color="black", linestyle=":",label="w/ fogi")
        
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlim(0, 1000)

        if show_legend:
            ax.legend()
        if title:
            ax.set_title(title, fontsize=9 if show_title else 0)
        plt.rcParams["xtick.direction"] = 'in'
        plt.rcParams["ytick.direction"] = 'in'
        plt.rcParams['xtick.major.width'] = 0.9#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 0.9#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 8 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 0.9# 軸の線幅edge linewidth。囲みの太さ
        plt.rcParams['xtick.top'] = True
        plt.rcParams['xtick.bottom'] = True
        plt.rcParams['ytick.left'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams['lines.markersize'] = 2.
        return ax, time_data, sign*y_absorb

    def plot_waveform_8(self, ax=None, title=None, ph_amp_idx=None, fogi_freq_idx=None, show_legend=False, show_title=False, color="tab:blue", mode=0, show_absorbed=True, norm =False):
        if ax is None:
            fig, ax = plt.subplots()
        
        time_data = self.results['time'][ph_amp_idx][fogi_freq_idx]
        y_reflect = self.results['y_base_signals'][ph_amp_idx][fogi_freq_idx]* 1e3
        y_absorb = self.results['y_absorbed_signals'][ph_amp_idx][fogi_freq_idx]* 1e3
        if mode%2==0:
            sign = 1
        else:
            sign = -1
        norm_waveform = 1
        # if norm:
        #     norm_waveform = np.max(np.abs(y_reflect))
        ax.plot(time_data, sign*y_reflect.real/norm_waveform, lw=2, alpha=0.7, color=color, label="w/o fogi")
        ax.plot(time_data, sign*y_reflect.imag/norm_waveform, lw=2, alpha=0.7, color=color, linestyle="--",label="w/ fogi")
        # ax.plot(time_data, sign*y_absorb/norm_waveform, lw=1.5, color="black", linestyle=":",label="w/ fogi")
        if show_absorbed:
            sign_abs = np.sign(np.real(np.vdot(y_reflect, y_absorb)))
            ax.plot(time_data, sign_abs*sign*y_absorb/norm_waveform, lw=1.5, color="black", linestyle=":",label="w/ fogi")
        
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlim(0, 2500)

        if show_legend:
            ax.legend()
        if title:
            ax.set_title(title, fontsize=9 if show_title else 0)
        plt.rcParams["xtick.direction"] = 'in'
        plt.rcParams["ytick.direction"] = 'in'
        plt.rcParams['xtick.major.width'] = 0.9#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 0.9#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 8 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 0.9# 軸の線幅edge linewidth。囲みの太さ
        plt.rcParams['xtick.top'] = True
        plt.rcParams['xtick.bottom'] = True
        plt.rcParams['ytick.left'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams['lines.markersize'] = 2.
        return ax, time_data, sign*y_absorb



def load_datadict(data_path, date, acquire_time, name=None):
    _, datadict = search_datadict_miyamura(data_path, date, acquire_time=acquire_time, name=name)
    return datadict

def s11(f,f_r,k_ex,k_in,phi,A,C,Ed,alpha):
        K_ex = 2*np.pi*k_ex
        K_in = 2*np.pi*k_in
        K_tot = 2*np.pi*(k_ex + k_in)
        ideal = ((1 - np.exp(1j*phi) / 2)*K_ex - K_in/2 - 2j*np.pi*(f-f_r))/((K_tot)/2 + 2j*np.pi*(f - f_r))
        env = A*np.exp(1j*(alpha - 2*np.pi*(f-f_r)*Ed))
        return env * (ideal + (C)*(f-f_r))

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

def fourier_tr_padding_centered(x, y, center_x, n_padding=500):
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

def Gaussian(x, sigma, center):
    return np.exp(-(x-center)**2/2/sigma**2)/np.sqrt(2*np.pi)/sigma
def guess_params_gaussian(data,x):
    par_ini = {"sigma":0.05, "center":0,}
    par_max = {"sigma":10, "center":10,}
    par_min = {"sigma":0, "center":-10,}
    par_vary = {"sigma":True, "center":True,}
    return par_ini,par_max,par_min,par_vary
def gaussian_fit(data, x):
    model = lmf.Model(Gaussian)
    params = model.make_params()
    par_ini,par_max,par_min,par_vary=guess_params_gaussian(data,x)
    for name in params:
        params[name].set(
            value=par_ini[name],  # 初期値
            min=par_min[name],  # 下限値
            max=par_max[name],  # 上限値
            vary=par_vary[name] # パラメータを動かすかどうか
        )
    result=model.fit(data, x=x, params=params, method='leastsq')
    return result

def polynomial(x, a0, a1, a2, a3, a4, a5, a6):
    return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5 + a6*x**6
def guess_params_pl():
    par_ini = {'a0': 0,'a1': 0,'a2': 0,'a3': 0,'a4': 0,'a5': 0,'a6': 0,}
    par_max = {'a0': 1,'a1': np.inf,'a2': np.inf,'a3': np.inf,'a4': np.inf,'a5': np.inf,'a6': np.inf,}
    par_min = {'a0': 0,'a1': 0,'a2': 0,'a3': -np.inf,'a4': 0,'a5': -np.inf,'a6': 0,}
    par_vary = {'a0': 0,'a1': 0,'a2': 1,'a3': 0,'a4': 1,'a5': 0,'a6':1,}
    return par_ini,par_max,par_min,par_vary
def polynomial_fit(data, x,):
    model = lmf.Model(polynomial)
    params = model.make_params()
    par_ini,par_max,par_min,par_vary=guess_params_pl()
    for name in params:
        params[name].set(
            value=par_ini[name],  # 初期値
            min=par_min[name],  # 下限値
            max=par_max[name],  # 上限値
            vary=par_vary[name] # パラメータを動かすかどうか
        )
    result=model.fit(data, x=x, params=params, method='leastsq')
    delm = result.eval_uncertainty(sigma=1)
    print(result.params.valuesdict())
    return result

def polynomial_even(x, a0, a2, a4, a6, a8, a10):
    return a0 + a2*x**2 + a4*x**4 + a6*x**6 + a8*x**8 + a10*x**10
def guess_params_plev():
    par_ini = {'a0': 0, 'a2': 0, 'a4': 0, 'a6': 0, 'a8': 0, 'a10':0}
    par_max = {'a0': 1,'a2': np.inf,'a4': np.inf,'a6': np.inf,'a8': np.inf,'a10': np.inf,}
    par_min = {'a0': 0,'a8': 0,'a2': 0,'a10': 0,'a4': 0,'a6': 0,}
    par_vary = {'a0': 0, 'a2': 1, 'a4': 1, 'a6': 1, 'a8': 1, 'a10':1,}
    return par_ini,par_max,par_min,par_vary
def polynomial_even_fit(data, x,):
    model = lmf.Model(polynomial_even)
    params = model.make_params()
    par_ini,par_max,par_min,par_vary=guess_params_plev()
    for name in params:
        params[name].set(
            value=par_ini[name],  # 初期値
            min=par_min[name],  # 下限値
            max=par_max[name],  # 上限値
            vary=par_vary[name] # パラメータを動かすかどうか
        )
    result=model.fit(data, x=x, params=params, method='leastsq')
    delm = result.eval_uncertainty(sigma=1)
    print(result.params.valuesdict())
    return result

def s11_ge(f, f_r, k_ex, k_in, kai):
    k_tot = k_ex + k_in
    s11_g = 1 - k_ex/(k_tot/2 - 1j*(f-f_r))
    s11_e = 1 - k_ex/(k_tot/2 - 1j*(f-f_r-kai))
    return (s11_g/s11_e).conj()
def guess_params2(data,x):
    par_ini = {"f_r":10.51, "k_ex":0.04, "k_in":1e-3, "kai":-6e-3,}
    par_max = {"f_r":10.6, "k_ex":0.08, "k_in":10e-3, "kai":0e-3,}
    par_min = {"f_r":10.45, "k_ex":0.02, "k_in":0, "kai":-10e-3,}
    par_vary = {"f_r":True, "k_ex":True, "k_in":True, "kai":True,}
    return par_ini,par_max,par_min,par_vary
def probe_resonator_fit(data, x):
    model = lmf.Model(s11_ge)
    params = model.make_params()
    par_ini,par_max,par_min,par_vary=guess_params2(data,x)
    for name in params:
        params[name].set(
            value=par_ini[name],  # 初期値
            min=par_min[name],  # 下限値
            max=par_max[name],  # 上限値
            vary=par_vary[name] # パラメータを動かすかどうか
        )
    result=model.fit(data, f=x, params=params, method='leastsq')
    return result

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

def decay(t, gamma, A, off):
    return A * np.exp(-(t-t[0])*gamma) + off
def guess_params_decay(data, t):
    off_ini = np.mean(data[-5:])#data[-1]
    moving_average = np.convolve(data, np.ones(25)/25, mode="valid")
    A_ini = moving_average[0]#np.sign(data[0]-data[-1])*(max(data) - min(data))
    par_ini = {
        'gamma': abs((moving_average[0]-moving_average[5])/(t[0]-t[5]))/A_ini,#-(data[0]-data[2])/(t[0]-t[2]) ,
        'A': A_ini,'off': off_ini
    }
    par_max = {'gamma': 1,'A': 2*A_ini,'off': 5*np.abs(off_ini)}
    par_min = {'gamma': 0,'A': 0.5*A_ini,'off': -5*np.abs(off_ini)}
    par_vary = {'gamma': True,'A': True,'off': True,}
    return par_ini,par_max,par_min,par_vary
def decay_fit(data, t):
    decay_model = lmf.Model(decay)
    params_decay = decay_model.make_params()
    decay_par_ini,decay_par_max,decay_par_min,decay_par_vary=guess_params_decay(data,t)
    for name in params_decay:
        params_decay[name].set(
            value=decay_par_ini[name],  # 初期値
            min=decay_par_min[name],  # 下限値
            max=decay_par_max[name],  # 上限値
            vary=decay_par_vary[name] # パラメータを動かすかどうか
        )
    result=decay_model.fit(data, t=t, params=params_decay, method='leastsq')
    delm = result.eval_uncertainty(sigma=1)
    return result

def load_note(data_path, date, name):
    lines = []
    with open(f'{data_path}\\{date}\\{name}', encoding='utf-8') as f:
        lines = f.readlines()  
    return"".join(lines)

def Watt_dBm(watt):
    """Watt to dBm"""
    return 10 * np.log10(watt / 1e-3)
def dBm_Watt(dBm):
    """dBm to Watt, return unit is watt"""
    return 10 ** (dBm/10) * 1e-3

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

from scipy.linalg import sqrtm
def basis_fidelity(I):
    M = I.shape[0]
    rho = I / np.trace(I)
    sigma = np.identity(M) / M
    sqrt_rho = sqrtm(rho)
    inner = sqrtm(sqrt_rho @ sigma @ sqrt_rho)
    F = (np.trace(inner))**2
    return np.real(F)

def separability(I):
    """Compute separability from overlap matrix I."""
    M = I.shape[0]
    S_list = []
    for m in range(M):
        numerator = np.abs(I[m, m])**2
        denominator = np.sum(np.abs(I[m, :])**2)
        S_list.append(numerator / denominator)
    return np.mean(S_list), S_list 
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

