import numpy as np
import matplotlib.pyplot as plt
import lmfit as lmf
from lmfit.models import (
    LorentzianModel, GaussianModel, SineModel, LinearModel, ConstantModel
)
import scipy.signal as sg
from scipy.special import erfc
from scipy import interpolate
from sklearn.decomposition import PCA
from datataking import search_datadict_miyamura
from scipy.integrate import quad
from scipy.special import spence
from mpmath import *


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

    def extract_decay_rate_and_freq(self, fit_start=30, fp=0.02, fs=0.05, gpass=1, gstop=90, plot=False):
        """
        return: decay rate, photon_freq, (stderrs, photon_frequencies)
        """
        photon_frequencies, arg_max_frequencies, fourier_amps = [], [], []
        decay_rates, stderrs, _ = [], [], []
        if plot:
            fig, axes = plt.subplots(len(self.waveforms), 1, figsize=(10, len(self.waveforms) * 3))
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
                axes[i].plot(self.time, waveform, label="Waveform")
                axes[i].plot(t, env, label="Envelope")
                axes[i].plot(t, result.best_fit, label="Fit")
                axes[i].legend()
                axes[i].set_title(f"Waveform {i + 1}")
                axes[i].set_xlabel("Time")
                axes[i].set_ylabel("Amplitude")
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
        # center = (time[0] + time[-1]) / 2
        # def squared_sech(x):
        #     def func(var):
        #         return 1/np.cosh(const*var)**2
        #     norm_func = func(x)/quad(func, time[0], time[-1])[0]
        #     return norm_func
        # def squared_anti_sech(x):
        #     def func(var):
        #         return (var/np.cosh(const*var))**2
        #     norm_func = func(x)/quad(func, time[0], time[-1])[0]
        #     return norm_func
        # if form=="sech":
        #     top = (const/2)/(np.cosh(const*(time-center)))**2
        #     bottom = 1-(1/2)*(np.tanh(const*(time-center))-np.tanh(const*(time[0]-center))) ####### Sunada changed
        #     # top = squared_sech(t-center) #(const/2)/(np.cosh(const*(time-center)))**2
        #     # bottom = 1- quad(squared_sech, center, time)[0]** 2
        # elif form=="gaussian":
        #     normalized_time = (time-center)/const
        #     top = np.exp(-normalized_time**2)/const/np.sqrt(np.pi)
        #     bottom = 1-(1/2)*erfc(-normalized_time)
        # elif form=="square":
        #     top = time
        #     bottom = time*(time[-1]-time[0])
        # elif form=="anti_sech":                                                             ####### Sunada added
        #     top = (6*const**3/np.pi**2)*(time-center)**2/(np.cosh(const*(time-center)))**2
        #     def f(t):
        #         return 6*(spence(1+np.exp(-2*t))+t*(-t-2*np.log(np.exp(-2*t)+1)+t*np.tanh(t)))/np.pi**2
        #     bottom = 1-(f(const*(time-center))-f(const*(time[0]-center)))
        # if plot:
        #     plt.figure(figsize=(3,1))
        #     plt.plot(time, top/bottom/2/np.pi)
        #     plt.show()
        # return top/bottom/2/np.pi
        center = (time[0] + time[-1]) / 2
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
        if form=="sech_0":
            top = (const/2)/(np.cosh(const*(time-center)))**2
            bottom = 1-(1/2)*(np.tanh(const*(time-center))-np.tanh(const*(time[0]-center)))
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

    def get_values(self, datadict):
        time = np.unique(datadict["time"]["values"])
        fogi_freqs = np.unique(datadict["fogi_frequency"]["values"])
        waveforms = datadict["waveform"]["values"].reshape(len(fogi_freqs), len(time))
        return waveforms, time, fogi_freqs

    def generate_results(self, lo_freq):
        for k, datadict in self.datadicts.items():
            waveforms, time, fogi_freqs = self.get_values(datadict)
            obj = WaveformAnalysis(waveforms, time, fogi_freqs, lo_freq=lo_freq, freq_relation="RF=LO-IF") 
            decay_rate, photon_freqs, _ = obj.extract_decay_rate_and_freq(self.params[k]["fit_start"], 
                                            self.params[k]["fp"], 
                                            self.params[k]["fs"],
                                            self.params[k]["gpass"],
                                            self.params[k]["gstop"],
                                            self.params[k]["plot"])
            if self.T2e is not None: self.results_dict[k]["decay_rates"] = decay_rate * 2  - 1 / self.T2e
            else: self.results_dict[k]["decay_rates"] = decay_rate * 2
            self.results_dict[k]["stderrs"] = _[0]
            self.results_dict[k]["fogi_freqs"] = fogi_freqs
            self.results_dict[k]["photon_freqs"] = photon_freqs

    def plot_decayrates(self):
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        for k, v in self.results_dict.items():
            ax.plot(v["fogi_freqs"], v["decay_rates"], label=f"amp {self.amps_dict[k]}")
            ax2.plot(v["photon_freqs"], v["decay_rates"], label=f"amp {self.amps_dict[k]}")
        ax.set_xlabel('fogi freq (GHz)')
        ax.set_ylabel('$\Gamma_f $ (MHz)')
        ax2.set_xlabel('photon freq (GHz)')
        ax2.set_ylabel('$\Gamma_f $ (MHz)')
        ax2.legend(bbox_to_anchor=(1.6, 1.1), loc='upper right')
        ax.grid()
        ax2.grid()
        plt.show()

    def generate_control_pulse(self, fogi_lo, target_freq, duration, const, form="sech", plot=True, tr=True):
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
        # print(Gammas, amps)
        res = polynomial_even_fit(x=amps, data=Gammas)
        c = tuple(res.params.valuesdict().values())
        x = np.linspace(0, amps[-1], 10001)
        Gamma_to_amp = interpolate.interp1d(polynomial_even(x, c[0], c[1], c[2], c[3], c[4], c[5]),
                                             x, bounds_error=False, fill_value=(0, amps[-1]))
        AWGamp_of_time = lambda time:Gamma_to_amp(self.gamma_t(time, const=const, form=form))

        if tr:                                              ####### changed by Sunada
            envelope = AWGamp_of_time(time)[::-1]
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
            ax.plot(time, envelope)
            ax.set_title('AWGamplitude vs time')
            ax2.plot(time, 1/np.cosh(const * (time - duration/2)))
            # print(time, const, duration)
            ax2.set_title('target shape')
            plt.show()

        # fogi frequency
        fogi_freq_of_AWGamp = interpolate.interp1d(amps, fogi_freq_shifted, bounds_error=False, 
                                                   fill_value=(fogi_freq_shifted[0], fogi_freq_shifted[-1]))
        
        def fogi_freq_of_time(time):
            if tr:
                f_freq = fogi_freq_of_AWGamp(AWGamp_of_time(time)[::-1])
            else:
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
        # if form == "anti_sech":                                 ### changed by Sunada
        #     for i in range(int(len(time)/2), len(time)):
        #         envelope[i] = -envelope[i]
        waveform_at_AWG = envelope*np.exp(2j * np.pi * phase_of_time) 
        if plot:
            plt.figure(figsize=(4, 3))
            plt.plot(time,waveform_at_AWG.real)
            plt.plot(time,waveform_at_AWG.imag)
            plt.plot(time,envelope)
            plt.show()
        return waveform_at_AWG, envelope


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