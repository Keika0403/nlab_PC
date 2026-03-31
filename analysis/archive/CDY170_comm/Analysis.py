import numpy as np
import matplotlib.pyplot as plt
import lmfit as lmf
from lmfit.models import (
    LorentzianModel, GaussianModel, SineModel, LinearModel, ConstantModel
)
import scipy.signal as sg
from scipy import interpolate, optimize
from scipy.ndimage import shift as nd_shift
from sklearn.decomposition import PCA
from datataking import search_datadict_miyamura
from scipy.integrate import quad
from mpmath import *
import time as T
from plottr.data.datadict_storage import DataDict, DDH5Writer, datadict_from_hdf5
from scipy import signal


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
            ncols = 3
            nrows = (len(self.waveforms) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))
            axes = axes.flatten()
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
                axes[i].plot(self.time, waveform, label="Waveform")
                axes[i].plot(t, env, label="Envelope")
                axes[i].plot(t, result.best_fit, label="Fit")
                axes[i].legend()
                axes[i].set_title(f"Waveform {i}")
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

class SechPhotonAnalysis:
    def __init__(self, data_path, result_dict, duration=1000, const=2.5e-3 * 2 * np.pi, readout_lo_freq=9.47):
        self.data_path = data_path
        self.result_dict = result_dict
        self.duration = duration
        self.const = const
        self.readout_lo_freq = readout_lo_freq

    @staticmethod
    def sech_curve(x, time, kappa, order):
        t_shift = time - x[1]
        if order == 0:
            return x[0] / np.cosh(kappa * t_shift)
        elif order == 1:
            return np.abs(x[0] * t_shift / np.cosh(kappa * t_shift))
        elif order == 2:
            return np.abs(x[0] * ((t_shift**2 - (np.pi / (2 * kappa))**2 / 3) / np.cosh(kappa * t_shift)))
        elif order == 3:
            return np.abs(x[0] * ((t_shift**3 - (7 * t_shift * (np.pi / (2 * kappa))**2 / 5)) / np.cosh(kappa * t_shift)))
        return None

    def fit_waveform(self, time, data, x0=[2e-3, 460], get_center=False, curve='sech_0'):
        order = int(curve.split('_')[-1])
        kappa = self.const

        cost_func = lambda x: np.linalg.norm(self.sech_curve(x, time, kappa, order) - data)
        best_x = optimize.minimize(cost_func, x0=x0, method='Nelder-Mead').x

        if get_center:
            return best_x, self.sech_curve(best_x, time, kappa, order)
        return self.sech_curve(best_x, time, kappa, order)

    @staticmethod
    def iq_symmetry(wave, time):
        wave_rev = np.flip(wave).conj()
        norm = np.sum(np.abs(wave)**2)**2
        c = np.zeros(len(wave))

        for i in range(len(wave)):
            wave_rev_i = np.roll(wave_rev, i)
            cand = np.abs(np.sum(wave * wave_rev_i))**2
            c[i] = cand / norm

        return np.max(c)

    def target_overlap(self, wave, time, curve='sech_0'):
        order = int(curve.split('_')[-1])
        kappa = self.const
        t_center = time[-1] / 2

        target = self.sech_curve([1, t_center], time, kappa, order)
        norm1 = np.sqrt(np.sum(np.abs(wave)**2) * (time[1] - time[0]))
        norm2 = np.sqrt(np.sum(np.abs(target)**2) * (time[1] - time[0]))

        m1_t = np.abs(wave / norm1)
        m2_t = target / norm2
        m_2t_star_rev = np.flip(m2_t).conj() 
        m=0
        for i in range(len(wave)):
            m_t_star_rev_i = np.roll(m_2t_star_rev, i)
            mult = m1_t * m_t_star_rev_i
            mult_sum=np.sum(mult) * (time[1] - time[0])
            cand = np.abs(mult_sum)**2
            if m<cand:
                m=cand
        return m
    
    def analyze_waveforms(self, passing_band):
        fig, axes = plt.subplots(len(self.result_dict), 2, figsize=(8, 2 * len(self.result_dict)))

        for i, (k, v) in enumerate(self.result_dict.items()):
            ax, ax2 = axes[i]
            target_freq = v["target_freq"]
            form = v["form"]

            _, datadict = search_datadict_miyamura(self.data_path, v["date"], acquire_time=v["acquire_time"])
            time, waveform = start_stop(datadict["time"]["values"].ravel(), datadict["waveform"]["values"].ravel(), 0, self.duration)

            xfft, fourier = fourier_tr_padding(time, waveform)
            xfft, fourier = start_stop(xfft, fourier, xfft[0], 0)
            photon_freq = self.readout_lo_freq + xfft

            ax.plot(photon_freq, 10 * np.abs(fourier), label=f"data ({form})")
            ax.vlines(target_freq, 0, 10 * max(np.abs(fourier)) * 1.1, color="r", ls="--", label="target")
            ax.set_xlim(9.32, 9.38)
            ax.set_xlabel("Photon Frequency (GHz)")
            ax.set_ylabel("Amplitude (a.u.)")
            ax.legend()

            phase = np.angle(demodulate(time, waveform, self.readout_lo_freq - target_freq))
            signal = 2 * lowpass(
                time, waveform * np.exp(2j * np.pi * (self.readout_lo_freq - target_freq) * time),
                passing_band, 0.03, 0.1, 90
                ) * np.exp(-1j * phase)
            res = self.fit_waveform(time, np.abs(signal), curve=f"{form}")

            ax2.plot(time, 1000 * waveform, "tab:blue", lw=1, alpha=0.5)
            ax2.plot(time, 1000 * np.abs(signal), "tab:blue", lw=3, label=f"data ({form})")
            ax2.plot(time, 1000 * res, "r--", lw=2, label="target", alpha=1)
            ax2.set_xlabel("Time (ns)")
            ax2.set_ylabel("Amplitude (a.u.)")
            ax2.set_xlim(0, self.duration)
            ax2.legend()

            symmetry = self.iq_symmetry(signal, time)
            overlap = self.target_overlap(signal, time, curve=f"{form}")
            print(f"Form: {form}, Symmetry: {symmetry}, Overlap: {overlap}")

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

    def generate_control_pulse_rx(self, ctrl_pulse_path):
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
            
            ctrl_pulse = datadict["control_pulse"]["values"]
            env = datadict["control_envelope"]["values"]
            phase = np.unwrap(np.angle(ctrl_pulse))
            env_rvs = np.flip(env)
            zero_points = self.find_zero_point(env_rvs)
            flipped_envelope = self.flip_control_envelope(env_rvs, zero_points)

            phase_rvs = np.flip(phase)
            inst_omega_minus = -np.diff(phase_rvs)
            phase_recieve = np.zeros(phase.shape)
            for i in range(1, len(phase_recieve)):
                phase_recieve[i] = phase_recieve[i-1] + inst_omega_minus[i-1]

            recieve_pulse = flipped_envelope * np.exp(1j * phase_recieve)

            target_shape_note += f"\nRecieveVersion of {key}"
            
            ax.plot(np.arange(len(recieve_pulse)), recieve_pulse.real, label=f"{key} ctrl_pulse_rx")
            ax.plot(np.arange(len(recieve_pulse)), flipped_envelope, label=f"{key} ctrl_envelope_rx")
            ax.legend()

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

            print(f"Saved Recieve_Pulse for {key} at {ctrl_pulse_path}")
        plt.tight_layout()
        plt.show()

    def phase_subtraction_and_pulse_flip(self, ctrl_pulse_path):
        fig, axes = plt.subplots(len(self.result_dict), 2, figsize=(8, 2 * len(self.result_dict)))

        for i, (k, v) in enumerate(self.result_dict.items()):
            ax, ax2 = axes[i]
            target_freq = v["target_freq"]

            _, datadict = search_datadict_miyamura(self.data_path, v["date"], acquire_time=v["acquire_time"])
            time = datadict["time"]["values"].ravel()
            waveform = datadict["waveform"]["values"].ravel()
            init_phase = np.angle(demodulate(time, waveform, self.readout_lo_freq - target_freq))

            signal = 2 * lowpass(time, 
                                waveform * np.exp(2j * np.pi * (self.readout_lo_freq - target_freq) * time),
                                0.01, 0.03, 0.5, 90) * np.exp(-1j * init_phase)
            
            ax.plot(time, np.abs(signal), "r-", lw=3, label=f"{k} abs(signal)")
            ax.plot(time, signal.real, label=f"{k} real")
            ax.plot(time, signal.imag, label=f"{k} imag")

            phase = np.unwrap(np.angle(signal))
            phase_of_time = interpolate.interp1d(time, phase)

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

            ax2.plot(ctrl_pulse.real, label=f"{k} ctrl_pulse")
            ax2.plot(flipped_envelope, label=f"{k} new_ctrl_envelope")
            ax2.plot(flipped_ctrl_pulse.real, label=f"{k} new_ctrl_pulse")
            ax2.plot(time, phase, label=f"{k} phase")

            ax.legend()
            ax2.legend()

            # save
            data = DataDict(
                time=dict(unit="ns"),
                control_pulse=dict(axes=["time"]),
                control_envelope=dict(axes=["time"]),
            )
            data.validate()

            with DDH5Writer(data, ctrl_pulse_path, name=f"Control_Pulse_{k}") as writer:
                writer.add_tag(["control_pulse", "corrected"])
                writer.save_text("target_shape.md", "corrected \n" + target_shape_note)
                writer.add_data(
                    time=np.arange(len(flipped_ctrl_pulse)),
                    control_pulse=flipped_ctrl_pulse,
                    control_envelope=flipped_envelope
                )
            T.sleep(1.0)

        plt.tight_layout()
        plt.show()

class SpatiotemporalAnalysis:
    def __init__(self, header, data, ctrl_path, ctrl_pulse_tx_path, ctrl_pulse_rx_path, num_of_ph_amp, num_of_fogi_timing, f_if):
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
        self.rates = []
        self.y_absorbed_envs = []
        self.y_base_envs = []

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
        phase = np.angle(demodulate(time, waveform, self.f_if*1e-9))
        signal = 2 * lowpass(time, 
                            waveform * np.exp(2j*np.pi*self.f_if*1e-9*time),
                            0.01, 0.03, 0.1, 90) * np.exp(-1j*phase)
        norm=np.sum(2*np.abs(signal)**2)**0.5
        times.append(time)
        signals.append(signal/norm)
        
    def process_data(self):
        for p in range(self.num_of_ph_amp):
            rate = []
            energy = []
            y_absorbed_env = []
            y_base_env = []
            
            for n in range(len(self.fogi_delay[0])):
                x = self.time[p][n]
                y_absorbed = self.waveform[p][n]
                y_absorbed_shift = y_absorbed * np.exp(-1j * 2 * np.pi * self.f_if * (x * 1e-9))
                y_absorbed_LPF = np.abs(LPF(y_absorbed_shift, 500e6, 20e6, 30e6, 5, 40)) * 2
                y_absorbed_env.append(y_absorbed_LPF)
                
                y_base = self.waveform_zero_fogi[p][n]
                y_base_shift = y_base * np.exp(-1j * 2 * np.pi * self.f_if * (x * 1e-9))
                y_base_LPF = np.abs(LPF(y_base_shift, 500e6, 20e6, 30e6, 5, 40)) * 2
                y_base_env.append(y_base_LPF)
                
                E = np.sum(y_absorbed_LPF ** 2) * 2
                E1 = np.sum(y_base_LPF ** 2) * 2
                
                energy.append(E1 / 100)
                rate.append((1 - E / E1) * 100)
            
            self.energys.append(energy)
            self.rates.append(rate)
            self.y_absorbed_envs.append(y_absorbed_env)
            self.y_base_envs.append(y_base_env)
        self.results = self.get_results()
        
    def get_results(self):
        return {
            "fogi_delay": self.fogi_delay,
            "time": self.time,
            "waveform": self.waveform,
            "waveform_zero_fogi": self.waveform_zero_fogi,
            "energys": self.energys,
            "rates": self.rates,
            "y_absorbed_envs": self.y_absorbed_envs,
            "y_base_envs": self.y_base_envs
        }
    
    def roll_with_interpolation(self, array, shift):
        return nd_shift(array, shift=shift, mode='constant', cval=0.0)
    
    def overlap_comm(self, data_tx, data_rx, taus):
        ys = []
        for tau in taus:
            shift = tau/2
            shifted_data_rx = self.roll_with_interpolation(data_rx, shift=shift)
            new_func = data_tx * shifted_data_rx
            y = np.abs(np.sum(new_func) * 2)**2
            ys.append(y)
        return np.array(ys) * 100

    def overlap_curve_fitting(self, data, signal_ph_tx, signal_ph_rx, taus):
        def overlap_curve(taus, delay):
            def new_func(taus, delay):
                delay = int(round(delay))
                return self.overlap_comm(signal_ph_tx, signal_ph_rx, taus + delay)
            return new_func(taus, delay)

        par_ini = {'delay':52}
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
    
    def plot_results(self, ax=None, label=None, color=None, fitting=True):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel(r'F0g1 delay $\tau$ (ns)', size="large")
            ax.set_ylabel('Absorption rate R', size="large")
            
            plt.tick_params(pad=10, top=True, bottom=True, left=True, right=True)
            plt.rcParams["xtick.direction"] = 'in'
            plt.rcParams["ytick.direction"] = 'in'
            plt.tight_layout()
        
        
        if fitting:
            fit_result = self.overlap_curve_fitting(
                np.ravel(self.results['rates']), 
                np.asarray(self.ctrl_pulse_tx_waveform[0]), 
                np.asarray(self.ctrl_pulse_rx_waveform[0])[::-1], 
                np.ravel(self.results['fogi_delay'])
            )
            ax.plot(np.ravel(self.results['fogi_delay'])+fit_result.params['delay'].value, fit_result.best_fit, "-", label=label, color=color)
            ax.plot(np.ravel(self.results['fogi_delay'])+fit_result.params['delay'].value, np.ravel(self.results['rates']), 
                "o", label=label, color=color)
        else:
            ax.plot(np.ravel(self.results['fogi_delay']), np.ravel(self.results['rates']), 
                "o", label=label, color=color)
        return ax
    
    @staticmethod
    def plot_absorption_rate_matrix(data):
        extent = (0, 3, 0, 3)

        plt.imshow(data, extent=extent, filternorm=False, vmax=100)
        colorbar = plt.colorbar(label="Absorption rate $R_{nm}$ (%)")
        colorbar.set_ticks([0, 20, 40, 60, 80, 100])
        plt.ylabel("Tx mode n")
        plt.xlabel("Rx mode m")
        plt.tick_params(top=True, bottom=True, left=True, right=True)

        num_x_ticks = data.shape[1]
        num_y_ticks = data.shape[0]
        x_ticks = np.linspace(extent[0] + 0.37, extent[1] - 0.37, num_x_ticks)
        y_ticks = np.linspace(extent[2] + 0.37, extent[3] - 0.37, num_y_ticks)
        plt.xticks(x_ticks, range(num_x_ticks))
        plt.yticks(y_ticks, range(num_y_ticks))

        for j in range(num_y_ticks):
            for k in range(num_x_ticks):
                value = r"$\sim 0$" if int(np.round(data[k, j])) < 0 else int(np.round(data[k, j]))
                plt.text(x_ticks[j], 3 - y_ticks[k], f"{value}", ha='center', va='center', color="red")

        plt.show()
    
    def plot_waveform(self, ax=None, title=None, ph_amp_idx=None, fogi_freq_idx=None):
        if ax is None:
            fig, ax =plt.subplots()
            plt.rcParams["font.size"] = 20
            plt.tick_params(top=True, bottom=True, left=True, right=True)
            plt.rcParams["xtick.direction"] = 'in'
            plt.rcParams["ytick.direction"] = 'in'

        ax.set_xlabel('Time (ns)')
        ax.set_ylabel(r'ADC amplitude (mV)')

        ax.plot(
            self.results['time'][ph_amp_idx][fogi_freq_idx],  
            self.results['y_base_envs'][ph_amp_idx][fogi_freq_idx] * 1e3, 
            lw=3, color='tab:blue', label=r"reflection"
        )
        ax.plot(
            self.results['time'][ph_amp_idx][fogi_freq_idx],  
            self.results['y_absorbed_envs'][ph_amp_idx][fogi_freq_idx] * 1e3, 
            lw=3, color='tab:red', label=r"absorption"
        )

        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.set_ylim(-0.05, 0.25)
        ax.set_xlim(0, 1500)
        ax.set_title(f"{title}")
        ax.legend()

        return ax





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
