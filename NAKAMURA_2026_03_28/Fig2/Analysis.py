import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy.signal as sg
from scipy import interpolate, optimize
from scipy.ndimage import shift as nd_shift
from datataking import search_datadict_miyamura
from scipy.special import gamma, zeta
from mpmath import *
import time as T
import math
from scipy import signal

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

    def analyze_waveforms_padded(self, passing_band, legend=False, savefig=False, target_waveform=[]):
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

    def roll_with_interpolation(self, array, shift):
        return nd_shift(array, shift=shift, mode='constant', cval=0.0)


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
