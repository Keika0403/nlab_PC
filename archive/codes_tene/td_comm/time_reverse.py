import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from plottr.data.datadict_storage import datadict_from_hdf5
import lmfit


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


header = "D:/K_sunada/result/CDY153/"

def time_reverse(data, id,f_if, duration):
    dd = datadict_from_hdf5(header+data+"/data")
    x = dd['time']['values'][id][0:duration]
    y = dd['waveform']['values'][id][0:duration]
    # y = y[::-1]
    y1 =  y* np.exp(-1j*2 *np.pi* f_if*(x*1e-9))
    y_LPF = np.array(np.abs(LPF(y1, 500e6, 10e6, 30e6, 5, 40))*2**0.5)

    x_comp = []
    y_comp = []
    for i in range(len(y_LPF)-1):
        x_comp.append(x[i])
        x_comp.append((x[i]+x[i+1])/2)
        y_comp.append(y_LPF[i])
        y_comp.append((y_LPF[i]+y_LPF[i+1])/2)
    x_comp.append(x[-1])
    x_comp.append(x[-1]+1)
    y_comp.append(y_LPF[-1])
    y_comp.append(y_LPF[-1])
    
    time = np.array(x_comp)
    control_pulse = np.array(y_comp) * np.cos(2*np.pi* f_if*(time*1e-9))
    return control_pulse 


def square(freq, amp, duration):
    x = np.linspace(0, duration-1, duration)
    y = amp* np.cos(2*np.pi* freq*(x*1e-9))
    return y