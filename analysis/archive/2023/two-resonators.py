from lmfit import Model
import numpy as np
import matplotlib.pyplot as plt

def S11_two_resonators(freq, ff, fr, ke, ki, gamma, g, a0, a1, Poff,ed):
    numer = 1j*(freq-fr)+(ke-ki)/2 + (g**2)/(1j*(freq-ff)-gamma/2)
    denom = 1j*(freq-fr)-(ke+ki)/2 + (g**2)/(1j*(freq-ff)-gamma/2)
    model = (numer/denom * (a0+a1*(freq-freq[0])) * np.exp(1j*(Poff-ed*2*np.pi*freq))).conj()
    return model

def params_guess(freq, S11):
    mag = np.abs(S11)
    Poff = np.angle(S11[0])
    a0 = np.median(mag)
    S11 = S11 * np.exp(-1j*Poff)
    dip_id = np.argmin(mag)
    fc0 = freq[dip_id]
    hptv = (np.max(mag) + np.min(mag))/2
    hpfp = freq[dip_id:][np.argmin(np.abs(mag[dip_id:]-hptv))]
    hpfm = freq[:dip_id][np.argmin(np.abs(mag[:dip_id]-hptv))]
    ktot = hpfp - hpfm
    ksub = np.min(mag)/a0 * ktot
    # print(hptv,hpfp,hpfm)
    kM = (ktot+ksub)/2
    km = ktot - kM
    if np.real(S11[dip_id]) < 0:
        ke = kM
        ki = km
    else:
        ke = km
        ki = kM
    
    params_int = {'fc':fc0, 'ke':ke, 'ki':ki, 'a0':a0, 'a1': 0, 'Poff':Poff, 'ed': 0}
    return params_int

def Model_fit(freq, S11):
    S11_model = Model(S11_two_resonators)
    params = S11_model.make_params(**params_guess(freq,S11))
    result = S11_model.fit(S11,params,freq=freq)

    return result