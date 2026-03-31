import numpy as np
import libys as ys


def complex2ri(z):
    if isinstance(z, np.ndarray):
        return z.view(float).reshape(z.shape + (2,))
    else:
        return np.array([z.real, z.imag])


class Averaged2:

    def __init__(self, iq0, iq1):
        self.iq0 = iq0
        self.iq1 = iq1
        self.vector = complex2ri(iq1 - iq0) / abs(iq1 - iq0)**2

    def get_populations(self, iq):
        p1 = np.inner(complex2ri(iq - self.iq0), self.vector)
        p0 = 1 - p1
        return p0, p1


class Averaged3:

    def __init__(self, iq0, iq1, iq2):
        self.iq0 = iq0
        self.iq1 = iq1
        self.iq2 = iq2
        inverse = np.linalg.inv(np.array([
            [iq0.real, iq1.real, iq2.real],
            [iq0.imag, iq1.imag, iq2.imag],
            [       1,        1,        1]
        ]))
        self.matrix = inverse[:, :2]
        self.vector = inverse[:, 2]

    def get_populations(self, iq):
        p0, p1, p2 = self.matrix @ complex2ri(iq).T + self.vector[:, None]
        return p0, p1, p2


class SingleShot:

    def __init__(
        self,
        mode: np.ndarray,  # dtype = complex
        f_sideband: float,
        threshold: float,
        sampling_int: float,
    ):
        self.mode = mode
        self.f_sideband = f_sideband
        self.threshold = threshold
        self.sampling_int = sampling_int

    def get_bool(
        self,
        waveforms: np.ndarray,  # shape = (n_shots, n_samples)
        t0s: np.ndarray,
    ) -> np.ndarray:  # shape = (len(t0s), n_shots), dtype = bool

        iqs = ys.demodulate.get_iq(waveforms, t0s, self.mode, self.f_sideband, self.sampling_int)
        return iqs.real < self.threshold
