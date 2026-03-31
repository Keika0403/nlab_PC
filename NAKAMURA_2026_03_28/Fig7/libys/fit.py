import numpy as np
import lmfit as lmf
import matplotlib as mpl
import matplotlib.pyplot as plt
import uncertainties as uc

from numpy import sqrt, pi, sin, cos


def get_ufloats(result: lmf.model.ModelResult):
    values = []

    for name, param in result.params.items():
        values.append(param.value)

    return uc.correlated_values(values, result.covar)


def rounded_report(result: lmf.model.ModelResult):
    lines = []

    for name, param in result.params.items():
        if param.stderr is None:
            value = param.value
        else:
            value = uc.ufloat(param.value, param.stderr)

        lines.append(f'{name} = {value}')

    return '\n'.join(lines)


def complex_to_ri(z):
    return z.view(float).reshape([len(z), 2])


class Projector:

    def __init__(
        self,
        z: np.ndarray,
        ref_vec: np.ndarray = None
    ):
        # covariance matrix
        cov = np.cov(complex_to_ri(z).T)

        # calculate the eigenvalues (= max&min variance)
        tr = np.trace(cov)
        det = np.linalg.det(cov)
        self.max_var = (tr + sqrt(tr**2 - 4*det)) / 2
        self.min_var = (tr - sqrt(tr**2 - 4*det)) / 2

        # calculate the normalized eigenvector of max variance
        eigenvec = (cov - self.min_var * np.identity(2))[:, 0]
        self.vec = eigenvec / np.linalg.norm(eigenvec)
        self.angle = np.arctan2(self.vec[1], self.vec[0])

        # match the direction of ref_vec
        if ref_vec is not None and np.inner(ref_vec, self.vec) < 0:
            self.vec = -self.vec

    def project(
        self,
        z: np.ndarray,
        angle: float = 0
    ) -> np.ndarray:

        rotation = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
        vec = rotation @ self.vec
        return vec[0] * z.real + vec[1] * z.imag
