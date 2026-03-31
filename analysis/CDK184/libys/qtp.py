import numpy as np
import qutip as qtp

from numpy import exp, sqrt
from scipy.constants import pi
from scipy.special import factorial, eval_hermite, erf


import matplotlib.pyplot as plt
import matplotlib as mpl

from qutip import Qobj
from qutip.matplotlib_utilities import complex_phase_cmap
from numpy import angle

# from libys.plot3d import Axes3DForceZOrder as Axes3D
from mpl_toolkits.mplot3d import Axes3D


def rotation(N: int, theta: float):
    diags = [exp(-1j * n * theta) for n in range(N)]
    return qtp.qdiags(diags, 0)


def fidelity(rho: qtp.Qobj, ket: qtp.Qobj):
    return rho.matrix_element(ket.dag(), ket).real


# ⟨x|n⟩
def fock_wave_function(x, n):
    coefficient = (2**n * factorial(n) * sqrt(pi))**-0.5
    return coefficient * exp(-x**2 / 2) * eval_hermite(n, x)


# |x⟩
# N - number of dimensions in Hilbert space
def x_ket(N: int, x: float):
    n = np.arange(N)
    array = fock_wave_function(x, n)
    return qtp.Qobj(array)


# ⟨x|ρ|x⟩
def x_probability_density(rho: qtp.Qobj, x_array: np.ndarray):
    N = rho.shape[0]  # number of dimensions in Hilbert space
    pd = np.empty_like(x_array)  # probability density

    for i in range(len(pd)):
        x = x_ket(N, x_array[i])
        pd[i] = rho.matrix_element(x, x).real

    return pd


# ∫_lo^hi ⟨0|x⟩⟨x|0⟩ dx
def int_0x_x0(lo, hi):
    return (erf(hi) - erf(lo))/2


# ∫_lo^hi ⟨0|x⟩⟨x|1⟩ dx
def int_0x_x1(lo, hi):
    return (exp(-lo**2) - exp(-hi**2)) / sqrt(2*pi)


# ∫_lo^hi ⟨1|x⟩⟨x|1⟩ dx
def int_1x_x1(lo, hi):
    return np.where(
        np.isinf(hi),
        (erf(hi) - erf(lo))/2 + lo*exp(-lo**2) / sqrt(pi),
        (erf(hi) - erf(lo))/2 + (lo*exp(-lo**2) - hi*exp(-hi**2)) / sqrt(pi)
    )


# ∫_lo^hi |x⟩⟨x| dx
def int_xx(lo: float, hi: float):
    return qtp.Qobj([
        [int_0x_x0(lo, hi), int_0x_x1(lo, hi)],
        [int_0x_x1(lo, hi), int_1x_x1(lo, hi)]
    ])


# ⟨i|_n q |j⟩_n
def submatrix(q: qtp.Qobj, n: int, i: int, j: int):
    N = len(q.dims[0])  # number of qubits
    proj_i = tensor_1hot(N, n, qtp.basis(0).dag())
    proj_j = tensor_1hot(N, n, qtp.qeye(), qtp.basis(j).proj())
    return proj_i * q * proj_j


# calculate operator-sum representation Σ_k E_k ρ E_k†
def operator_sum(rho: qtp.Qobj, *E: qtp.Qobj):
    return sum(E_k * rho * E_k.dag() for E_k in E)


# create a one-hot tensor
def tensor_1hot(N: int, n: int, most: qtp.Qobj, one: qtp.Qobj):
    qobjs = N * [most]
    qobjs[n] = one
    return qtp.tensor(qobjs)


# amplitude-damp n-th qubit
# rho - density matrix
# N - number of qubits
# gamma - decay probability
def amplitude_damp(rho: qtp.Qobj, N: int, n: int, gamma: float):
    e0 = qtp.Qobj([
        [1, 0],
        [0, sqrt(1-gamma)]
    ])

    e1 = qtp.Qobj([
        [0, sqrt(gamma)],
        [0, 0]
    ])

    e0_N = tensor_1hot(N=N, n=n, most=qtp.qeye(2), one=e0)
    e1_N = tensor_1hot(N=N, n=n, most=qtp.qeye(2), one=e1)
    return operator_sum(rho, e0_N, e1_N)


def amplitude_damp_all(rho: qtp.Qobj, N: int, gamma: float):
    for n in range(N):
        rho = amplitude_damp(rho, N, n, gamma)

    return rho


# create an N-qubit 1D cluster state
def cluster_1d(N: int):
    zero = qtp.fock(2, 0)
    one = qtp.fock(2, 1)
    plus = (zero + one) / sqrt(2)
    psi = qtp.tensor(N * [plus])

    for i in range(N - 1):
        cz = qtp.csign(N=N, control=i, target=i+1)
        psi = cz * psi

    return psi


def cluster_1d_stabilizers(N: int):
    I = qtp.qeye(2)
    X = qtp.sigmax()
    Z = qtp.sigmaz()
    stabilizers = []
    stabilizers.append(qtp.tensor([X, Z] + (N-2)*[I]))

    for n in range(1, N-1):
        stabilizers.append(qtp.tensor((n-1)*[I] + [Z, X, Z] + (N-2-n)*[I]))

    stabilizers.append(qtp.tensor((N-2)*[I] + [Z, X]))
    return stabilizers


def matrix_histogram_complex(
    M,
    xlabels=None,
    ylabels=None,
    labelsize=12,
    title=None,
    limits=None,
    phase_limits=None,
    colorbar=True,
    fig=None,
    ax=None,
    threshold=None,
    azim=-30,
    elev=30,
    force_zorder=True,
    scale_x=1,
    scale_y=1,
    scale_z=1
):
    """
    Draw a histogram for the amplitudes of matrix M, using the argument
    of each element for coloring the bars, with the given x and y labels
    and title.

    Parameters
    ----------
    M : Matrix of Qobj
        The matrix to visualize

    xlabels : list of strings
        list of x labels

    ylabels : list of strings
        list of y labels

    title : string
        title of the plot (optional)

    limits : list/array with two float numbers
        The z-axis limits [min, max] (optional)

    phase_limits : list/array with two float numbers
        The phase-axis (colorbar) limits [min, max] (optional)

    ax : a matplotlib axes instance
        The axes context in which the plot will be drawn.

    threshold: float (None)
        Threshold for when bars of smaller height should be transparent. If
        not set, all bars are colored according to the color map.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.

    Raises
    ------
    ValueError
        Input argument is not valid.

    """

    if isinstance(M, Qobj):
        # extract matrix data from Qobj
        M = M.full()

    n = np.size(M)
    xpos, ypos = np.meshgrid(range(M.shape[0]), range(M.shape[1]))
    xpos = xpos.T.flatten() - 0.4
    ypos = ypos.T.flatten() - 0.4
    zpos = np.zeros(n)
    dx = dy = 0.8 * np.ones(n)
    Mvec = M.flatten()
    dz = abs(Mvec)

    # make small numbers real, to avoid random colors
    idx, = np.where(abs(Mvec) < 0.001)
    Mvec[idx] = abs(Mvec[idx])

    if phase_limits:  # check that limits is a list type
        phase_min = phase_limits[0]
        phase_max = phase_limits[1]
    else:
        phase_min = -pi
        phase_max = pi

    norm = mpl.colors.Normalize(phase_min, phase_max)
    cmap = complex_phase_cmap()  # plt.get_cmap('twilight')

    colors = cmap(norm(angle(Mvec)))
    if threshold is not None:
        colors[:, 3] = 1 * (dz > threshold)

    if ax is None:
        if fig is None:
            fig = plt.figure()

        ax = Axes3D(fig, azim=azim, elev=elev)
        fig.add_axes(ax)

    for x in range(M.shape[0]):
        for y in range(M.shape[1] - 1, -1, -1):
            i = x * M.shape[1] + y
            ax.bar3d(xpos[i], ypos[i], zpos[i], dx[i], dy[i], dz[i], color=colors[i])
    # plt.show()

    if title and fig:
        ax.set_title(title)

    # x axis
    # ax.axes.w_xaxis.set_major_locator(plt.IndexLocator(1, 0.4))
    if xlabels is None:
        xlabels = list(map(str, range(M.shape[0])))
    ax.set_xticks(np.arange(M.shape[0]))
    ax.set_xticklabels(xlabels, verticalalignment='center', horizontalalignment='right')
    ax.tick_params(axis='x', labelsize=labelsize, labelrotation=13)

    # y axis
    # ax.axes.w_yaxis.set_major_locator(plt.IndexLocator(1, 0.4))
    if ylabels is None:
        ylabels = list(map(str, range(M.shape[1])))
    ax.set_yticks(np.arange(M.shape[1]))
    ax.set_yticklabels(ylabels, verticalalignment='center', horizontalalignment='left')
    ax.tick_params(axis='y', labelsize=labelsize, labelrotation=-34)
    # ax.set_ylabel('光子数', fontsize=15)

    # z axis
    if limits is None:
        limits = [0, 1]
    correction = (limits[1] - limits[0]) / 49
    ax.set_zlim3d([limits[0] + correction, limits[1]])
    # ax.set_zlabel('絶\n対\n値', fontsize=15, rotation=0)
    # ax.zaxis.set_rotate_label(False)
    ax.zaxis._axinfo['juggled'] = (1, 2, 0)
    ax.tick_params(axis='z', labelsize=12)

    # color axis
    if colorbar:
        cax, kw = mpl.colorbar.make_axes(ax, shrink=.75, pad=.0)
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb.set_ticks([-pi, -pi / 2, 0, pi / 2, pi])
        cb.set_ticklabels((r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))
        cb.ax.tick_params(labelsize=12) 
        # cb.set_label('偏\n角', fontsize=15, rotation=0, verticalalignment='center', horizontalalignment='right')

    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))

    return fig, ax
