import numpy as np
import uncertainties.unumpy as unp

import matplotlib as mpl
mpl.use('QT5Agg')

from matplotlib.pyplot import *
from numpy import pi


def set_font(font, math='dejavusans'):
    if font == 'IPAexGothic':
        rcParams['font.sans-serif'] = ['IPAexGothic']
        rcParams['font.family'] = 'sans-serif'
    elif font == 'IPAexMincho':
        rcParams['font.serif'] = ['IPAexMincho']
        rcParams['font.family'] = 'serif'
    elif font == 'Yu Gothic Medium':
        rcParams['font.sans-serif'] = ['Yu Gothic']
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.weight'] = 'medium'
    elif font == 'Tahoma':
        rcParams['font.sans-serif'] = ['Tahoma']
        rcParams['font.family'] = 'sans-serif'

    rcParams['mathtext.fontset'] = math
    rcParams['svg.fonttype'] = 'none'


def errorbar_ufloat_y(x, y_ufloat, *args, **kwargs):
    y = unp.nominal_values(y_ufloat)
    yerr = unp.std_devs(y_ufloat)
    return errorbar(x, y, yerr=yerr, *args, **kwargs)


def errorbar_ufloat_xy(x_ufloat, y_ufloat, *args, **kwargs):
    x = unp.nominal_values(x_ufloat)
    xerr = unp.std_devs(x_ufloat)
    y = unp.nominal_values(y_ufloat)
    yerr = unp.std_devs(y_ufloat)
    return errorbar(x, y, xerr=xerr, yerr=yerr, *args, **kwargs)


def edges(
    x: np.ndarray
) -> np.ndarray:

    x_edges = np.empty(len(x)+1)
    x_edges[0] = x[0] - (x[1] - x[0])/2
    x_edges[-1] = x[-1] + (x[-1] - x[-2])/2
    x_edges[1:-1] = (x[:-1] + x[1:])/2
    return x_edges


def heatmap(
    x_list: np.ndarray,
    y_list: np.ndarray,
    color_list: np.ndarray,
    **kwargs
) -> mpl.colorbar.Colorbar:

    x_unique = np.unique(x_list)
    y_unique = np.unique(y_list)
    color_array = np.full((len(y_unique), len(x_unique)), np.nan)

    for x, y, color in zip(x_list, y_list, color_list):
        i = np.searchsorted(x_unique, x)
        j = np.searchsorted(y_unique, y)
        color_array[j, i] = color

    mesh = gca().pcolormesh(edges(x_unique), edges(y_unique), color_array, **kwargs)
    return gcf().colorbar(mesh)


def heatmap_angle(
    x_list: np.ndarray,
    y_list: np.ndarray,
    color_list: np.ndarray,
    **kwargs
) -> mpl.colorbar.Colorbar:

    return heatmap(x_list, y_list, color_list, cmap='twilight_shifted', vmin=-pi, vmax=pi, **kwargs)


def heatmap_angle_pi(
    x_list: np.ndarray,
    y_list: np.ndarray,
    color_list: np.ndarray,
    **kwargs
) -> mpl.colorbar.Colorbar:

    return heatmap(x_list, y_list, color_list / pi, cmap='twilight_shifted', vmin=-1, vmax=1, **kwargs)


def hist_rows(
    x: np.ndarray,  # x.shape == (m, n)
    y: np.ndarray,  # y.shape == (m,)
    bins: np.ndarray,
    **kwargs
) -> mpl.colorbar.Colorbar:

    x_hist = np.empty((len(y), len(bins) - 1))

    for i in range(len(y)):
        x_hist[i, :] = np.histogram(x[i, :], bins)[0]

    mesh = gca().pcolormesh(bins, edges(y), x_hist, **kwargs)
    return gcf().colorbar(mesh)
