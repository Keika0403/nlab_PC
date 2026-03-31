import numpy as np
import pandas as pd
from configparser import ConfigParser


def read_csv(filename: str):
    ini = IniReader(filename[:-4] + '.ini')
    num_indeps = int(ini.general('independent'))  # number of indep vars
    num_deps = int(ini.general('dependent'))  # number of dep vars

    dataframe = pd.read_csv(filename, header=None)
    num_rows = len(dataframe.index)  # number of rows

    indep_data = []

    for i in range(num_indeps):
        indep_data.append(dataframe[i].values)

    dep_data = []

    for i in range(num_deps):
        category = ini.dependent(i, 'category')
        label = ini.dependent(i, 'label')

        if category.startswith('Amplitude ') or category.startswith('Phase ') or label == 'linmag' or label == 'phase':
            continue
        elif category.startswith('Real ') or label == 'Real':
            dep_data.append(dataframe[num_indeps + i].values.astype(complex))
        elif category.startswith('Imaginary ') or label == 'Imag':
            dep_data[-1] += 1j * dataframe[num_indeps + i].values
        else:
            dep_data.append(dataframe[num_indeps + i].values)

    return indep_data + dep_data


def read_waveforms(filename: str, v1=0, v2=0, v3=0, v4=0, v5=0, seq=0, ch=1, compat=False):
    ini = IniReader(filename[:-1] + '.ini')
    n_samples = int(ini.parameter('adboard.nbr_samples'))
    n_segments = int(ini.parameter('adboard.nbr_segments'))
    n_repetitions = int(ini.parameter('adboard.nbr_repetitions'))
    waveforms = np.empty((n_segments * n_repetitions, n_samples))

    for i in range(n_repetitions):
        if compat:
            path = f'{filename}v1-{v1}_v2-{v2}_seq{seq}_rep{i}_CH{ch}.npy'
        else:
            path = f'{filename}v1-{v1}_v2-{v2}_v3-{v3}_v4-{v4}_v5-{v5}_seq{seq}_rep{i}_CH{ch}.npy'
        waveforms[i*n_segments:(i+1)*n_segments, :] = np.load(path)

    return waveforms


class IniReader:
    def __init__(self, filename: str):
        self.ini = ConfigParser()
        self.ini.read_file(open(filename))

    def general(self, key: str):
        return self.ini['General'][key]

    def dependent(self, num: int, key: str):  # key = 'category', 'units', 'label'
        return self.ini[f'Dependent {num+1}'][key]

    def independent(self, num: int, key: str):  # key = 'units', 'label'
        return self.ini[f'Independent {num+1}'][key]

    def parameter(self, label: str):
        n_parameters = int(self.general('parameters'))

        for i in range(n_parameters):
            parameter = self.ini[f'Parameter {i+1}']

            if parameter['label'] == label:
                return parameter['data']

        raise LookupError(f'no parameter named {label}')
