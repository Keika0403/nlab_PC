import numpy as np
import matplotlib.pyplot as plt
import pathlib
import time
import os
import pickle


class Logger:

    def __init__(self, path, date, num):
        self.date = date
        self.num = num
        self.filename = pathlib.Path(path).stem
        self.parent = pathlib.Path(path).parent.as_posix()
        os.makedirs(f'{self.parent}/log', exist_ok=True)
        os.makedirs(f'{self.parent}/fig', exist_ok=True)
        os.makedirs(f'{self.parent}/npy', exist_ok=True)
        os.makedirs(f'{self.parent}/pkl', exist_ok=True)
        self.log_file = open(f'{self.parent}/log/{self.filename}.log', mode='a')
        time_string = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f'\n\n{time_string} {date}-{num}\n', file=self.log_file)
        self.log_file.flush()

    def print(self, *args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=self.log_file)
        self.log_file.flush()

    def figure(self, name, date=None, num=None, figsize=(240, 180), tight_layout=True, show_title=True):
        if date is None:
            date = self.date

        if num is None:
            num = self.num

        figname = f'{date}-{num}-{name}'
        figsize_in_inches = (figsize[0]/72, figsize[1]/72)  # convert from points to inches
        plt.figure(figname, figsize_in_inches, tight_layout=tight_layout)

        if show_title:
            plt.title(figname, fontsize=10)

    def save_fig(self, extension='png', dpi=600, transparent=True, fig=None, **kwargs):
        if fig is None:
            fig = plt.gcf()

        figname = fig.canvas.get_window_title()
        path = f'{self.parent}/fig/{self.filename}_{figname}.{extension}'
        fig.savefig(path, dpi=dpi, transparent=transparent, **kwargs)

    def save_figs(self, extension='png', dpi=600, transparent=True, show=True, **kwargs):
        for fignum in plt.get_fignums():
            fig = plt.figure(fignum)
            self.save_fig(extension=extension, dpi=dpi, transparent=transparent, fig=fig, **kwargs)

        if show:
            plt.show()

    def save_npy(self, arr, name, date=None, num=None):
        if date is None:
            date = self.date

        if num is None:
            num = self.num

        path = f'{self.parent}/npy/{self.filename}_{date}-{num}-{name}.npy'
        np.save(path, arr)

    def load_npy(self, name, other_filename=None, date=None, num=None):
        if other_filename is None:
            other_filename = self.filename

        if date is None:
            date = self.date

        if num is None:
            num = self.num

        path = f'{self.parent}/npy/{other_filename}_{date}-{num}-{name}.npy'
        return np.load(path)

    def save_pkl(self, obj, name, date=None, num=None):
        if date is None:
            date = self.date

        if num is None:
            num = self.num

        path = f'{self.parent}/pkl/{self.filename}_{date}-{num}-{name}.pkl'
        pkl_file = open(path, 'wb')
        pickle.dump(obj, pkl_file)

    def load_pkl(self, name, other_filename=None, date=None, num=None):
        if other_filename is None:
            other_filename = self.filename

        if date is None:
            date = self.date

        if num is None:
            num = self.num

        path = f'{self.parent}/pkl/{other_filename}_{date}-{num}-{name}.pkl'
        pkl_file = open(path, 'rb')
        return pickle.load(pkl_file)
