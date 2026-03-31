import os

import matplotlib.pyplot as plt
import numpy as np
from plottr.data.datadict_storage import DataDict, DDH5Writer
from sequence_parser import Sequence
from tqdm import tqdm

from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]

sequence = Sequence(ports)
sequence.call(readout_seq)

electrical_delay = 1e-9/(9.2214-9.2011)


data = DataDict(
    frequency=dict(unit="Hz"),
    s11=dict(axes=["frequency"]),
)
data.validate()

print(dig_ch.points_per_cycle)

with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())
    load_sequence(sequence, cycles=5000)
    for f in tqdm(np.linspace(10.34e9, 10.42e9, 101)):#9.0e9, 9.5e9, 201):
        lo1.frequency(f + readout_if_freq*1e9)
        data = run(sequence).mean(axis=0)
        # plt.plot(data)
        # plt.show()
        writer.add_data(
            frequency=f,
 #           duration=sequence.variable_dict["duration"][0].value,
            s11=demodulate(data) * np.exp(+2j * np.pi * f * electrical_delay),
        )