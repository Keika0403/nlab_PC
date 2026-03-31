import os

import matplotlib.pyplot as plt
import numpy as np
from plottr.data.datadict_storage import DataDict, DDH5Writer
from sequence_parser import Sequence
from tqdm import tqdm

from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]

sequence_g = Sequence(ports)
sequence_g.call(readout_seq)

sequence_e = Sequence(ports)
sequence_e.call(ge_pi_seq)
sequence_e.call(readout_seq)

electrical_delay =1e-9/(8.5244-8.5041)-1e-9/(9.495-9.093)

data = DataDict(
    frequency=dict(unit="Hz"),
    s11_g=dict(axes=["frequency"]),
    s11_e=dict(axes=["frequency"]),
    s11_eg=dict(axes=["frequency"]),
)
data.validate()

with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())
    
    for f in tqdm(np.linspace(9.e9, 9.5e9, 101)):
        lo1.frequency(f + readout_if_freq*1e9)
        s11s = []
        load_sequence(sequence_g, cycles=5000)
        s11s.append(demodulate(run(sequence_g).mean(axis=0)))
        load_sequence(sequence_e, cycles=5000)
        s11s.append(demodulate(run(sequence_e).mean(axis=0)))
    
        s11_eg = s11s[1] - s11s[0]

        writer.add_data(
            frequency=f,
            s11_g=s11s[0]* np.exp(+2j * np.pi * f * electrical_delay),
            s11_e=s11s[1]* np.exp(+2j * np.pi * f * electrical_delay),
            s11_eg = s11_eg 
            )