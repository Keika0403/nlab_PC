import os

import matplotlib.pyplot as plt
import numpy as np
from plottr.data.datadict_storage import DataDict, DDH5Writer
from sequence_parser import Sequence, Variable, Variables
from tqdm import tqdm

from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]
readout_if_freqs = -np.linspace(0.08, 0.16, 101)
amplitude = Variable("amplitude", np.linspace(0.4, 1.4, 10)[1:], "V")
variables = Variables([amplitude])

if measure_which == "tx":
    readout_pulse_tx.params["amplitude"] = amplitude
    sequence = Sequence(ports_tx)
    sequence.call(readout_seq_tx)
elif measure_which == "txQrx":
    readout_pulse_tx.params["amplitude"] = amplitude
    sequence = Sequence(ports_txQrx)
    sequence.call(readout_seq_tx)
elif measure_which == "rx":
    readout_pulse_rx.params["amplitude"] = amplitude
    sequence = Sequence(ports_rx)
    sequence.call(readout_seq_rx)
    
electrical_delay = -1/(10.4069-10.3982)+1/(10.3723-10.3643)-1/(10.11752-10.11034)
hvi_trigger.trigger_period(10000)  # ns

data = DataDict(
    frequency=dict(unit="GHz"),
    amplitude=dict(unit="V"),
    s11=dict(axes=["frequency", "amplitude"]),
)
data.validate()

with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())
    for update_command in tqdm(variables.update_command_list):
        sequence.update_variables(update_command)
        for if_freq in tqdm(readout_if_freqs):
            # readout_port_tx.if_freq = if_freq
            # readout_port_rx.if_freq = if_fres
            load_sequence(sequence, cycles=2000)
            data = run(sequence, plot=0, which=measure_which).mean(axis=0)
            s11 = demodulate(data, demodulation_if=if_freq) * np.exp(-2j * np.pi * (readout_lo_freq_rx-if_freq) * electrical_delay)
            a = sequence.variable_dict["amplitude"][0].value
            writer.add_data(
                frequency=readout_lo_freq_tx - if_freq,
                # frequency=readout_lo_freq_tx + if_freq,
                amplitude=a,
                s11=s11,
            )