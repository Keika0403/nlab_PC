import os

import matplotlib.pyplot as plt
import numpy as np
from plottr.data.datadict_storage import DataDict, DDH5Writer
from sequence_parser import Sequence, Variable, Variables
from tqdm import tqdm

from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]

duration = Variable("duration", np.linspace(100, 1100, 101), "ns")
variables = Variables([duration])

sequence = Sequence(ports)
sequence.call(ge_pi_seq)
sequence.add(Square(amplitude=0.1, duration=duration), qubit_drive_port_tx)
sequence.call(readout_seq)

data = DataDict(
    frequency=dict(unit="GHz"),
    duration=dict(unit="ns"),
    s11=dict(axes=["frequency", "duration"]),
)
data.validate()

with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())
    for update_command in tqdm(variables.update_command_list):
        sequence.update_variables(update_command)
        for f in tqdm(np.linspace(7.6, 7.8, 101), leave=False):  # Hz
            qubit_drive_port.if_freq = f-qubit_lo_freq
            sequence.draw()
            raise SystemError
            load_sequence(sequence, cycles=2000)
            writer.add_data(
                frequency=f,
                duration=sequence.variable_dict["duration"][0].value,
                s11=demodulate(run(sequence).mean(axis=0)),
            )
