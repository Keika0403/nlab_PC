import os

import matplotlib.pyplot as plt
import numpy as np
from plottr.data.datadict_storage import DataDict, DDH5Writer
from sequence_parser import Sequence, Variable, Variables
from sequence_parser.instruction import Delay
from tqdm import tqdm

from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]

delay = Variable("delay", np.linspace(0, 100000, 101), "ns")
variables = Variables([delay])

sequence = Sequence(ports)
sequence.call(ge_pi_seq)
sequence.call(ef_pi_seq)
sequence.add(Delay(delay), qubit_drive_port)
sequence.call(ge_half_pi_seq) 
sequence.trigger(ports)
sequence.call(readout_seq)
sequence.add(Acquire(100), dig_port)

data = DataDict(
    delay=dict(unit="ns"),
    s11=dict(axes=["delay"]),
)
data.validate()

with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())
    for update_command in tqdm(variables.update_command_list):
        sequence.update_variables(update_command)
        load_sequence(sequence, cycles=10000)
        writer.add_data(
            delay=sequence.variable_dict["delay"][0].value,
            s11=demodulate(run(sequence).mean(axis=0)),
        )