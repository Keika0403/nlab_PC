import os

import matplotlib.pyplot as plt
import numpy as np
from plottr.data.datadict_storage import DataDict, DDH5Writer
from sequence_parser import Sequence, Variable, Variables
from sequence_parser.instruction import Delay, VirtualZ
from tqdm import tqdm

from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]

delay = Variable("delay", np.linspace(0, 1000,  51), "ns")
variables = Variables([delay])

lo2.frequency(qubit_lo_freq*1e9)
detuning = -0.005

sequence = Sequence(ports)
sequence.call(ge_pi_seq)
sequence.add(SetDetuning(detuning + anharmonicity), qubit_drive_port)
sequence.add(ef_half_pi_pulse_drag, qubit_drive_port)
sequence.add(Delay(delay), qubit_drive_port)
sequence.add(VirtualZ(np.pi), qubit_drive_port)
sequence.add(ef_half_pi_pulse_drag, qubit_drive_port)
sequence.add(VirtualZ(np.pi), qubit_drive_port)
sequence.trigger([qubit_drive_port, readout_port, dig_port])
sequence.call(readout_seq)
sequence.add(Acquire(100), dig_port)

# sequence.draw()
# raise SystemError


data = DataDict(
    delay=dict(unit="ns"),
    s11=dict(axes=["delay"])
)
data.validate()

with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())
    for update_command in tqdm(variables.update_command_list):
        sequence.update_variables(update_command)
        load_sequence(sequence, cycles=5000)
        data = run(sequence).mean(axis=0)
        writer.add_data(
            delay=sequence.variable_dict["delay"][0].value,
            s11 = demodulate(data)
        )