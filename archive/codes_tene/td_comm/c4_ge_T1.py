import os

import matplotlib.pyplot as plt
import numpy as np
from plottr.data.datadict_storage import DataDict, DDH5Writer
from sequence_parser import Sequence, Variable, Variables
from sequence_parser.instruction import Delay
from tqdm import tqdm

from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]

delay = Variable("delay", np.linspace(0, 200000, 101), "ns")
variables = Variables([delay])

# sequence = Sequence(ports_tx)
# sequence.call(ge_pi_seq_tx)
# sequence.add(Delay(delay), qubit_drive_port_tx)
# sequence.call(readout_seq_tx)

# sequence = Sequence(port_list=ports_txQrx)
# sequence.call(ge_pi_seq_rx)
# sequence.add(Delay(delay), qubit_drive_port_rx)
# sequence.trigger(ports_txQrx)
# sequence.call(readout_seq_tx)
# sequence.add(Acquire(100), dig_port_tx)

sequence = Sequence(ports_rx)
sequence.call(ge_pi_seq_rx)
sequence.add(Delay(delay), qubit_drive_port_rx)
sequence.call(readout_seq_rx)

data = DataDict(
    delay=dict(unit="ns"),
    s11=dict(axes=["delay"]),
)
data.validate()

try:
    with DDH5Writer(data, data_path, name=measurement_name) as writer:
        writer.add_tag(tags)
        writer.backup_file([__file__, setup_file])
        writer.save_text("wiring.md", wiring)
        writer.save_dict("station_snapshot.json", station.snapshot())
        for update_command in tqdm(variables.update_command_list):
            sequence.update_variables(update_command)
            load_sequence(sequence, cycles=3000)
            writer.add_data(
                delay=sequence.variable_dict["delay"][0].value,
                s11=demodulate(run(sequence, which=measure_which[:2]).mean(axis=0) * voltage_step_tx))
finally:
    off()
    print('finished')