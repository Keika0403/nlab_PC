import os

import matplotlib.pyplot as plt
import numpy as np
from plottr.data.datadict_storage import DataDict, DDH5Writer
from sequence_parser import Sequence, Variable, Variables
from sequence_parser.instruction import Delay, VirtualZ
from tqdm import tqdm

from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]

half_delay = Variable("half_delay", np.linspace(0, 40000, 101), "ns")
variables = Variables([half_delay])

# lo2.frequency(qubit_lo_freq*1e9)

# sequence = Sequence(ports_tx)
# sequence.call(ge_half_pi_seq_tx)
# sequence.add(Delay(half_delay), qubit_drive_port_tx)
# sequence.add(VirtualZ(np.pi), qubit_drive_port_tx)
# sequence.call(ge_pi_seq_tx)
# sequence.add(Delay(half_delay), qubit_drive_port_tx)
# sequence.add(VirtualZ(np.pi), qubit_drive_port_tx)
# sequence.call(ge_half_pi_seq_tx)
# sequence.call(readout_seq_tx)

# sequence = Sequence(port_list=ports_txQrx)
# sequence.call(ge_half_pi_seq_rx)
# sequence.add(Delay(half_delay), qubit_drive_port_rx)
# sequence.add(VirtualZ(np.pi), qubit_drive_port_rx)
# sequence.call(ge_pi_seq_rx)
# sequence.add(Delay(half_delay), qubit_drive_port_rx)
# sequence.add(VirtualZ(np.pi), qubit_drive_port_rx)
# sequence.call(ge_half_pi_seq_rx)
# sequence.trigger(ports_txQrx)
# sequence.call(readout_seq_tx)

sequence = Sequence(ports_rx)
sequence.call(ge_half_pi_seq_rx)
sequence.add(Delay(half_delay), qubit_drive_port_rx)
sequence.add(VirtualZ(np.pi), qubit_drive_port_rx)
sequence.call(ge_pi_seq_rx)
sequence.add(Delay(half_delay), qubit_drive_port_rx)
sequence.add(VirtualZ(np.pi), qubit_drive_port_rx)
sequence.call(ge_half_pi_seq_rx)
sequence.call(readout_seq_rx)



# sequence.draw()
# raise SystemError

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
        load_sequence(sequence, cycles=3000)
        writer.add_data(
            delay=2 * sequence.variable_dict["half_delay"][0].value,
            s11=demodulate(run(sequence).mean(axis=0)),
        )