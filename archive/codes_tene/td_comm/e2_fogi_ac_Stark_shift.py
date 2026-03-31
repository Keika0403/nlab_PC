import os

import numpy as np
from setup_td import *
from sequence_parser import Sequence, Variable, Variables
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm


measurement_name = os.path.basename(__file__)[:-3]


amplitudes = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.3, 1.4]) #np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4])
fogi_freqs = np.linspace(5.9, 6.3, 81) #4.4
area=20
num_of_cycles = 2000

var = Variables()
amplitude = Variable("amplitude", value_array=amplitudes, unit="V")
duration = Variable('duration', value_array=np.round((area/amplitudes)*0.5)*2, unit='ns')#value_array=np.round((area/amplitudes)/2)*2, unit='ns')
var.add([amplitude, duration])
var.compile()

if measure_which == "tx":
    seq = Sequence(ports_tx)
    seq.call(ge_pi_seq_tx)
    seq.call(ef_pi_seq_tx)
    seq.trigger([qubit_drive_port_tx, fogi_port_tx])
    seq.add(Square(amplitude=amplitude, duration=duration), fogi_port_tx)
    seq.trigger([qubit_drive_port_tx, fogi_port_tx, readout_port_tx, dig_port_tx])
    seq.call(readout_seq_tx)
    # seq.add(Acquire(100), dig_port)
elif measure_which == "txQrx":
    seq = Sequence(ports_txQrx)
    seq.call(ge_pi_seq_rx)
    seq.call(ef_pi_seq_rx)
    seq.trigger([qubit_drive_port_rx, fogi_port_rx])
    seq.add(Square(amplitude=amplitude, duration=duration), fogi_port_rx)
    seq.trigger([qubit_drive_port_rx, fogi_port_rx, readout_port_tx, dig_port_tx])
    seq.call(readout_seq_tx)
    # seq.add(Acquire(100), dig_port)
elif measure_which == "rx":
    seq = Sequence(ports_rx)
    seq.call(ge_pi_seq_rx)
    seq.call(ef_pi_seq_rx)
    seq.trigger([qubit_drive_port_rx, fogi_port_rx])
    seq.add(Square(amplitude=amplitude, duration=duration), fogi_port_rx)
    seq.trigger([qubit_drive_port_rx, fogi_port_rx, readout_port_rx, dig_port_rx])
    seq.call(readout_seq_rx)
    # seq.add(Acquire(100), dig_port)
# seq.draw()
# raise SystemError

data = DataDict(
    fogi_freq=dict(unit="GHz"),
    amplitude = dict(unit="V"),
    s11=dict(axes=["fogi_freq", "amplitude"]),
)
data.validate()


with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())
    for update_command in tqdm(var.update_command_list):
        for fogi_freq in tqdm(fogi_freqs):
            fogi_port_tx.if_freq = fogi_freq - fogi_lo_freq_tx
            fogi_port_rx.if_freq = fogi_freq - fogi_lo_freq_rx
            seq.update_variables(update_command)
            # seq.draw()
            # raise SystemError
            load_sequence(seq, cycles=num_of_cycles)
            seq.compile()

            data = run(seq).mean(axis=0)*voltage_step_tx
            spara=demodulate(data)
            writer.add_data(
                fogi_freq=fogi_freq,
                amplitude = seq.variable_dict["amplitude"][0].value,
                s11=demodulate(run(seq).mean(axis=0)),
        )
        awg3.flush_waveform()
        awg2.flush_waveform()
        awg1.flush_waveform()
        dig_ch_tx.stop()
