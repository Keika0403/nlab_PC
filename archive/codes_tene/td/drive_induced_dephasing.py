import os

import numpy as np
from setup_td import *
from sequence_parser import Sequence, Variable, Variables
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm

measurement_name = os.path.basename(__file__)[:-3]

half_delay = np.arange(0, 5000, 100)
detuning = +0.0004
target_frequency = 9.35
drive_amplitudes = np.sqrt(np.linspace(0, 0.03**2, 21))
num_of_cycles = 10000

var = Variables()
amplitude = Variable('amplitude', value_array=drive_amplitudes, unit=' V')
v1 = Variable('half_delay', value_array=half_delay, unit=' ns')
# v2 = Variable("drive_duration", value_array=half_delay*2+ge_pi_pulse.params["duration"], unit=' ns' )
var.add(amplitude)
# var.add([v1, v2])
var.add(v1)
var.compile()

# v1 = 1000
# amplitude = 0.1

seq = Sequence(port_list=[qubit_drive_port, readout_port, dig_port, JPA_port])
seq.add(SetDetuning(detuning), qubit_drive_port)
seq.call(ge_half_pi_seq)
seq.trigger(ports)
seq.add(SetDetuning(readout_lo_freq-target_frequency-readout_port.if_freq), readout_port)
seq.add(Delay(v1), qubit_drive_port)
seq.add(Square(amplitude=amplitude, duration=v1), readout_port, copy=False)
# seq.add(Square(amplitude=amplitude, duration=ge_pi_pulse.params["duration"]), readout_port, copy=False)
seq.trigger([qubit_drive_port, dig_port, JPA_port, readout_port])

seq.add(SetDetuning(0), qubit_drive_port)
seq.add(VirtualZ(pi), qubit_drive_port)
seq.call(ge_pi_seq)
seq.add(VirtualZ(pi), qubit_drive_port)

seq.add(SetDetuning(detuning), qubit_drive_port)
seq.trigger([qubit_drive_port, dig_port, JPA_port, readout_port])
seq.add(Delay(v1), qubit_drive_port)
seq.add(Square(amplitude=amplitude, duration=v1), readout_port, copy=False)
seq.trigger([qubit_drive_port, dig_port, JPA_port])
seq.add(SetDetuning(detuning), qubit_drive_port)
seq.add(VirtualZ(pi), qubit_drive_port)
seq.call(ge_half_pi_seq)
seq.add(VirtualZ(pi), qubit_drive_port)
seq.add(SetDetuning(0), readout_port)
seq.trigger([qubit_drive_port, readout_port, dig_port, JPA_port])
seq.call(readout_single_shot_seq)

# seq.draw()
# raise SystemError

data = DataDict(
    amplitude=dict(unit="V"),
    delay=dict(unit="ns"),
    s11=dict(axes=["amplitude", "delay"]),
)
data.validate()
try:
    with DDH5Writer(data, data_path, name=measurement_name) as writer:
        writer.add_tag(tags)
        writer.backup_file([__file__, setup_file])
        writer.save_text("wiring.md", wiring)
        writer.save_dict("station_snapshot.json", station.snapshot())
        for update_command in tqdm(var.update_command_list):
            seq.update_variables(update_command)
            load_sequence(seq, cycles=num_of_cycles)
            data = run(seq).mean(axis=0)*voltage_step
            spara=demodulate(data)
            writer.add_data(
                amplitude=seq.variable_dict['amplitude'][0].value,
                delay=seq.variable_dict['half_delay'][0].value*2+ge_pi_pulse.params["duration"],
                s11=demodulate(run(seq).mean(axis=0) * voltage_step)
            )
            awg2.flush_waveform()
            awg1.flush_waveform()
            dig_ch.stop()
except:print('An error occurred')
finally:
    off()
    print('finished')