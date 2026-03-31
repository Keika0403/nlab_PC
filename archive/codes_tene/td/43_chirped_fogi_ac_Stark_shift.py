import os

import numpy as np
from setup_td import *
from sequence_parser import Sequence, Variable, Variables
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm


measurement_name = os.path.basename(__file__)[:-3]


amplitudes = np.array([0.2, 0.6 , 0.8,1.,1.2, 1.4])
fogi_freqs = np.linspace(5.1, 5.7, 31)
area=200
num_of_cycles = 5000

var = Variables()
amplitude = Variable("amplitude", value_array=amplitudes, unit="V")
duration = Variable('duration', value_array=np.round((area/amplitudes)*0.5)*2, unit='ns')#value_array=np.round((area/amplitudes)/2)*2, unit='ns')
var.add([amplitude, duration])
var.compile()


seq = Sequence(ports)
seq.call(ge_pi_seq)
seq.call(ef_pi_seq)
seq.trigger([qubit_drive_port, fogi_port])
seq.add(Square(amplitude=amplitude, duration=duration), fogi_port)
seq.trigger([qubit_drive_port, fogi_port, readout_port, dig_port])
seq.call(readout_seq)



data = DataDict(
    fogi_freq=dict(unit="GHz"),
    amplitude = dict(unit="V"),
    s11=dict(axes=["fogi_freq", "amplitude"]),
)
data.validate()





dig_ch.cycles(num_of_cycles)
lo1.output(True)
lo2.output(True)
lo3.output(True)
with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())
    for update_command in tqdm(var.update_command_list):
        for fogi_freq in tqdm(fogi_freqs):
            fogi_port.if_freq = fogi_freq - fogi_lo_freq
            seq.update_variables(update_command)
            load_sequence(seq, cycles=num_of_cycles, chirp=True)
            seq.compile()

            data = run(seq).mean(axis=0)*voltage_step
            spara=demodulate(data)
            writer.add_data(
                fogi_freq=fogi_freq,
                amplitude = seq.variable_dict["amplitude"][0].value,
                s11=demodulate(run(seq).mean(axis=0)),
        )

        awg2.flush_waveform()
        awg1.flush_waveform()
        dig_ch.stop()
