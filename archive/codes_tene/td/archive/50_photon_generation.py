import os

import numpy as np
from setup_td import *
from sequence_parser import Sequence, Variable, Variables
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm


measurement_name = os.path.basename(__file__)[:-3]


fogi_freq = 5.874
fogi_amplitude = 0.6
fogi_duration = 1000
fogi_if_freq = -(fogi_freq - fogi_lo_freq)

acquisition_time = 1000
num_of_cycles = 50000
repetition = 1


seq = Sequence(port_list = [qubit_drive_port, fogi_port, dig_port])
seq.call(ge_half_pi_seq)
seq.call(ef_pi_seq)
seq.trigger([qubit_drive_port, fogi_port, dig_port])
seq.add(Square(amplitude=fogi_amplitude, duration=fogi_duration), fogi_port)
seq.add(Acquire(acquisition_time), dig_port)


seq1 = Sequence(port_list = [qubit_drive_port, fogi_port, dig_port])
seq1.add(VirtualZ(np.pi), qubit_drive_port)
seq1.call(ge_half_pi_seq)
seq1.add(VirtualZ(np.pi), qubit_drive_port)
seq1.call(ef_pi_seq)
seq1.trigger([qubit_drive_port, fogi_port, dig_port])
seq1.add(Square(amplitude=fogi_amplitude, duration=fogi_duration), fogi_port)
seq1.add(Acquire(acquisition_time), dig_port)

# seq.draw()
# raise SystemError

data = DataDict(
    time=dict(unit="ns"),
    waveform=dict(axes=["time"]),
    waveform1=dict(axes=["time"]),
)
data.validate()



with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())

    waveform = []
    waveform1 = []
    for _ in tqdm(range(repetition)):
        fogi_port.if_freq=fogi_if_freq
        load_sequence(seq, cycles=num_of_cycles)
        data = run(seq).mean(axis=0)*voltage_step
        # print(data.shape)
        waveform = np.append(waveform, data)

        load_sequence(seq1, cycles=num_of_cycles)
        data1 = run(seq1).mean(axis=0) * voltage_step
        # print(data.shape)
        waveform1 = np.append(waveform1, data1)
    waveform = waveform.reshape(int(repetition), dig_ch.points_per_cycle())
    waveform1 = waveform1.reshape(int(repetition), dig_ch.points_per_cycle())
    # print(np.array(waveform).shape)
    waveform = np.array(waveform).mean(axis=0)
    waveform1 = np.array(waveform1).mean(axis=0)
    # print(waveform.shape)    
    # raise SystemError
    writer.add_data(
        time = dig_ch.sampling_interval()*np.arange(len(waveform)),
        waveform = waveform,
        waveform1 = waveform1,
    )

    awg2.flush_waveform()
    awg1.flush_waveform()
    dig_ch.stop()
