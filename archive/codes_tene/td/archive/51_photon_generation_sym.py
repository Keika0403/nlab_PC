import os

import numpy as np
from setup_td import *
from sequence_parser import Sequence, Variable, Variables
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm


measurement_name = os.path.basename(__file__)[:-3]


fogi_freq = 5.468
fogi_amplitude = 0.9
# fogi_duration = 200
fogi_if_freq = -(fogi_freq - fogi_lo_freq)

acquisition_time = 500
num_of_cycles = 50000
repetition = 25


seq_sym = Sequence(port_list = [qubit_drive_port, fogi_port, dig_port])
seq_sym.call(ge_half_pi_seq)
seq_sym.call(ef_pi_seq)
seq_sym.trigger([qubit_drive_port, fogi_port, dig_port])
seq_sym.add(RaisedCos(amplitude=fogi_amplitude, duration=500), fogi_port)
seq_sym.add(Acquire(acquisition_time), dig_port)



# seq.compile()
# plt.plot(qubit_drive_port.waveform.real)

##0-1
seq_sym1 = Sequence(port_list = [qubit_drive_port, fogi_port, dig_port])
seq_sym1.add(VirtualZ(np.pi), qubit_drive_port)
seq_sym1.call(ge_half_pi_seq)
seq_sym1.add(VirtualZ(np.pi), qubit_drive_port)
seq_sym1.call(ef_pi_seq)
seq_sym1.trigger([qubit_drive_port, fogi_port, dig_port])
seq_sym1.add(RaisedCos(amplitude=fogi_amplitude, duration=500), fogi_port)
seq_sym1.add(Acquire(acquisition_time), dig_port)

seq_offset = Sequence(port_list = [qubit_drive_port, fogi_port, dig_port])
seq_offset.call(ge_half_pi_seq)
seq_offset.call(ef_pi_seq)
seq_offset.trigger([qubit_drive_port, fogi_port, dig_port])
seq_offset.add(RaisedCos(amplitude=0, duration=500), fogi_port)
seq_offset.add(Acquire(acquisition_time), dig_port)

# seq.compile()
# plt.plot(qubit_drive_port.waveform.real)

# #seq.draw()
# # plt.show()
# raise SystemError

data = DataDict(
    time=dict(unit="ns"),
    waveform=dict(axes=["time"]),
    waveform1=dict(axes=["time"]),
    waveform2=dict(axes=["time"]),
)
data.validate()



with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())

    waveform = []
    waveform1 = []
    waveform2 = []
    for _ in tqdm(range(repetition)):
        load_sequence(seq_sym, cycles=num_of_cycles, chirp=True)
        data = run(seq_sym).mean(axis=0) * voltage_step
        # print(data.shape)
        waveform = np.append(waveform, data)

        load_sequence(seq_offset, cycles=num_of_cycles,chirp=True)
        data = run(seq_offset).mean(axis=0) * voltage_step
        waveform2 = np.append(waveform2, data)

        load_sequence(seq_sym1, cycles=num_of_cycles,chirp=True)
        data = run(seq_sym1).mean(axis=0) * voltage_step
        waveform1 = np.append(waveform1, data)

        # print(data.shape)
        
    waveform = waveform.reshape(int(repetition), dig_ch.points_per_cycle())
    waveform2 = waveform2.reshape(int(repetition), dig_ch.points_per_cycle())
    waveform1 = waveform1.reshape(int(repetition), dig_ch.points_per_cycle())

    # print(np.array(waveform).shape)
    waveform = np.array(waveform).mean(axis=0)
    waveform2 = np.array(waveform2).mean(axis=0)
    waveform1 = np.array(waveform1).mean(axis=0)

    # print(waveform.shape)    
    # raise SystemError
    writer.add_data(
        time = dig_ch.sampling_interval()*np.arange(len(waveform)),
        waveform = waveform,
        waveform1 = waveform1,
        waveform2 = waveform2,
    )

    awg2.flush_waveform()
    awg1.flush_waveform()
    dig_ch.stop()
