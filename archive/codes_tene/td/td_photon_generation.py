import os

import numpy as np
from setup_td import *
from sequence_parser import Sequence, Variable, Variables
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm


measurement_name = os.path.basename(__file__)[:-3]


fogi_amplitude = 0.9
fogi_freq = 5.528

acquisition_time = 1000
num_of_cycles = 50000
repetition = 1

seq = Sequence(port_list = [qubit_drive_port, fogi_port, dig_port])
seq.call(ge_half_pi_seq)
seq.call(ef_pi_seq)
seq.trigger([qubit_drive_port, fogi_port, dig_port])
seq.add(Square(amplitude=fogi_amplitude, duration=1000), fogi_port)
seq.add(Acquire(acquisition_time), dig_port)



# seq.draw()
# raise SystemError

data = DataDict(
    time=dict(unit="ns"),
    waveform=dict(axes=["time"]),
)
data.validate()



with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())

    waveform = []
    for _ in tqdm(range(repetition)):
        load_sequence(seq, cycles=num_of_cycles, chirp=True)
        data = run(seq).mean(axis=0)
        # print(data.shape)
        waveform = np.append(waveform, data)
    waveform = waveform.reshape(int(repetition), dig_ch.points_per_cycle())
    print(np.array(waveform).shape)
    waveform = np.array(waveform).mean(axis=0)
    print(waveform.shape)    
    # raise SystemError
    writer.add_data(
        time = dig_ch.sampling_interval()*np.arange(len(waveform)),
        waveform = waveform,
    )

    awg2.flush_waveform()
    awg1.flush_waveform()
    dig_ch.stop()
