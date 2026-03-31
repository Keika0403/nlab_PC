import os

import numpy as np
from setup_td import *
from sequence_parser import Sequence, Variable, Variables
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm

measurement_name = os.path.basename(__file__)[:-3]


fogi_freq = 5.419
fogi_amplitude = 0.9
fogi_duration = 3000

photon_freq = 10.301
photon_amplitude = 0.8
photon_duration = 3000


acquisition_time = 1000
num_of_cycles = 50000
repetition = 50
acquisition_time = 3000


def seq(fogi_amp, photon_amp):
    seq = Sequence(port_list = [qubit_drive_port, fogi_port, readout_port, dig_port])

    seq.add(Delay(1000), fogi_port)
    seq.add(Square(amplitude=fogi_amp, duration=fogi_duration), fogi_port)

    seq.add(Delay(1000), readout_port)
    seq.trigger([dig_port, readout_port])
    seq.add(SetDetuning(photon_freq - readout_lo_freq  - readout_if_freq), readout_port)
    seq.add(Square(amplitude = photon_amp, duration=photon_duration), readout_port)
    seq.add(Acquire(acquisition_time), dig_port)
    # seq.add(Delay(100), readout_port)
    # seq.call(ef_pi_seq)

    # seq.trigger([readout_port, dig_port])
    # seq.add(SetDetuning(0), readout_port)
    # seq.call(readout_seq)
    return seq

# seq(fogi_amplitude, photon_amplitude).draw()
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
        fogi_port.if_freq = fogi_freq - fogi_lo_freq

        load_sequence(seq(fogi_amplitude, photon_amplitude), cycles=num_of_cycles,  chirp=True)
        data = run(seq(fogi_amplitude, photon_amplitude)).mean(axis=0)* voltage_step
        waveform = np.append(waveform, data)

        load_sequence(seq(0, photon_amplitude), cycles=num_of_cycles,  chirp=True)
        data1 = run(seq(0, photon_amplitude)).mean(axis=0)* voltage_step
        waveform1 = np.append(waveform1, data1)
        
        load_sequence(seq(0, 0), cycles=num_of_cycles,  chirp=True)
        data2 = run(seq(0, 0)).mean(axis=0)* voltage_step
        waveform2 = np.append(waveform2, data2)

    waveform = waveform.reshape(int(repetition), dig_ch.points_per_cycle())
    waveform1 = waveform1.reshape(int(repetition), dig_ch.points_per_cycle())
    waveform2 = waveform2.reshape(int(repetition), dig_ch.points_per_cycle())

    waveform = np.array(waveform).mean(axis=0)
    waveform1 = np.array(waveform1).mean(axis=0)
    waveform2 = np.array(waveform2).mean(axis=0)
 #   print(waveform.shape)    
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