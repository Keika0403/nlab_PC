import os

import numpy as np
from setup_td import *
from setup_td_tomography import *
from sequence_parser import Sequence, Variable, Variables
from plottr.data.datadict_storage import DataDict, DDH5Writer, datadict_from_hdf5
from tqdm import tqdm

measurement_name = os.path.basename(__file__)[:-3]


fogi_amplitude = 0.9 
fogi_freq = 5.602
fogi_duration=1000

acquisition_time = 1000
num_of_cycles = 50000
repetition = 50
acquisition_time = 1000

photon_freq = 10.558
photon_amplitude = 0.05  
photon_duration = 1000


def seq(fogi_amp):
    seq = Sequence(port_list = [qubit_drive_port, fogi_port, readout_port, dig_port])
#    seq.add(Delay(fogi_timing), fogi_port)

    seq.add(Delay(1170), fogi_port)
    seq.add(Square(amplitude=fogi_amp, duration=fogi_duration), fogi_port)

    seq.add(Delay(2000), readout_port)
#    seq.trigger([dig_port, readout_port])
#    seq.add(SetDetuning(-(photon_freq - readout_lo_freq ) - readout_if_freq), readout_port)
    seq.add(Acquire(acquisition_time), readout_port)
#    seq.add(Delay(1000), readout_port)

    return seq

# seq(fogi_amplitude).draw()
# raise SystemError


data = DataDict(
    time=dict(unit="ns"),
    waveform=dict(axes=["time"]),
    waveform1=dict(axes=["time"]),
    waveform2=dict(axes=["time"]),
)
data.validate()

header = "D:/K_sunada/result/CDY148/"
data_ph = "/2023-09-21/2023-09-21T102917_480c9226-td_photon_generation"
dd = datadict_from_hdf5(header+data_ph+"/data")
y_ph = dd['waveform']['values']

def y_lerp(y): #sennkeihokann
    y_new = []
    for i in range(len(y)):
        y_new.append(y[i-1])
        y_new.append((y[i-1]+y[i])/2)
    y_new.append(y[len(y)-1])
    return y_new-np.mean(y)

with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())

    waveform = []
    waveform1 = []
    waveform2 = []

    for _ in tqdm(range(repetition)):
#        load_sequence(seq(fogi_amplitude, photon_amplitude), cycles=num_of_cycles, readout=False, chirp=True)
        load_sequence_w_append(seq(fogi_amplitude), readout_port,y_lerp(y_ph[::-1]), cycles=num_of_cycles)
        data = run(seq(fogi_amplitude)).mean(axis=0)* voltage_step
        waveform = np.append(waveform, data)

        load_sequence_w_append(seq(0), readout_port,y_lerp(y_ph[::-1]), cycles=num_of_cycles)
        data1 = run(seq(0)).mean(axis=0)* voltage_step
        waveform1 = np.append(waveform1, data1)
        
        load_sequence(seq(0), cycles=num_of_cycles, readout=False, chirp=True)
        data2 = run(seq(0)).mean(axis=0)* voltage_step
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