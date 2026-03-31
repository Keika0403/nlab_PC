import os

import numpy as np
from setup_td import *
from sequence_parser import Sequence, Variable, Variables
from plottr.data.datadict_storage import DataDict, DDH5Writer
from sequence_parser.instruction import Delay
from tqdm import tqdm

measurement_name = os.path.basename(__file__)[:-3]


fogi_freq = np.linspace(5.468, 5.508, 11)#2M  5.448
fogi_amplitude = 0.9
fogi_duration = 500

photon_freq = np.linspace(10.273, 10.313, 11)#2M  10.303
photon_amp = 1.4
photon_duration = 500

num_of_cycles = 5000
fogi_timing = 0
# variables = Variables([fogi_timing])

def fogi_timing_sweep(timing):
    seq = Sequence(port_list = [qubit_drive_port, fogi_port, readout_port, dig_port])
    seq.add(Delay(timing), fogi_port)
    seq.add(Square(amplitude=fogi_amplitude, duration=fogi_duration), fogi_port)
#    seq.add(RaisedCos(amplitude=fogi_amplitude, duration=fogi_duration), fogi_port)
    seq.trigger([qubit_drive_port, fogi_port])
    # seq.call(ef_pi_seq)

    # seq.add(Delay(1000), readout_port)
    seq.add(SetDetuning(-(photon_freq - readout_lo_freq)- readout_if_freq), readout_port)
#    seq.add(RaisedCos(amplitude=photon_amp, duration=photon_duration), readout_port)
    seq.add(Square(amplitude=photon_amp, duration=photon_duration), readout_port)
    # seq.add(Delay(1000), readout_port)

    seq.trigger([readout_port, dig_port])
    seq.add(SetDetuning(0), readout_port)
    seq.call(readout_seq)
    return seq

# photon_freq=10.48
# fogi_freq = 5.67

# sequence = fogi_timing_sweep(0)

# sequence.draw()
# raise SystemError


data = DataDict(
    fogi_frequency=dict(unit="Hz"),
    photon_frequency=dict(unit='Hz'),
    s11=dict(axes=["fogi_frequency", "photon_frequency"]),
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
    for ph_f in tqdm(photon_freq):
        photon_freq=ph_f
        for fogi_f in tqdm(fogi_freq):
            fogi_port.if_freq = (fogi_f - fogi_lo_freq)
            seq=fogi_timing_sweep(fogi_timing)
            load_sequence(seq, cycles=num_of_cycles, chirp=True)
            
            data = run(seq).mean(axis=0)*voltage_step
            spara=demodulate(data)
            writer.add_data(
                fogi_frequency=fogi_f,
                photon_frequency=ph_f,
                s11=demodulate(run(seq).mean(axis=0)),
            )

        awg2.flush_waveform()
        awg1.flush_waveform()
        dig_ch.stop()

