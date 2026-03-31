import os

import numpy as np
from setup_td import *
from sequence_parser import Sequence
from plottr.data.datadict_storage import DataDict, DDH5Writer
from sequence_parser.instruction import Delay
from tqdm import tqdm

measurement_name = os.path.basename(__file__)[:-3]

fogi_freq = 5.468
fogi_amplitude = 0.5
fogi_duration = 500

photon_freq = 10.299
photon_amp = np.linspace(0, 1.5, 101)
photon_duration = 500

num_of_cycles = 5000
fogi_timing = 0


def photon_amp_sweep(photon_amp):
    seq = Sequence(port_list = [qubit_drive_port, fogi_port, readout_port, dig_port])
    seq.add(Delay(fogi_timing), fogi_port)
    seq.add(Square(amplitude = fogi_amplitude, duration = fogi_duration), fogi_port)
    seq.trigger([qubit_drive_port, fogi_port])
   # seq.call(ef_pi_seq)


    # seq.add(Delay(1000), readout_port)
    seq.add(SetDetuning(-(photon_freq - readout_lo_freq)- readout_if_freq), readout_port)
    seq.add(Square(amplitude = photon_amp, duration = photon_duration), readout_port)
    # seq.add(Delay(1000), readout_port)


    seq.trigger([readout_port, dig_port])
    seq.add(SetDetuning(0), readout_port)
    seq.call(readout_seq)
    return seq

# sequence = photon_amp_sweep(0.1)

# sequence.draw()
# raise SystemError


data = DataDict(
    photon_amp=dict(unit="V"),
    s11=dict(axes=["photon_amp"]),
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
    for a in tqdm(photon_amp):
        fogi_port.if_freq = fogi_freq - fogi_lo_freq
        seq = photon_amp_sweep(a)
        load_sequence(seq, cycles=num_of_cycles, chirp=True)
        data = run(seq).mean(axis=0)*voltage_step
        spara=demodulate(data)
        writer.add_data(
            photon_amp=a,
            s11=demodulate(run(seq).mean(axis=0)),
        )

        awg2.flush_waveform()
        awg1.flush_waveform()
        dig_ch.stop()
