import os

import numpy as np
from setup_td import *
from sequence_parser import Sequence, Variable, Variables
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm


measurement_name = os.path.basename(__file__)[:-3]

fogi_freqs = np.linspace(5.3, 5.8, 101) #5.465
fogi_amplitude =1.4
num_of_cycles = 5000

seq = Sequence(ports)
seq.call(ge_pi_seq)
seq.call(ef_pi_seq)
seq.trigger([qubit_drive_port, fogi_port])
seq.add(Square(amplitude=fogi_amplitude, duration=500), fogi_port)
seq.trigger(ports)
seq.call(readout_seq)

# seq.draw()
# raise SystemError

data = DataDict(
    fogi_freq=dict(unit="GHz"),
    s11=dict(axes=["fogi_freq"]),
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
    for fogi_freq in tqdm(fogi_freqs):
        fogi_port.if_freq = fogi_freq - fogi_lo_freq
        load_sequence(seq, cycles=num_of_cycles)#, chirp=True)
        data = run(seq).mean(axis=0)
        spara=demodulate(data)
        writer.add_data(
            fogi_freq=fogi_freq,
            s11=spara,
        )

        awg2.flush_waveform()
        awg1.flush_waveform()
        dig_ch.stop()
