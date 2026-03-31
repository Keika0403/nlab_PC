import os

import numpy as np
from setup_td import *
from sequence_parser import Sequence
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm


measurement_name = os.path.basename(__file__)[:-3]


import os

import numpy as np
from setup_td import *
from sequence_parser import Sequence
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm


measurement_name = os.path.basename(__file__)[:-3]
fogi_freq =   [5.4045, 5.384, 5.3495,  5.3037, 5.2539,  5.2101]
fogi_amplitude =[0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
num_of_cycles = 10000

duration = np.linspace(0, 1000, 51)


def fogi_time_sweep(duration, a):
    seq = Sequence(port_list = [qubit_drive_port, fogi_port, readout_port, dig_port])
    seq.call(ge_pi_seq)
    seq.call(ef_pi_seq)
    seq.trigger([qubit_drive_port, fogi_port])
    seq.add(Square(amplitude=a, duration=duration), fogi_port)
    seq.trigger([qubit_drive_port, fogi_port, readout_port, dig_port])
    seq.call(readout_seq)
    return seq

# seq.draw()
# raise SystemError


data = DataDict(
    fogi_duration=dict(unit="ns"),
    fogi_amp = dict(unit="V"),
    s11=dict(axes=["fogi_duration", "fogi_amp"]),
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
    for i in range(len(fogi_freq)):
        fogi_port.if_freq = fogi_freq[i] - fogi_lo_freq
        amp=fogi_amplitude[i]
        for d in tqdm(duration):
            seq = fogi_time_sweep(d, amp)
            load_sequence(seq, cycles=num_of_cycles)#, chirp=True)
            data = run(seq).mean(axis=0)*voltage_step
            spara=demodulate(data)
            writer.add_data(
                fogi_duration=d,
                fogi_amp = amp,
                s11=demodulate(run(seq).mean(axis=0)),
            )

        awg2.flush_waveform()
        awg1.flush_waveform()
        dig_ch.stop()
