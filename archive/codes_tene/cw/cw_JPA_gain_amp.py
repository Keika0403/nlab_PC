import os

import numpy as np
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm

from cw_setup_JPA import *

measurement_name = os.path.basename(__file__)[:-3]


vna.sweep_type("linear frequency")
vna.s_parameter("S21")
vna.power(-30)  # dBm probe power
vna.start(9.3e9)  # Hz
vna.stop(9.7e9)  # Hz
vna.points(201)
vna.if_bandwidth(20) 
vna.electrical_delay(1e-9/(9.0217-9.0019))

amps = np.append([0], np.linspace(1.0, 1.4, 10))

var = Variables()
amplitude = Variable('drive_amplitude', value_array=amps, unit='Volt')
var.add(amplitude)
var.compile()
seq = Sequence(port_list=[JPA_port])
seq.add(Square(amplitude=amplitude, duration=1000), JPA_port, copy=False)



data = DataDict(
    frequency=dict(unit="GHz"),
    amplitude = dict(unit="V"),
    s11=dict(axes=["frequency", "amplitude"]),
    s11_per_base = dict(axes=["frequency", "amplitude"]),
    Gain = dict(axes=["frequency", "amplitude"])
)
data.validate()


# yoko.ramp_current(0e-6, step=1e-8, delay=0)
# yoko._set_range("CURR",1e-3)

with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())
    for update_command in tqdm(var.update_command_list):
        seq.update_variables(update_command)
        pump_start(seq)
        vna.run_sweep()
        if seq.variable_dict["drive_amplitude"][0].value == 0:
            base = vna.trace()
            continue
        else:
            s11 = vna.trace()
            writer.add_data(
            frequency=vna.frequencies(),
            amplitude = seq.variable_dict["drive_amplitude"][0].value,
            s11 = s11,
            s11_per_base = s11 / base,
            Gain = 20 * np.log10(np.abs(s11 / base))
            )

        pump_stop()

