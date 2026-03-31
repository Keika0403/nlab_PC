import os

import numpy as np
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm

from cw_setup_JPA import *

measurement_name = os.path.basename(__file__)[:-3]


vna.sweep_type("linear frequency")
vna.s_parameter("S21")
vna.power(-20)  # dBm
vna.start(10.214e9)#Hz 10.414
vna.stop(10.614e9)# Hz
vna.points(201)
vna.if_bandwidth(100) 
vna.electrical_delay(1e-9/(9.7738-9.7531))

amps = np.append([0], np.linspace(0.6, 1.1, 11))
currents = np.linspace(87, 97, 11)*1e-6

var = Variables()
amplitude = Variable('drive_amplitude', value_array=amps, unit='Volt')
var.add(amplitude)
var.compile()
seq = Sequence(port_list=[JPA_port])
seq.add(Square(amplitude=amplitude, duration=1000), JPA_port, copy=False)



data = DataDict(
    frequency=dict(unit="GHz"),
    amplitude = dict(unit="V"),
    current = dict(unit = "A"),
    s11=dict(axes=["frequency", "amplitude", "current"]),
    s11_per_base = dict(axes=["frequency", "amplitude", "current"]),
    Gain = dict(axes=["frequency", "amplitude", "current"])
)
data.validate()


# yoko.ramp_current(0e-6, step=1e-8, delay=0)
# yoko._set_range("CURR",1e-3)

with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())
    for current in tqdm(currents):
        yoko.ramp_current(current, step=1e-8, delay=0)
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
                current = current,
                amplitude = seq.variable_dict["drive_amplitude"][0].value,
                s11 = s11,
                s11_per_base = s11 / base,
                Gain=20 * np.log10(np.abs(s11 / base)),
            )

        pump_stop()

