import os

import numpy as np
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm

from cw_setup_JPA import *

measurement_name = os.path.basename(__file__)[:-3]


# vna.sweep_type("linear frequency")
vna.s_parameter("S21")
vna.power(-10)  # dBm probe power
vna.start(9.0e9)  # Hz
vna.stop(9.5e9)  # Hz
vna.points(201)
vna.if_bandwidth(10) 
# vna.electrical_delay(1e-9/(11.99-11.9695))

pump_powers = np.linspace(10, 24, 13)


# configure_drive_sweep(vna_freq=9.966e9, points=201)

data = DataDict(
    frequency=dict(unit="GHz"),
    pump_power = dict(unit="dBm"),
    s11=dict(axes=["frequency", "pump_power"]),
    s11_per_base = dict(axes=["frequency", "pump_power"]),
    Gain = dict(axes=["frequency", "pump_power"])
)
data.validate()


# yoko.ramp_current(0e-6, step=1e-8, delay=0)
# yoko._set_range("CURR",1e-3)

with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())

    # wo pump
    lo1.output(False)
    vna.run_sweep()
    base = vna.trace()
    for power in tqdm(pump_powers):
        lo1.power(power)
        lo1.frequency(target_freq*1e9)
        lo1.output(True)
        vna.run_sweep()
        s11 = vna.trace()

        writer.add_data(
        frequency=vna.frequencies(),
        pump_power = power,
        s11 = s11,
        s11_per_base = s11 / base,
        Gain = 20 * np.log10(np.abs(vna.trace() / base))
        )

        lo1.output(False)

