import os

import numpy as np
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm

from setup_cw import *

measurement_name = os.path.basename(__file__)[:-3]

vna.s_parameter("S21")
# vna.power(-10)  # dBm
vna.if_bandwidth(10)  # Hz

drive_source.frequency_start(7.66e9)
drive_source.frequency_stop(7.72e9)
drive_source.power(-15)

# current_source._set_range("CURR",10e-3)
# current_source.on()
# current_source.ramp_current(-4e-3, step=1e-7, delay=0)

## omega_rg = 10.304e+9
configure_drive_sweep(vna_freq=10.17e9, points=121) #-1MHz


data = DataDict(
    frequency=dict(unit="Hz"),
    power=dict(unit="dBm"),
    s11=dict(axes=["frequency", "power"])
)
data.validate()

with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())
    for power in tqdm(np.linspace(0, 5, 21)):
        vna.power(power)
        run_drive_sweep()
        writer.add_data(
            frequency=drive_source.frequencies(),
            power=power,
            s11=vna.trace(),
        )

# current_source.ramp_current(0e-6, step=1e-6, delay=0)
# current_source.off()
# current_source._set_range("CURR",1e-3)