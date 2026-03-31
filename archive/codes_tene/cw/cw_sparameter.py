import os

import numpy as np
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm

from setup_cw import *

measurement_name = os.path.basename(__file__)[:-3]

vna.sweep_type("linear frequency")
vna.s_parameter("S21")
vna.start(10.0e9)  # Hz
vna.stop(10.8e9)  # Hz
vna.points(401)
vna.if_bandwidth(100)
vna.power(10) #dBm
vna.electrical_delay(1e-9/(9.7738-9.7531))
data = DataDict(
    frequency=dict(unit="Hz"),
#    power=dict(unit="dBm"),
    s21=dict(axes=["frequency"])
)
data.validate()

with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())
#    for power in tqdm(np.linspace(-50, 0, 6)):
#        vna.power(power)
    vna.run_sweep()
    writer.add_data(
        frequency=vna.frequencies(),
       # power=power,
        s21=vna.trace(),
        )
