import os

import numpy as np
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm

from setup_cw import *

measurement_name = os.path.basename(__file__)[:-3]

vna.sweep_type("linear frequency")
# vna.s_parameter("S21")
vna.start(1.e9)  # Hz
vna.stop(15.e9)  # Hz
vna.points(1501)
vna.if_bandwidth(100)
vna.power(-30) #dBm
# vna.electrical_delay(1e-9/(9.8313-9.8061)+1e-9/(10.2-9.85)) 
data = DataDict(
    frequency=dict(unit="Hz"),
#    power=dict(unit="dBm"),
    s11=dict(axes=["frequency"]),
    s21=dict(axes=["frequency"])
)
data.validate()

with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())
    vna.s_parameter("S11")
    vna.run_sweep()
    s11=vna.trace()
    vna.s_parameter("S21")
    vna.run_sweep()
    s21=vna.trace()
    writer.add_data(
        frequency=vna.frequencies(),
       # power=power,
        s11=s11,
        s21=s21,
        )
