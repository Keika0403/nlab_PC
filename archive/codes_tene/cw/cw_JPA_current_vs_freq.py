import os

import numpy as np
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm

from cw_setup_JPA import *

measurement_name = os.path.basename(__file__)[:-3]


vna.sweep_type("linear frequency")
vna.s_parameter("S21")
vna.power(-20)  # dBm
vna.start(8e9)  # Hz
vna.stop(12e9)  # Hz
vna.points(401)
vna.if_bandwidth(100) 
vna.electrical_delay(1e-9/(9.7738-9.7531))

# yoko._set_range("CURR",1e-3)
yoko.on()
# yoko.ramp_current(-4e-3, step=1e-7, delay=0)

data = DataDict(
    frequency=dict(unit="GHz"),
    current=dict(unit="uA"),
    s11=dict(axes=["frequency", "current"])
)
data.validate()

with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())
    for current in tqdm(np.linspace(0, 200e-6, 21)):
        yoko.ramp_current(current, step=1e-8, delay=0)
        vna.run_sweep()
        writer.add_data(
            frequency=vna.frequencies()*1e-9,
            current=current*1e6,
            s11=vna.trace(),
        )

yoko.ramp_current(0e-6, step=1e-8, delay=0)
yoko.off()
yoko._set_range("CURR",1e-3)