import time

import qcodes as qc
from qcodes.instrument_drivers.yokogawa.GS200 import GS200

from qcodes_drivers.E82x7 import E82x7 as E8247
from qcodes_drivers.E82x7 import E82x7
from qcodes_drivers.N5222A import N5222A
from qcodes_drivers.E5071C import E5071C

setup_file = __file__
tags = ["CW", "CDK184", "rx"]
data_path = f"D:\\K_Sunada\\result\\{tags[1]}"
wiring = "\n".join([
    "N5222A_port1 - 2000mm - 20dB - In3B",#readout_in
    "LO - 1000mm  - 20dB - 10dB - 10dB - In3A",
    "N5222A_port2 - 2000mm -  Out2B - Miteq", #readout_out
])

station = qc.Station()

vna = N5222A("vna", "TCPIP0::192.168.100.73::inst0::INSTR")
vna.electrical_delay(-1e-9/(9.767-9.389))#1e-9/(10.7997-10.7788))  # s
vna.meas_trigger_input_type("level")
vna.meas_trigger_input_polarity("positive")
vna.aux1.output_polarity("negative")
vna.aux1.output_position("after")
vna.aux1.aux_trigger_mode("point")
station.add_component(vna)

drive_source = E8247("drive_source", 'TCPIP0::192.168.100.7::inst0::INSTR')
drive_source.trigger_input_slope("negative")
station.add_component(drive_source)


def configure_drive_sweep(vna_freq: float, points: int):
   vna.sweep_type("linear frequency")
   vna.start(vna_freq)
   vna.stop(vna_freq)
   vna.points(points)
   vna.sweep_mode("hold")
   vna.trigger_source("external")
   vna.trigger_scope("current")
   vna.trigger_mode("point")
   vna.aux1.output(True)
   drive_source.frequency_mode("list")
   drive_source.point_trigger_source("external")
   drive_source.sweep_points(points)


def run_drive_sweep():
   vna.output(True)
   drive_source.output(True)
   drive_source.start_sweep()
   vna.sweep_mode("single")
   try:
       while not (vna.done() and drive_source.sweep_done()):
           time.sleep(0.1)
   finally:
       vna.output(False)
       drive_source.output(False)
