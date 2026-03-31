from qcodes_drivers.E4407B import E4407B
from qcodes_drivers.iq_calibrator import IQCalibrator

from setup_td import *

spectrum_analyzer = E4407B("spectrum_analyzer", "GPIB0::18::INSTR")

# iq_calibrator = IQCalibrator(
#     [__file__, setup_file],
#     data_path,
#     wiring,
#     station,
#     awg2,
#     awg_Qdrive_I,
#     awg_Qdrive_Q,
#     spectrum_analyzer,
#     lo2.frequency(),
#     if_lo=-290,  # MHz
#     if_hi=290,  # MHz
#     if_step=10,  # MHz
#     i_amp=0.9,  # V
# )
# lo2.output(True)

# iq_calibrator = IQCalibrator(
#     [__file__, setup_file],
#     data_path,
#     wiring,
#     station,
#     awg2,
#     awg_fogi_I,
#     awg_fogi_Q,
#     spectrum_analyzer,
#     lo3.frequency(),
#     if_lo=-290,  # MHz
#     if_hi=290,  # MHz
#     if_step=10,  # MHz
#     i_amp=0.9,  # V
# )
# lo3.output(True)

iq_calibrator = IQCalibrator(
    [__file__, setup_file],
    data_path,
    wiring,
    station,
    awg1,
    awg_JPA_I,
    awg_JPA_Q,
    spectrum_analyzer,
    lo1.frequency() *2,
    if_lo=-290,  # MHz
    if_hi=290,  # MHz
    if_step=10,  # MHz
    i_amp=0.9,  # V
)
lo1.output(True)


iq_calibrator.minimize_lo_leakage()
iq_calibrator.minimize_image_sideband()
iq_calibrator.measure_rf_power()