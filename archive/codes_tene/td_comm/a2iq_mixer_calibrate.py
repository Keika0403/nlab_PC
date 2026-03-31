from qcodes_drivers.E4407B import E4407B
from qcodes_drivers.iq_calibrator import IQCalibrator

from setup_td import *

spectrum_analyzer = E4407B("spectrum_analyzer", "GPIB0::18::INSTR")

# iq_calibrator = IQCalibrator(
#     [__file__, setup_file],
#     data_path,
#     wiring,
#     station,
#     awg3,
#     awg_Qdrive_I_rx,
#     awg_Qdrive_Q_rx,
#     spectrum_analyzer,
#     lo4.frequency(),
#     if_lo=-290,  # MHz
#     if_hi=290,  # MHz
#     if_step=10,  # MHz
#     i_amp=0.9,  # V

# )

iq_calibrator = IQCalibrator(
    [__file__, setup_file],
    data_path,
    wiring,
    station,
    awg4,
    awg_JPA_I_rx,
    awg_JPA_Q_rx,
    spectrum_analyzer,
    lo6.frequency() *2,
    if_lo=-290,  # MHz
    if_hi=290,  # MHz
    if_step=10,  # MHz
    i_amp=0.9,  # V
)

# iq_calibrator = IQCalibrator(
#     [__file__, setup_file],
#     data_path,
#     wiring,
#     station,
#     awg3,
#     awg_fogi_I_rx,
#     awg_fogi_Q_rx,
#     spectrum_analyzer,
#     lo5.frequency(),
#     if_lo=-290,  # MHz
#     if_hi=290,  # MHz
#     if_step=10,  # MHz
#     i_amp=0.9,  # V
# )

# lo1.output(True)
# lo2.output(True)
# lo3.output(True)
# lo4.output(True)
# lo5.output(True)
lo6.output(True)
iq_calibrator.minimize_lo_leakage()
iq_calibrator.minimize_image_sideband()
iq_calibrator.measure_rf_power()