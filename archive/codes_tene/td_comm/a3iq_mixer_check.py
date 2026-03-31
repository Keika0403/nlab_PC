import matplotlib.pyplot as plt

from qcodes_drivers.E4407B import E4407B
from qcodes_drivers.iq_corrector import IQCorrector
from setup_td import *

spectrum_analyzer = E4407B("spectrum_analyzer", "GPIB0::18::INSTR")

# iq_corrector_fogi = IQCorrector(
#     awg_fogi_I_rx,
#     awg_fogi_Q_rx,
#     data_path,
#     lo_leakage_datetime="2025-08-10T134005",
#     rf_power_datetime="2025-08-10T134801",
#     len_kernel=41,
#     fit_weight=10,
#     plot=True,
# )
# plt.show()

iq_corrector_JPA = IQCorrector(
    awg_JPA_I_rx,
    awg_JPA_Q_rx,
    data_path,
    lo_leakage_datetime="2025-08-11T180649",
    rf_power_datetime="2025-08-11T181527",
    len_kernel=41,
    fit_weight=10,
    plot=True,
)
plt.show()

# iq_corrector_q = IQCorrector(
#     awg_Qdrive_I_rx,
#     awg_Qdrive_Q_rx,
#     data_path,
#     lo_leakage_datetime="2025-08-10T130233",
#     rf_power_datetime="2025-08-10T131045",
#     len_kernel=41,
#     fit_weight=10,
#     plot=True,
# )
# plt.show()


# lo4.output(True)
# iq_corrector_q.check(
#     [__file__, setup_file],
#     data_path,
#     wiring,
#     station,
#     awg3,
#     spectrum_analyzer,
#     lo4.frequency(),
#     if_step=10,
#     amps=np.linspace(0.8, 1.4, 8),
# )

# lo5.output(True)
# iq_corrector_fogi.check(
#     [__file__, setup_file],
#     data_path,
#     wiring,
#     station,
#     awg3,
#     spectrum_analyzer,
#     lo5.frequency(),
#     if_step=10,
#     amps=np.linspace(0.8, 1.4, 7),
# )

lo6.output(True)
iq_corrector_JPA.check(
    [__file__, setup_file],
    data_path,
    wiring,
    station,
    awg4,
    spectrum_analyzer,
    lo6.frequency()*2,
    if_step=10,
    amps=np.linspace(0.8, 1.4, 7),
)

