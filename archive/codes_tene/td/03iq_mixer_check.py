import matplotlib.pyplot as plt

from qcodes_drivers.E4407B import E4407B
from qcodes_drivers.iq_corrector import IQCorrector
from setup_td import *

spectrum_analyzer = E4407B("spectrum_analyzer", "GPIB0::18::INSTR")



############ qdrive

# iq_corrector_q = IQCorrector(
#     awg_Qdrive_I,
#     awg_Qdrive_Q,
#     data_path,
#     lo_leakage_datetime="2024-10-11T143031",
#     rf_power_datetime="2024-10-11T143840",
#     len_kernel=41,
#     fit_weight=10,
#     plot=True,
# )
# plt.show()

# lo2.output(True)
# iq_corrector_q.check(
#     [__file__, setup_file],
#     data_path,
#     wiring,
#     station,
#     awg2,
#     spectrum_analyzer,
#     lo2.frequency(),
#     if_step=10,
#     amps=np.linspace(0.8, 1.4, 7),
# )


############ fogi

# iq_corrector_fogi = IQCorrector(
#     awg_fogi_I,
#     awg_fogi_Q,
#     data_path,
#     lo_leakage_datetime="2024-10-11T144856",
#     rf_power_datetime="2024-10-11T145734",
#     len_kernel=41,
#     fit_weight=10,
#     plot=True,
# )
# plt.show()

# lo3.output(True)
# iq_corrector_fogi.check(
#     [__file__, setup_file],
#     data_path,
#     wiring,
#     station,
#     awg2,
#     spectrum_analyzer,
#     lo3.frequency(),
#     if_step=10,
#     amps=np.linspace(0.8, 1.4, 7),
# )

############ JPA
iq_corrector_JPA = IQCorrector(
    awg_JPA_I,
    awg_JPA_Q,
    data_path,
    lo_leakage_datetime="2024-10-11T150413",
    rf_power_datetime="2024-10-11T151207",
    len_kernel=41,
    fit_weight=10,
    plot=True,
)
plt.show()

lo1.output(True)
iq_corrector_JPA.check(
    [__file__, setup_file],
    data_path,
    wiring,
    station,
    awg1,
    spectrum_analyzer,
    lo1.frequency()*2,
    if_step=10,
    amps=np.linspace(0.8, 1.4, 7),
)

