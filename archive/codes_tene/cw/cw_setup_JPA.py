from qcodes.instrument_drivers.yokogawa.GS200 import GS200
from qcodes_drivers.M3202A import M3202A, SD_AWG_CHANNEL
from qcodes_drivers.iq_corrector import IQCorrector
from sequence_parser.sequence import Sequence
from sequence_parser.iq_port import IQPort
from sequence_parser.variable import Variable, Variables
from sequence_parser.instruction import Square
from setup_cw import *

with open(__file__) as file:
    setupJPA_script = file.read()

pump_wiring = f"\n".join([
    "M3202A_#7-1 - VLFX-400+ - 500mm - 20dB - I_J",
    "M3202A_#7-2 - VLFX-400+ - 500mm - 20dB - Q_J",
    "RFoutC - cntr - 10dB - ZVE323LN-K+ - cntr - 3dB - 3dB - ZVA213s+ - 1000mm - In1C", #rx
])


target_freq = 10.414
lo_freq = 10.29
JPA_if_freq = target_freq*2 - lo_freq*2 # f_pump - f_lo*2 #-2.425
JPA_port = IQPort('JPA_port', if_freq=JPA_if_freq)

seq_JPA = Sequence(port_list=[JPA_port])
seq_JPA.add(Square(amplitude=0.4, duration=1000), JPA_port, copy=False)

# yoko = GS200("yoko", "TCPIP0::192.168.100.213::inst0::INSTR") #readout

# # yoko.source_mode('CURR')
# yoko.current_range(1e-3)

# # station.add_component(yoko)
# if yoko.state():
#     print('yoko was on')
#     pass
# else:
#     yoko.ramp_current(0, step=1e-8, delay=0)
#     yoko.on()
#     print('yoko was off')
# assert yoko.state()
# yoko.ramp_current(90e-6, step=1e-8, delay=0)
# # raise SystemError

yoko = GS200("yoko", "TCPIP0::192.168.100.99::inst0::INSTR")
# current_source.ramp_current(0e-6, step=1e-7, delay=0)
station.add_component(yoko)
if yoko.state():
    print('yoko was on')
    pass
else:
    yoko.ramp_current(0, step=1e-8, delay=0)
    yoko.on()
    print('yoko was off')
assert yoko.state()
yoko.ramp_current(80e-6,step=1e-8, delay=0)


lo1:E8247 = E8247('lo1', 'TCPIP0::192.168.100.101::inst0::INSTR')
lo1.power(24)
lo1.frequency(lo_freq*1e9)
station.add_component(lo1)

awg1 = M3202A('awg1', chassis=1, slot=7)
station.add_component(awg1)
awg_i = awg1.ch1
awg_q = awg1.ch2

iq_corrector_JPA = IQCorrector(
    awg_i,
    awg_q,
    data_path,
    # lo_leakage_datetime="2025-08-11T094928", #comm
    # rf_power_datetime="2025-08-11T095713", #comm
    # lo_leakage_datetime="2025-08-11T100853", #tx
    # rf_power_datetime="2025-08-11T101638",
    lo_leakage_datetime="2025-08-11T180649",
    rf_power_datetime="2025-08-11T181527",
    len_kernel=41,
    fit_weight=10,
)


def pump_start(sequence: Sequence):
    sequence.compile()
    if JPA_port in sequence.port_list:
        i, q = iq_corrector_JPA.correct(JPA_port.waveform, cyclic=True)
        awg1.stop_all()
        awg1.flush_waveform()
        awg1.load_waveform(i, 1, append_zeros=True)
        awg1.load_waveform(q, 2, append_zeros=True)
        awg_i.queue_waveform(1, trigger="auto", cycles=0, per_cycle=False)
        awg_q.queue_waveform(2, trigger="auto", cycles=0, per_cycle=False)
        awg_i.dc_offset(iq_corrector_JPA.i_offset)
        awg_q.dc_offset(iq_corrector_JPA.q_offset)
    else:
        raise AssertionError
    lo1.output(True)
    awg1.start_all()


def pump_stop():
    lo1.output(False)
    awg1.stop_all()
    awg1.flush_waveform()