
from sequence_parser import Port, Sequence
from sequence_parser.iq_port import Port
from sequence_parser.instruction import *
from numpy import pi
from setup_td import *
from plottr.data.datadict_storage import datadict_from_hdf5

setup_parameters_file = __file__

"""
<units>
frequency : GHz
time : ns
"""

wiring = "\n".join([
    """
    "readout",
    "M3202A_#2-2 - 500mm - 20dB - IF_R",
    "E8257D - 1500mm - LO_R",
    "RF_R - 1000mm - 3dB - 3dB - In3G",

    "qubit drive",
    "M3202A_#4-1 - 500mm - 20dB - I_Q",
    "M3202A_#4-2 - 500mm - 20dB - Q_Q",
    "EXG_N5173B - 1000mm - LO_Q",
    "RF_Q - 1000mm - DC(-10dB) - In1D",

    "fogi drive",
    "M3202A_#4-3 - 500mm - 20dB - Ifogi1",
    "M3202A_#4-4 - 500mm - 20dB - Qfogi1",
    "MXG_N5183B - 1000mm - LO1",
    "RFout_fogi1 - 10dB - (ZX60-83LN-S+) - 1000mm - BPF - DC(through) - In1D",

    JPA
    "M3202A_#2-3 - 500mm - 20dB - Iin1C",
    "M3202A_#2-4 - 500mm - 20dB - Qin1C",
    "RF-J - cntr - 3dB - 10dB - ZVA-213-S+(12dB) - ZVA-213-S+(12dB) - 1000mm - In1C",

    Output
    Miteq - Out1B- 1500mm - OUT-R
    DIZ - 500mm - M3102A_#7-2 

    """,
    ])
tags = ["TD", "CDY169", "Out2B"]
data_path = f"D:\\K_Sunada\\result\\{tags[1]}"

# Frequency settings
readout_freq = 10.387
readout_lo_freq = 10.5#9.054 
readout_if_freq = readout_lo_freq - readout_freq
JPA_if_freq = readout_freq * 2 - readout_lo_freq * 2

qubit_lo_freq = 7.5
ge_freq = 7.685 - 0.00252 -0.001405 -1.6455e-05 + 8.72e-6
anharmonicity =  -0.374 - 0.0006948 + 0.00262693 + 1.553e-5
ge_if_freq = ge_freq-qubit_lo_freq
print(ge_freq, anharmonicity)

fogi_lo_freq = 2 * qubit_lo_freq - readout_lo_freq# - 1/ 100000#5.35
fogi_freq = 5.6 # amp=0.5
fogi_if_freq = -(fogi_freq - fogi_lo_freq)# determined by reset pulse
# fogi_if_freq_tr = (fogi_freq - fogi_lo_freq)

# fogi_amplitude = 1.4
# fogi_duration = 3000
print(fogi_lo_freq)

print(f'LO_r + LO_fogi - 2*LO_q = {readout_lo_freq + fogi_lo_freq - 2*qubit_lo_freq}')
if readout_lo_freq + fogi_lo_freq - 2*qubit_lo_freq!=0:
    print(f'for photon generation trigger period should be integer*{round(1/(readout_lo_freq + fogi_lo_freq - 2*qubit_lo_freq), 1)} ns')

# Ports
readout_port = Port("readout_port", if_freq = readout_if_freq)
qubit_drive_port = Port("qubit_drive_port", if_freq = ge_if_freq)
fogi_port = Port('fogi', if_freq=fogi_if_freq)
# fogi_port_tr = Port('fogi_tr', if_freq=fogi_if_freq_tr)
dig_port = Port('dig')
JPA_port = Port('JPA_port', if_freq=JPA_if_freq)
ports = [dig_port, JPA_port, readout_port, qubit_drive_port, fogi_port]
ports_wo_JPA = [dig_port, readout_port, qubit_drive_port, fogi_port]

# readout pulse
width_readout_pulse = 100
readout_pulse = Square(amplitude=0.5, duration=100)
dig_acquire = Acquire(readout_pulse.params["duration"])
readout_seq = Sequence(port_list=[readout_port, dig_port])
readout_seq.trigger(ports)
readout_seq.add(ResetPhase(0), readout_port, copy=False)
readout_seq.add(readout_pulse, readout_port, copy=False)
readout_seq.add(dig_acquire, dig_port, copy=False)
readout_seq.trigger([readout_port, dig_port])
readout_seq.add(Delay(10), readout_port, copy=False)
# readout_seq.draw()

JPA_phase = ResetPhase(0) #1.29 *np.pi
JPA_pump=Square(amplitude=1.0, duration = 1000) #0.43 for oneshot
seq_JPA = Sequence(port_list=[JPA_port])
seq_JPA.add(JPA_phase, JPA_port, copy=False)
seq_JPA.add(JPA_pump, JPA_port, copy=False)

JPA_phase_pi = ResetPhase(np.pi)
JPA_pump_pi=Square(amplitude=1.0, duration=1000)
seq_JPA_pi = Sequence(port_list=[JPA_port])
seq_JPA_pi.add(JPA_phase_pi, JPA_port, copy=False)
seq_JPA_pi.add(JPA_pump_pi, JPA_port, copy=False)

# ge pi pulse
ge_pi_pulse = Gaussian(amplitude= 1.04708, fwhm=10, duration=20, zero_end=True)
ge_pi_pulse_drag = HalfDRAG(ge_pi_pulse, beta= 0.545)#1.83)
ge_pi_seq = Sequence()
ge_pi_seq.add(ge_pi_pulse_drag, qubit_drive_port, copy=False)

# ge half pi pulse
ge_half_pi_pulse = Gaussian(amplitude=ge_pi_pulse.params['amplitude']/2, fwhm=10, duration=20, zero_end=True)
ge_half_pi_pulse_drag = HalfDRAG(ge_half_pi_pulse, beta= 0.709)#1.83)#0.310)#-1.032) # 1.35
ge_half_pi_seq = Sequence()
ge_half_pi_seq.add(ge_half_pi_pulse_drag, qubit_drive_port, copy=False)


# ef pi pulse
ef_pi_pulse = Gaussian(amplitude=0.558, fwhm=10, duration=20, zero_end=True)
ef_pi_pulse_drag = HalfDRAG(ef_pi_pulse, beta= -0.760)##0.154)#0.9223)
ef_pi_seq = Sequence(port_list=[qubit_drive_port])
ef_pi_seq.add(SetDetuning(anharmonicity), qubit_drive_port)
ef_pi_seq.add(ef_pi_pulse_drag, qubit_drive_port, copy=False)
ef_pi_seq.add(SetDetuning(0), qubit_drive_port)

# ef half pi pulse
ef_half_pi_pulse = Gaussian(amplitude=ef_pi_pulse.params['amplitude']/2, fwhm=10, duration=20, zero_end=True)
ef_half_pi_pulse_drag = HalfDRAG(ef_half_pi_pulse, beta=-0.463)#0.228)#1.027)
ef_half_pi_seq = Sequence()
ef_half_pi_seq.add(SetDetuning(anharmonicity), qubit_drive_port)
ef_half_pi_seq.add(ef_half_pi_pulse_drag, qubit_drive_port, copy=False)
ef_half_pi_seq.add(SetDetuning(0), qubit_drive_port)

# single shot
JPA_phase_ss = ResetPhase(0.25*np.pi) #1.5 *np.pi
JPA_pump_ss=Square(amplitude=0.99, duration = 400) #0.43 for oneshot
seq_JPA_ss = Sequence(port_list=[JPA_port])
seq_JPA_ss.add(JPA_phase_ss, JPA_port, copy=False)
seq_JPA_ss.add(JPA_pump_ss, JPA_port, copy=False)


readout_pulse_ss = Square(amplitude=1.2, duration=400)
dig_acquire_ss = Acquire(readout_pulse_ss.params["duration"])
readout_seq_ss = Sequence(port_list=[readout_port, dig_port])
readout_seq_ss.trigger(ports)
readout_seq_ss.add(ResetPhase(0), readout_port, copy=False)
readout_seq_ss.add(readout_pulse_ss, readout_port, copy=False)
readout_seq_ss.add(dig_acquire_ss, dig_port, copy=False)
readout_seq_ss.trigger([readout_port, dig_port])
readout_seq_ss.add(Delay(10), readout_port, copy=False)

readout_single_shot_seq = Sequence(port_list=[readout_port, JPA_port, dig_port])
readout_single_shot_seq.trigger([readout_port, JPA_port, dig_port])
readout_single_shot_seq.call(readout_seq_ss)
readout_single_shot_seq.call(seq_JPA_ss)
readout_single_shot_seq.trigger([readout_port, JPA_port, dig_port])
readout_single_shot_seq.add(Delay(60), readout_port)

# # Reset pulse
Appeared_anhm = SetDetuning(-0.3775)
reset_pulse_ef = FlatTop(Gaussian(amplitude=0.00559, fwhm=10, duration=20), top_duration=1100)
reset_pulse_fogi = FlatTop(Gaussian(amplitude=0.5, fwhm=10, duration=20), top_duration=1100)
reset_sequence = Sequence()
reset_sequence.add(Appeared_anhm, qubit_drive_port, copy=False)
reset_sequence.add(reset_pulse_ef, qubit_drive_port, copy=False) # qubit drive to e-f
reset_sequence.add(reset_pulse_fogi, fogi_port, copy=False) # f0-g1 drive
reset_sequence.trigger([qubit_drive_port, fogi_port])
reset_sequence.add(SetDetuning(0), qubit_drive_port, copy=False)
reset_sequence.add(SetDetuning(0), fogi_port, copy=False)
reset_sequence.add(ResetPhase(), qubit_drive_port)
reset_sequence.add(ResetPhase(), fogi_port)

def minus_sequence(seq:Sequence):
    assert len(seq.port_list)==1, (f"number of ports in the given sequence must be 1 but obtained {len(seq.port_list)}")
    minus_seq = Sequence()
    minus_seq.add(VirtualZ(pi), seq.port_list[0])
    minus_seq.call(seq)
    minus_seq.add(VirtualZ(pi), seq.port_list[0])
    minus_seq.compile()
    return minus_seq

# readout threshold
header = "D:/K_sunada/result/CDY158/"
data ="/2024-05-13/2024-05-13T173511_b63a777e-82_readout_fidelity_threshold"
dd =datadict_from_hdf5(header + data +"/data")
mean_g = dd["pulse_g"]["values"].ravel()
mean_e = dd["pulse_e"]["values"].ravel()
ein_vec = (mean_e - mean_g)/np.sum((mean_e - mean_g)**2)