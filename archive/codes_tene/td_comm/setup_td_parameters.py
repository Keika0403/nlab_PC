from sequence_parser import Port, Sequence
from sequence_parser.iq_port import Port
from sequence_parser.instruction import *
from numpy import pi
from datataking import search_datadict_miyamura
import os

setup_parameters_file = __file__

measure_which = "tx" # "tx", "rx", "txQrx", "both"
print(f"measure_which : {measure_which}")

tags = ["TD", "CDK184", measure_which] # "MPC2-BW9500-1-610", 
data_path = f"D:\\K_Sunada\\result\\{tags[1]}"
# iq_data_path = f"D:\\K_Sunada\\result\\CDK183"
os.makedirs(data_path, exist_ok=True)
wiring = "\n".join([
    "readout",
    "M3202A_#2-1 - VLFX-400+ - 500mm - 20dB - IF_R",
    "E8257D - 1000mm - LO1",
    "RF_R - 1000mm - 3dB - 3dB - In2C", #comm
    # "RF_R - 1000mm - 3dB - 3dB - In2B", #tx

    "readout rx",
    "M3202A_#2-2 - VLFX-400+ - 500mm - 20dB - IF_R",
    "E8257D - 1000mm - LO_R",
    "RF_R - 3dB - 3dB - 1000mm - In3B",

    "qubit drive tx",
    "M3202A_#3-1 - VLFX-400+ -  500mm - 20dB - I_Q",
    "M3202A_#3-2 - VLFX-400+ -  500mm - 20dB - Q_Q",
    "RS - 1500mm - LO2",
    "RF_Q - 1000mm - DC(-10dB) - In2A",

    "qubit drive rx",
    "M3202A_#4-1 - VLFX-400+ - 500mm - 10dB - 10dB - Q_Q",
    "M3202A_#4-2 - VLFX-400+ - 500mm - 10dB - 10dB - I_Q",
    "RS - 1500mm - LO4",
    "RF_Q - 1000mm - DC(-10dB) - In3A",

    "fogi drive tx",
    "M3202A_#4-4 - VLFX-400+ -  500mm - 10dB - 3dB - 3dB - Ifogi1",
    "M3202A_#4-3 - VLFX-400+ -  500mm - 10dB - 3dB - 3dB - Qfogi1",
    "MXG_N5183B - 1000mm - LO3",
    "RFout_fogi1 - cntr - 10dB - 3dB - 3dB - (ZX60-24-S+) - 1000mm - VBFZ6260S+ - DC(through) - In2A",

    "fogi drive rx",
    "M3202A_#9-4 - VLFX-400+ - 500mm - 10dB - 3dB - 3dB - Ifogi2",
    "M3202A_#9-3 - VLFX-400+ - 500mm - 10dB - 3dB - 3dB - Qfogi2",
    "MXG_N5183B - 2000mm - LO4",
    "RFout_fogi1 - cntr - 10dB - 3dB - 3dB - (ZX60-24-S+) - 1000mm - VBF6260S+ - DC(through) - In3A",

    "JPA comm",
    "M3202A_#2-3 - VLFX-400+ - 500mm - 20dB - I_J",
    "M3202A_#2-4 - VLFX-400+ - 500mm - 20dB - Q_J",
    # "RFoutC - cntr - 10dB - ZVA213s+ - cntr - 10dB - ZVA213s+ - 1000mm - In1D", #tx
    "RFoutC - cntr - 10dB - ZVA213s+ - cntr - 10dB - ZVA213s+ - 1000mm - In1B", #comm

    "JPA rx",
    "M3202A_#7-1 - 500mm - 20dB - I_fogi",
    "M3202A_#7-2 - 500mm - 20dB - Q_fogi",
    "RFoutC - cntr - 10dB - ZVE323LN-K+ - cntr - 3dB - 3dB - ZVA213s+ - 1000mm - In1C",

    "output tx",
    # "Miteq - Out1B -  1000mm - RFin2A", #tx
    "Miteq - Out1A -  1000mm - RFin2A", #comm
    "IFout1 - 500mm - M3102A_#5-1", 

    "output rx",
    "Miteq - Out2B- 1000mm - RFin1B",
    "IFout1 - 500mm - M3102A_#5-2 ",
    ])

# Frequency settings
# readout_lo_freq_tx =10.27
# readout_freq_tx = 10.152
readout_lo_freq_tx =9.5
readout_freq_tx = 9.38
readout_if_freq_tx = readout_lo_freq_tx - readout_freq_tx

readout_lo_freq_rx = 10.29
readout_freq_rx = 10.414
readout_if_freq_rx = readout_lo_freq_rx - readout_freq_rx

if measure_which == "tx" or measure_which == "txQrx": demo_if = readout_if_freq_tx
elif measure_which == "rx": demo_if = readout_if_freq_rx
elif measure_which == "both": demo_if = readout_if_freq_tx

qubit_lo_freq_tx = 8
ge_freq_tx =  8.207+0.000405764-1.388e-05-1.684e-05+4.0565e-05
anharmonicity_tx = -0.3515-0.00038774-1.309e-05
ge_if_freq_tx = ge_freq_tx-qubit_lo_freq_tx
print(f"(tx) ge_freq        :{ge_freq_tx:.6f} GHz")
print(f"(tx) anharmonicity  :{anharmonicity_tx:.6f} GHz")

qubit_lo_freq_rx = 7.8
ge_freq_rx = 7.976+0.001277485-2.569e-05
anharmonicity_rx = -0.3558-9.583e-05-4.83e-06-1.066e-05-0.00030885+2.890e-05+3.984e-05
ge_if_freq_rx = ge_freq_rx-qubit_lo_freq_rx
print(f"(rx) ge_freq        :{ge_freq_rx:.6f} GHz")
print(f"(rx) anharmonicity  :{anharmonicity_rx:.6f} GHz")

fogi_lo_freq_tx = 2 * qubit_lo_freq_tx - 9.5 #6.5
fogi_reset_freq_tx = 6.75 # amp=0.5
fogi_if_freq_tx = fogi_reset_freq_tx - fogi_lo_freq_tx # determined by reset pulse
print(f"(tx) fogi lo        :{fogi_lo_freq_tx:.4f} GHz")

fogi_lo_freq_rx = 2 * qubit_lo_freq_rx - 9.5 #6.1
fogi_reset_freq_rx = 6.2 # amp=0.5
fogi_if_freq_rx = fogi_reset_freq_rx - fogi_lo_freq_rx # determined by reset pulse
print(f"(rx) fogi lo        :{fogi_lo_freq_rx:.4f} GHz")

# Ports
readout_port_tx = Port("readout_port_tx", if_freq = readout_if_freq_tx)
qubit_drive_port_tx = Port("qubit_drive_port_tx", if_freq = ge_if_freq_tx)
fogi_port_tx = Port('fogi_tx', if_freq=fogi_if_freq_tx)

readout_port_rx = Port("readout_port_rx", if_freq = readout_if_freq_rx)
qubit_drive_port_rx = Port("qubit_drive_port_rx", if_freq = ge_if_freq_rx)
fogi_port_rx = Port('fogi_rx', if_freq=fogi_if_freq_rx)
dig_port_tx = Port('dig_tx')
dig_port_rx = Port('dig_rx')
JPA_port_tx = Port('JPA_port_tx', if_freq=readout_freq_tx * 2  - readout_lo_freq_tx * 2)
JPA_port_rx = Port('JPA_port_rx', if_freq=readout_freq_rx * 2 - readout_lo_freq_rx * 2)
ports_tx = [dig_port_tx, JPA_port_tx, readout_port_tx, qubit_drive_port_tx, fogi_port_tx]
ports_tx_wo_JPA = [dig_port_tx, readout_port_tx, qubit_drive_port_tx, fogi_port_tx]

ports_rx_wo_JPA = [dig_port_rx, readout_port_tx, qubit_drive_port_rx, fogi_port_rx]
# ports_rx = [dig_port_rx, JPA_port_rx, readout_port_rx, qubit_drive_port_rx, fogi_port_rx]
ports_rx = [qubit_drive_port_rx, fogi_port_rx]
# ports_rx_wo_JPA = [dig_port_rx, readout_port_rx, qubit_drive_port_rx, fogi_port_rx]
ports_txQrx = [dig_port_tx, JPA_port_tx, readout_port_tx, qubit_drive_port_rx, fogi_port_rx]
ports_txQrx_wo_JPA = [dig_port_tx, readout_port_tx, qubit_drive_port_rx, fogi_port_rx]
ports = ports_tx + ports_rx
# ports_for_comm = ports_tx + [qubit_drive_port_rx, fogi_port_rx]

# readout threshold
try: 
    if measure_which == "tx" or measure_which == "both":
        _, datadict = search_datadict_miyamura(data_path, "2025-08-13", 
                        acquire_time="095617", name="hold") # op readout
        # _, datadict = search_datadict_miyamura(data_dir, "2024-07-28", 
        #                 acquire_time="205920", name="hold") # op photon
        mean_g_tx = datadict["pulse_g"]["values"].ravel()
        mean_e_tx = datadict["pulse_e"]["values"].ravel()
        ein_vec_tx = (mean_e_tx - mean_g_tx)/np.sum((mean_e_tx - mean_g_tx)**2)
        print("got threshold for tx", ein_vec_tx.shape)
    # elif measure_which == "txQrx":
    #     _, datadict = search_datadict_miyamura(data_path, "2024-07-28", 
    #                     acquire_time="205348", name="hold") # op photon
    #     print("got threshold for txQrx")
    #     mean_g_tx = datadict["pulse_g"]["values"].ravel()
    #     mean_e_tx = datadict["pulse_e"]["values"].ravel()
    #     ein_vec_tx = (mean_e_tx - mean_g_tx)/np.sum((mean_e_tx - mean_g_tx)**2)

    if measure_which == "rx" or measure_which == "both":
        _, datadict = search_datadict_miyamura(data_path, "2025-08-13", 
                        acquire_time="105133", name="hold")
        mean_g_rx = datadict["pulse_g"]["values"].ravel()
        mean_e_rx = datadict["pulse_e"]["values"].ravel()
        ein_vec_rx = (mean_e_rx - mean_g_rx)/np.sum((mean_e_rx - mean_g_rx)**2)
        print("got threshold for rx", ein_vec_rx.shape)
except FileNotFoundError:
    print("SingleShot file not found")
finally:
    pass

# readout pulse
if measure_which=="tx" or measure_which=="both": readout_pulse_tx = Square(amplitude=0.5, duration=121) # tx
else:                   readout_pulse_tx = Square(amplitude=0.6, duration=120) # txQrx
dig_acquire_tx = Acquire(readout_pulse_tx.params["duration"])
readout_seq_tx = Sequence(port_list=[readout_port_tx, dig_port_tx])
readout_seq_tx.trigger(ports_tx) 
readout_seq_tx.add(ResetPhase(0), readout_port_tx, copy=False)
readout_seq_tx.add(readout_pulse_tx, readout_port_tx, copy=False)
readout_seq_tx.add(dig_acquire_tx, dig_port_tx, copy=False)
readout_seq_tx.trigger([readout_port_tx, dig_port_tx])
readout_seq_tx.add(Delay(10), readout_port_tx, copy=False)

readout_pulse_rx = Square(amplitude=1.0, duration=120)
dig_acquire_rx = Acquire(readout_pulse_rx.params["duration"])
readout_seq_rx = Sequence(port_list=[readout_port_rx, dig_port_rx])
readout_seq_rx.trigger(ports_rx)
readout_seq_rx.add(ResetPhase(0), readout_port_rx, copy=False)
readout_seq_rx.add(readout_pulse_rx, readout_port_rx, copy=False)
readout_seq_rx.add(dig_acquire_rx, dig_port_rx, copy=False)
readout_seq_rx.trigger([readout_port_rx, dig_port_rx])
readout_seq_rx.add(Delay(10), readout_port_rx, copy=False)
# readout_seq.draw()


JPA_phase_tx = ResetPhase(1.52 * np.pi) # for BW
if measure_which=="tx" or measure_which=="both": JPA_pump_tx=Square(amplitude=0.67, duration=121)
else:                   JPA_pump_tx=Square(amplitude=0.7, duration=120)
seq_JPA_tx = Sequence(port_list=[JPA_port_tx])
seq_JPA_tx.add(SetDetuning(0), JPA_port_tx)
seq_JPA_tx.add(JPA_phase_tx, JPA_port_tx, copy=False)
seq_JPA_tx.add(JPA_pump_tx, JPA_port_tx, copy=False)

JPA_phase_rx = ResetPhase(0.44 * np.pi)
JPA_pump_rx=Square(amplitude=0.96, duration=120)
seq_JPA_rx = Sequence(port_list=[JPA_port_rx])
seq_JPA_rx.add(SetDetuning(0), JPA_port_rx)
seq_JPA_rx.add(JPA_phase_rx, JPA_port_rx, copy=False)
seq_JPA_rx.add(JPA_pump_rx, JPA_port_rx, copy=False)

readout_ss_seq_tx = Sequence(port_list=[readout_port_tx, JPA_port_tx, dig_port_tx])
readout_ss_seq_tx.trigger([readout_port_tx, JPA_port_tx, dig_port_tx])
readout_ss_seq_tx.call(readout_seq_tx)
readout_ss_seq_tx.call(seq_JPA_tx)
readout_ss_seq_tx.trigger([readout_port_tx, JPA_port_tx, dig_port_tx])
readout_ss_seq_tx.add(Delay(96), readout_port_tx)

readout_ss_seq_rx = Sequence(port_list=[readout_port_rx, JPA_port_rx, dig_port_rx])
readout_ss_seq_rx.trigger([readout_port_rx, JPA_port_rx, dig_port_rx])
readout_ss_seq_rx.call(readout_seq_rx)
readout_ss_seq_rx.call(seq_JPA_rx)
readout_ss_seq_rx.trigger([readout_port_rx, JPA_port_rx, dig_port_rx])
readout_ss_seq_rx.add(Delay(96), readout_port_rx)

# # probe sequence
# acquire_length_tx = 300
# probe_seq_tx = Sequence(port_list=[readout_port_tx, dig_port_tx])
# probe_pulse_tx = Square(amplitude=0.3, duration=1000)
# probe_seq_tx.trigger([readout_port_tx, dig_port_tx])
# probe_seq_tx.add(ResetPhase(0), readout_port_tx)
# probe_seq_tx.add(probe_pulse_tx, readout_port_tx)
# probe_seq_tx.add(Acquire(probe_pulse_tx.params["duration"]), dig_port_tx)

# acquire_length_rx = 300
# probe_seq_rx = Sequence(port_list=[readout_port_rx, dig_port_rx])
# probe_pulse_rx = Square(amplitude=0.3, duration=1000)
# probe_seq_rx.trigger([readout_port_rx, dig_port_rx])
# probe_seq_rx.add(ResetPhase(0), readout_port_rx)
# probe_seq_rx.add(probe_pulse_rx, readout_port_rx)
# probe_seq_rx.add(Acquire(probe_pulse_rx.params["duration"]), dig_port_rx)

# ge pi pulse
ge_pi_pulse_tx = Gaussian(amplitude=1.4233642, fwhm=20, duration=48, zero_end=True)
ge_pi_pulse_drag_tx = HalfDRAG(ge_pi_pulse_tx, beta=0.2144378) #1.33
ge_pi_seq_tx = Sequence()
ge_pi_seq_tx.add(ge_pi_pulse_drag_tx, qubit_drive_port_tx, copy=False)

ge_pi_pulse_rx = Gaussian(amplitude=0.5215119529, fwhm=20, duration=48, zero_end=True)
ge_pi_pulse_drag_rx = HalfDRAG(ge_pi_pulse_rx, beta=-0.170448215) #1.33
ge_pi_seq_rx = Sequence()
ge_pi_seq_rx.add(ge_pi_pulse_drag_rx, qubit_drive_port_rx, copy=False)

# ge half pi pulse
ge_half_pi_pulse_tx = Gaussian(amplitude=ge_pi_pulse_tx.params['amplitude']/2, fwhm=ge_pi_pulse_tx.params["fwhm"], duration=ge_pi_pulse_tx.params["duration"], zero_end=True)
ge_half_pi_pulse_drag_tx = HalfDRAG(ge_half_pi_pulse_tx, beta=0.2862999) # 1.35
ge_half_pi_seq_tx = Sequence()
ge_half_pi_seq_tx.add(ge_half_pi_pulse_drag_tx, qubit_drive_port_tx, copy=False)

ge_half_pi_pulse_rx = Gaussian(amplitude=ge_pi_pulse_rx.params['amplitude']/2, fwhm=ge_pi_pulse_rx.params["fwhm"], duration=ge_pi_pulse_rx.params["duration"], zero_end=True)
ge_half_pi_pulse_drag_rx = HalfDRAG(ge_half_pi_pulse_rx, beta=-0.174468334) # 1.35
ge_half_pi_seq_rx = Sequence()
ge_half_pi_seq_rx.add(ge_half_pi_pulse_drag_rx, qubit_drive_port_rx, copy=False)

# ef pi
ef_pi_pulse_tx = Gaussian(amplitude=1.3218133318, fwhm=20, duration=48, zero_end=True)
ef_pi_pulse_drag_tx = HalfDRAG(ef_pi_pulse_tx, beta=0.7068233)
ef_pi_seq_tx = Sequence(port_list=[qubit_drive_port_tx])
ef_pi_seq_tx.add(SetDetuning(anharmonicity_tx), qubit_drive_port_tx)
ef_pi_seq_tx.add(ef_pi_pulse_drag_tx, qubit_drive_port_tx, copy=False)
ef_pi_seq_tx.add(SetDetuning(0), qubit_drive_port_tx)

ef_pi_pulse_rx = Gaussian(amplitude=0.3205452420, fwhm=20, duration=48, zero_end=True)
ef_pi_pulse_drag_rx = HalfDRAG(ef_pi_pulse_rx, beta=0.471238506)
ef_pi_seq_rx = Sequence(port_list=[qubit_drive_port_rx])
ef_pi_seq_rx.add(SetDetuning(anharmonicity_rx), qubit_drive_port_rx)
ef_pi_seq_rx.add(ef_pi_pulse_drag_rx, qubit_drive_port_rx, copy=False)
ef_pi_seq_rx.add(SetDetuning(0), qubit_drive_port_rx)

# ef half pi pulse
ef_half_pi_pulse_tx = Gaussian(amplitude=ef_pi_pulse_tx.params['amplitude']/2, fwhm=ef_pi_pulse_tx.params['fwhm'], duration=ef_pi_pulse_tx.params['duration'], zero_end=True)
ef_half_pi_pulse_drag_tx = HalfDRAG(ef_half_pi_pulse_tx, beta=0.6653118)
ef_half_pi_seq_tx = Sequence()
ef_half_pi_seq_tx.add(SetDetuning(anharmonicity_tx), qubit_drive_port_tx)
ef_half_pi_seq_tx.add(ef_half_pi_pulse_drag_tx, qubit_drive_port_tx, copy=False)
ef_half_pi_seq_tx.add(SetDetuning(0), qubit_drive_port_tx)

ef_half_pi_pulse_rx = Gaussian(amplitude=ef_pi_pulse_rx.params['amplitude']/2, fwhm=ef_pi_pulse_rx.params['fwhm'], duration=ef_pi_pulse_rx.params['duration'], zero_end=True)
ef_half_pi_pulse_drag_rx = HalfDRAG(ef_half_pi_pulse_rx, beta=0.413604716)
ef_half_pi_seq_rx = Sequence()
ef_half_pi_seq_rx.add(SetDetuning(anharmonicity_rx), qubit_drive_port_rx)
ef_half_pi_seq_rx.add(ef_half_pi_pulse_drag_rx, qubit_drive_port_rx, copy=False)
ef_half_pi_seq_rx.add(SetDetuning(0), qubit_drive_port_rx)

# Reset pulse
Appeared_anhm_tx = SetDetuning(-0.36404091)
reset_pulse_ef_tx = FlatTop(Gaussian(amplitude=0.06, fwhm=10, duration=20), top_duration=1000)
reset_pulse_fogi_tx = FlatTop(Gaussian(amplitude=0.2, fwhm=10, duration=20), top_duration=1000)
reset_sequence_tx = Sequence()
reset_sequence_tx.add(Appeared_anhm_tx, qubit_drive_port_tx, copy=False)
reset_sequence_tx.add(reset_pulse_ef_tx, qubit_drive_port_tx, copy=False) # qubit drive to e-f
reset_sequence_tx.add(reset_pulse_fogi_tx, fogi_port_tx, copy=False) # f0-g1 drive
reset_sequence_tx.trigger([qubit_drive_port_tx, fogi_port_tx])
reset_sequence_tx.add(SetDetuning(0), qubit_drive_port_tx, copy=False)
reset_sequence_tx.add(SetDetuning(0), fogi_port_tx, copy=False)
reset_sequence_tx.add(ResetPhase(), qubit_drive_port_tx)
reset_sequence_tx.add(ResetPhase(), fogi_port_tx)

Appeared_anhm_rx = SetDetuning(-0.36485747)
reset_pulse_ef_rx = FlatTop(Gaussian(amplitude=0.03, fwhm=10, duration=20), top_duration=1000)
reset_pulse_fogi_rx = FlatTop(Gaussian(amplitude=0.2, fwhm=10, duration=20), top_duration=1000)
reset_sequence_rx = Sequence()
reset_sequence_rx.add(Appeared_anhm_rx, qubit_drive_port_rx, copy=False)
reset_sequence_rx.add(reset_pulse_ef_rx, qubit_drive_port_rx, copy=False) # qubit drive to e-f
reset_sequence_rx.add(reset_pulse_fogi_rx, fogi_port_rx, copy=False) # f0-g1 drive
reset_sequence_rx.trigger([qubit_drive_port_rx, fogi_port_rx])
reset_sequence_rx.add(SetDetuning(0), qubit_drive_port_rx, copy=False)
reset_sequence_rx.add(SetDetuning(0), fogi_port_rx, copy=False)
reset_sequence_rx.add(ResetPhase(), qubit_drive_port_rx)
reset_sequence_rx.add(ResetPhase(), fogi_port_rx)

# def minus_sequence(seq:Sequence):
#     assert len(seq.port_list)==1, (f"number of ports in the given sequence must be 1 but obtained {len(seq.port_list)}")
#     minus_seq = Sequence()
#     minus_seq.add(VirtualZ(pi), seq.port_list[0])
#     minus_seq.call(seq)
#     minus_seq.add(VirtualZ(pi), seq.port_list[0])
#     minus_seq.compile()
#     return minus_seq