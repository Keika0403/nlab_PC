import numpy as np
import matplotlib.pyplot as plt
import qcodes as qc
from plottr.data.datadict_storage import DataDict, DDH5Writer
from qcodes_drivers.HVI_Trigger import HVI_Trigger
from qcodes_drivers.M3202A import M3202A, SD_AWG_CHANNEL
from qcodes_drivers.M3102A import M3102A#, SD_DIG_CHANNEL
from qcodes_drivers.SD_common.SD_DIG import SD_DIG_CHANNEL
from qcodes_drivers.E82x7 import E82x7 as E8247
from qcodes_drivers.N51x1 import N51x1 
from qcodes_drivers.SGS100A import RohdeSchwarz_SGS100A as RS
from sequence_parser import Sequence
from sequence_parser.variable import Variable, Variables
from sequence_parser.instruction import *
from setup_td_parameters import *
from qcodes_drivers.iq_corrector import IQCorrector
from tqdm import tqdm
from typing import Sequence as Seq

setup_file = __file__

# Instruments
station = qc.Station()

awg1:M3202A = M3202A('awg1', chassis=1, slot=2)
awg_readout_tx:SD_AWG_CHANNEL = awg1.ch1
awg_readout_rx:SD_AWG_CHANNEL = awg1.ch2
awg_JPA_I_tx:SD_AWG_CHANNEL = awg1.ch3
awg_JPA_Q_tx:SD_AWG_CHANNEL = awg1.ch4
station.add_component(awg1)

awg2:M3202A = M3202A('awg2', 1, 3)
awg_Qdrive_I_tx:SD_AWG_CHANNEL = awg2.ch1
awg_Qdrive_Q_tx:SD_AWG_CHANNEL = awg2.ch2
awg_fogi_I_tx:SD_AWG_CHANNEL = awg2.ch3
awg_fogi_Q_tx:SD_AWG_CHANNEL = awg2.ch4
station.add_component(awg2)

awg3:M3202A = M3202A('awg3', 1, 4)
awg_Qdrive_I_rx:SD_AWG_CHANNEL = awg3.ch1
awg_Qdrive_Q_rx:SD_AWG_CHANNEL = awg3.ch2
awg_fogi_I_rx:SD_AWG_CHANNEL = awg3.ch3
awg_fogi_Q_rx:SD_AWG_CHANNEL = awg3.ch4
station.add_component(awg3)

awg4:M3202A = M3202A('awg4', 1, 7)
awg_JPA_I_rx:SD_AWG_CHANNEL = awg4.ch1
awg_JPA_Q_rx:SD_AWG_CHANNEL = awg4.ch2
station.add_component(awg4)

hvi_trigger:HVI_Trigger = HVI_Trigger('hvi_trigger', "PXI0::1::BACKPLANE")
station.add_component(hvi_trigger)
hvi_trigger.digitizer_delay(400)
hvi_trigger.trigger_period(100000) # ns

dig:M3102A = M3102A('digitizer', chassis=1, slot=5)
station.add_component(dig)
dig_ch_tx:SD_DIG_CHANNEL = dig.ch1
dig_ch_tx.high_impedance(False)  # 50 Ohms
dig_ch_tx.half_range_50(0.5)  # V_pp / 2
dig_ch_tx.ac_coupling(False)  # dc coupling
dig_ch_tx.sampling_interval(2); sampling = dig_ch_tx.sampling_interval()  # ns
dig_ch_tx.trigger_mode("software/hvi")
dig_ch_tx.timeout(10000)  # ms                                                           
voltage_step_tx=dig_ch_tx.half_range_50()/(2**15-1)

dig_ch_rx:SD_DIG_CHANNEL = dig.ch2
dig_ch_rx.high_impedance(False)  # 50 Ohms
dig_ch_rx.half_range_50(0.5)  # V_pp / 2
dig_ch_rx.ac_coupling(False)  # dc coupling
dig_ch_rx.sampling_interval(2); sampling = dig_ch_rx.sampling_interval()  # ns
dig_ch_rx.trigger_mode("software/hvi")
dig_ch_rx.timeout(10000)  # ms                                                           
voltage_step_rx=voltage_step_tx#dig_ch_rx.half_range_50()/(2**15-1)

# tx readout
lo1:E8247 = E8247('lo1', 'TCPIP0::192.168.100.240::inst0::INSTR')
lo1.power(24)
lo1.frequency(readout_lo_freq_tx*1e9)
station.add_component(lo1)

# tx qubit drive
lo2:N51x1 = N51x1('lo2', 'TCPIP0::192.168.100.9::inst0::INSTR')
# lo2:RS = RS('lo2', 'TCPIP0::192.168.100.136::inst0::INSTR')
lo2.frequency(qubit_lo_freq_tx*1e9)
lo2.power(17)
station.add_component(lo2)

# tx fogi drive
lo3:E8247 = N51x1('lo3', 'TCPIP0::192.168.100.7::inst0::INSTR')
lo3.frequency(fogi_lo_freq_tx*1e9) 
lo3.power(13)
station.add_component(lo3)

# rx qubit drive
lo4:N51x1 = N51x1('lo4', 'TCPIP0::192.168.100.49::inst0::INSTR')
# lo4:RS = RS('lo4', 'TCPIP0::192.168.100.143::inst0::INSTR')
lo4.frequency(qubit_lo_freq_rx*1e9)
lo4.power(13)
station.add_component(lo4)

# rx fogi drive
lo5:N51x1 = E8247('lo5', 'TCPIP0::192.168.100.93::inst0::INSTR')
lo5.frequency(fogi_lo_freq_rx*1e9) 
lo5.power(16)
station.add_component(lo5)

# rx readout
lo6:E8247 = E8247('lo6', 'TCPIP0::192.168.100.101::inst0::INSTR')
lo6.frequency(readout_lo_freq_rx*1e9) 
lo6.power(22)
station.add_component(lo6)

from qcodes.instrument_drivers.yokogawa.GS200 import GS200

yoko:GS200 = GS200("yoko", "TCPIP0::192.168.100.213::inst0::INSTR")
station.add_component(yoko)
if yoko.state():
    print('yoko was on')
    pass
else:
    yoko.ramp_current(0, step=1e-8, delay=0)
    yoko.on()
    print('yoko was off')
assert yoko.state()
yoko.ramp_current(96e-6, step=1e-8, delay=0)  # photon 0.65

yoko_tx:GS200 = GS200("yoko_tx", "TCPIP0::192.168.100.95::inst0::INSTR")
station.add_component(yoko_tx)
if yoko_tx.state():
    print('yoko_tx was on')
    pass
else:
    yoko_tx.ramp_current(0, step=1e-8, delay=0)
    yoko_tx.on()
    print('yoko_tx was off')
assert yoko_tx.state()
yoko_tx.ramp_current(112e-6, step=1e-8, delay=0) # tx readout, current99, amp: 1.24

yoko_rx:GS200 = GS200("yoko_rx", "TCPIP0::192.168.100.99::inst0::INSTR")
station.add_component(yoko_rx)
if yoko_rx.state():
    print('yoko_rx was on')
    pass
else:
    yoko_rx.ramp_current(0, step=1e-8, delay=0)
    yoko_rx.on()
    print('yoko_rx was off')
assert yoko_rx.state()
yoko_rx.ramp_current(92e-6, step=1e-8, delay=0) 

# IQ calibration
iq_corrector_fogi_tx = IQCorrector(
    awg_fogi_I_tx,
    awg_fogi_Q_tx,
    data_path,
    lo_leakage_datetime="2025-08-10T132339",
    rf_power_datetime="2025-08-10T133158",
    len_kernel=41,
    fit_weight=10,
)

iq_corrector_q_tx = IQCorrector(
    awg_Qdrive_I_tx,
    awg_Qdrive_Q_tx,
    data_path,
    lo_leakage_datetime="2025-08-10T124348",
    rf_power_datetime="2025-08-10T125235",
    len_kernel=41,
    fit_weight=10,
)

iq_corrector_fogi_rx = IQCorrector(
    awg_fogi_I_rx,
    awg_fogi_Q_rx,
    data_path,
    lo_leakage_datetime="2025-08-10T134005",
    rf_power_datetime="2025-08-10T134801",
    len_kernel=41,
    fit_weight=10,
)

iq_corrector_q_rx = IQCorrector(
    awg_Qdrive_I_rx,
    awg_Qdrive_Q_rx,
    data_path,
    lo_leakage_datetime="2025-08-10T130233",
    rf_power_datetime="2025-08-10T131045",
    len_kernel=41,
    fit_weight=10,
)

iq_corrector_JPA_tx = IQCorrector(
    awg_JPA_I_tx,
    awg_JPA_Q_tx,
    data_path,
    # lo_leakage_datetime="2025-08-11T094928", #comm
    # rf_power_datetime="2025-08-11T095713", #comm
    lo_leakage_datetime="2025-08-11T100853",
    rf_power_datetime="2025-08-11T101638",
    len_kernel=41,
    fit_weight=10,
)

iq_corrector_JPA_rx = IQCorrector(
    awg_JPA_I_rx,
    awg_JPA_Q_rx,
    data_path,
    lo_leakage_datetime="2025-08-11T180649",
    rf_power_datetime="2025-08-11T181527",
    len_kernel=41,
    fit_weight=10,
)


# functions
def load_sequence(sequence: Sequence, cycles: int):
    sequence.compile()
    awg1.stop_all() ;awg1.flush_waveform()
    awg2.stop_all() ;awg2.flush_waveform()
    awg3.stop_all() ;awg3.flush_waveform()
    awg4.stop_all() ;awg4.flush_waveform()
    if readout_port_tx in sequence.port_list:
        awg1.load_waveform(readout_port_tx.waveform.real, 0, append_zeros=True)
        awg_readout_tx.queue_waveform(0, trigger="software/hvi", cycles=cycles)
    if readout_port_rx in sequence.port_list:
        awg1.load_waveform(readout_port_rx.waveform.real, 1, append_zeros=True)
        awg_readout_rx.queue_waveform(1, trigger="software/hvi", cycles=cycles)
    if qubit_drive_port_tx in sequence.port_list:
        i, q = iq_corrector_q_tx.correct(qubit_drive_port_tx.waveform)
        awg2.load_waveform(i, 1, append_zeros=True)
        awg2.load_waveform(q, 2, append_zeros=True)
        awg_Qdrive_I_tx.queue_waveform(1, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_Q_tx.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_I_tx.dc_offset(iq_corrector_q_tx.i_offset)
        awg_Qdrive_Q_tx.dc_offset(iq_corrector_q_tx.q_offset)
    if fogi_port_tx in sequence.port_list:
        i, q = iq_corrector_fogi_tx.correct(fogi_port_tx.waveform)
        awg2.load_waveform(i, 3, append_zeros=True)
        awg2.load_waveform(q, 4, append_zeros=True)
        awg_fogi_I_tx.queue_waveform(3, trigger="software/hvi", cycles=cycles)
        awg_fogi_Q_tx.queue_waveform(4, trigger="software/hvi", cycles=cycles)
        awg_fogi_I_tx.dc_offset(iq_corrector_fogi_tx.i_offset)
        awg_fogi_Q_tx.dc_offset(iq_corrector_fogi_tx.q_offset)
    if qubit_drive_port_rx in sequence.port_list:
        i, q = iq_corrector_q_rx.correct(qubit_drive_port_rx.waveform)
        awg3.load_waveform(i, 1, append_zeros=True)
        awg3.load_waveform(q, 2, append_zeros=True)
        awg_Qdrive_I_rx.queue_waveform(1, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_Q_rx.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_I_rx.dc_offset(iq_corrector_q_rx.i_offset)
        awg_Qdrive_Q_rx.dc_offset(iq_corrector_q_rx.q_offset)
    if fogi_port_rx in sequence.port_list:
        i, q = iq_corrector_fogi_rx.correct(fogi_port_rx.waveform)
        awg3.load_waveform(i, 3, append_zeros=True)
        awg3.load_waveform(q, 4, append_zeros=True)
        awg_fogi_I_rx.queue_waveform(3, trigger="software/hvi", cycles=cycles)
        awg_fogi_Q_rx.queue_waveform(4, trigger="software/hvi", cycles=cycles)
        awg_fogi_I_rx.dc_offset(iq_corrector_fogi_rx.i_offset)
        awg_fogi_Q_rx.dc_offset(iq_corrector_fogi_rx.q_offset)
    if JPA_port_tx in sequence.port_list:
        i, q = iq_corrector_JPA_tx.correct(JPA_port_tx.waveform)
        awg1.load_waveform(i, 2, append_zeros=True)
        awg1.load_waveform(q, 3, append_zeros=True)
        awg_JPA_I_tx.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_JPA_Q_tx.queue_waveform(3, trigger="software/hvi", cycles=cycles)
        awg_JPA_I_tx.dc_offset(iq_corrector_JPA_tx.i_offset)
        awg_JPA_Q_tx.dc_offset(iq_corrector_JPA_tx.q_offset)
    if JPA_port_rx in sequence.port_list:
        i, q = iq_corrector_JPA_rx.correct(JPA_port_rx.waveform)
        awg4.load_waveform(i, 1, append_zeros=True)
        awg4.load_waveform(q, 2, append_zeros=True)
        awg_JPA_I_rx.queue_waveform(1, trigger="software/hvi", cycles=cycles)
        awg_JPA_Q_rx.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_JPA_I_rx.dc_offset(iq_corrector_JPA_rx.i_offset)
        awg_JPA_Q_rx.dc_offset(iq_corrector_JPA_rx.q_offset)
    if dig_port_tx in sequence.port_list:
        if len(dig_port_tx.measurement_windows) == 0:
            acquire_start = 0
        else:
            acquire_start=dig_port_tx.measurement_windows[0][0]
            acquire_end=dig_port_tx.measurement_windows[-1][-1]
            dig_ch_tx.points_per_cycle(int((acquire_end-acquire_start)/dig_ch_tx.sampling_interval()))
        # print("sampling_interval:", dig_ch_tx.sampling_interval())
        # print("points_per_cycle:", dig_ch_tx.points_per_cycle())
        # print(dig_ch_tx.points_per_cycle() % dig_ch_tx.sampling_interval())
        assert dig_ch_tx.points_per_cycle() % dig_ch_tx.sampling_interval() == 0
        dig_ch_tx.delay(int(acquire_start//dig_ch_tx.sampling_interval()))
        dig_ch_tx.cycles(cycles)
    if dig_port_rx in sequence.port_list:
        if len(dig_port_rx.measurement_windows) == 0:
            acquire_start = 0
        else:
            acquire_start=dig_port_rx.measurement_windows[0][0]
            acquire_end=dig_port_rx.measurement_windows[-1][-1]
            dig_ch_rx.points_per_cycle(int((acquire_end-acquire_start)/dig_ch_rx.sampling_interval()))
        assert dig_ch_rx.points_per_cycle() % dig_ch_rx.sampling_interval() == 0
        dig_ch_rx.delay(int(acquire_start//dig_ch_rx.sampling_interval()))
        dig_ch_rx.cycles(cycles)

def run(sequence:Sequence, plot=False, lo_on=True, which=measure_which):
    hvi_trigger.output(False)
    if readout_port_tx in sequence.port_list:
        if lo_on:lo1.output(True)
        assert dig_ch_tx.cycles()==awg_readout_tx.cycles()[-1], f'{dig_ch_tx.cycles()}, {awg_readout_tx.cycles()}'
        awg_readout_tx.start()
    if readout_port_rx in sequence.port_list:
        if lo_on:lo6.output(True)
        assert dig_ch_rx.cycles()==awg_readout_rx.cycles()[-1], f'{dig_ch_rx.cycles()}, {awg_readout_rx.cycles()}'
        awg_readout_rx.start()
    if qubit_drive_port_tx in sequence.port_list:
        assert dig_ch_tx.cycles()==awg_Qdrive_I_tx.cycles()[-1], f'{dig_ch_tx.cycles()}, {awg_Qdrive_I_tx.cycles()}'
        if lo_on:lo2.output(True)
        awg_Qdrive_I_tx.start()
        awg_Qdrive_Q_tx.start()
        if fogi_port_tx in sequence.port_list:
            dig_ch_tx.cycles()==awg_fogi_I_tx.cycles()[-1], f'{dig_ch_tx.cycles()}, {awg_fogi_I_tx.cycles()}'
            if lo_on:lo3.output(True)
            awg_fogi_I_tx.start()
            awg_fogi_Q_tx.start()
    if qubit_drive_port_rx in sequence.port_list:
        assert dig_ch_tx.cycles()==awg_Qdrive_I_rx.cycles()[-1], f'{dig_ch_rx.cycles()}, {awg_Qdrive_I_rx.cycles()}'
        if lo_on:lo4.output(True)
        awg_Qdrive_I_rx.start()
        awg_Qdrive_Q_rx.start()
        if fogi_port_rx in sequence.port_list:
            dig_ch_rx.cycles()==awg_fogi_I_rx.cycles()[-1], f'{dig_ch_rx.cycles()}, {awg_fogi_I_rx.cycles()}'
            if lo_on:lo5.output(True)
            awg_fogi_I_rx.start()
            awg_fogi_Q_rx.start()
    if JPA_port_tx in sequence.port_list:
        if lo_on:lo1.output(True)
        assert dig_ch_tx.cycles()==awg_JPA_I_tx.cycles()[-1], f'{dig_ch_tx.cycles()}, {awg_JPA_I_tx.cycles()}'
        awg_JPA_I_tx.start()
        awg_JPA_Q_tx.start()
    if JPA_port_rx in sequence.port_list:
        if lo_on:lo6.output(True)
        assert dig_ch_rx.cycles()==awg_JPA_I_rx.cycles()[-1], f'{dig_ch_rx.cycles()}, {awg_JPA_I_rx.cycles()}'
        awg_JPA_I_rx.start()
        awg_JPA_Q_rx.start()
    dig_ch_tx.start()
    dig_ch_rx.start()
    hvi_trigger.output(True)
    data_rx = dig_ch_rx.read()
    data_tx = dig_ch_tx.read()
    hvi_trigger.output(False)
    if plot:
        plt.figure("tx")
        plt.plot(dig_ch_tx.sampling_interval() * np.arange(len(data_tx[0])), data_tx.mean(axis=0) * voltage_step_tx)
        plt.figure("rx")
        plt.plot(dig_ch_rx.sampling_interval() * np.arange(len(data_rx[0])), data_rx.mean(axis=0) * voltage_step_rx)
        plt.show()
    if which=="both":
        return data_tx, data_rx
    elif which[:2] == "tx":
        return data_tx
    elif which[:2] == "rx":
        return data_rx
    else: raise NotImplementedError

def demodulate(data, demodulation_if=demo_if):
    t = np.arange(data.shape[-1]) * dig_ch_tx.sampling_interval()
    return (data * np.exp(2j * np.pi * demodulation_if * t)).mean(axis=-1)

def off():
    awg1.stop_all()
    awg2.stop_all()
    awg3.stop_all()
    awg4.stop_all()
    hvi_trigger.output(False)
    dig_ch_tx.stop()
    dig_ch_rx.stop()
    lo1.output(False)
    lo2.output(False)
    lo3.output(False)
    lo4.output(False)
    lo5.output(False)
    lo6.output(False)

def load_note(data_path, date, name):
    lines = []
    with open(f'{data_path}\\{date}\\{name}', encoding='utf-8') as f:
        lines = f.readlines()  
    return"".join(lines)

def postselection(data:np.ndarray, start:int|float, stop:int|float, which:str=measure_which, which_result="both"):
    if which=="tx":
        DIG = dig_ch_tx ;mean_g = mean_g_tx ;ein_vec = ein_vec_tx
    elif which=="rx":
        DIG = dig_ch_rx ;mean_g = mean_g_rx ;ein_vec = ein_vec_rx
    else:
        raise NotImplementedError(f"{which} is not either 'tx' or 'rx'")
    start_idx = int(start // DIG.sampling_interval())
    end_idx = int(stop // DIG.sampling_interval())
    delete_idx = []
    for i in range(len(data)):
        single_shot = data[i, start_idx:end_idx]
        if np.dot(single_shot - mean_g, ein_vec) > 1/2:
            delete_idx.append(i)
    g_data = np.delete(data, delete_idx, axis=0)
    e_data = data[delete_idx]
    if which_result == "g": return g_data
    elif which_result == "e": return e_data
    elif which_result == "both": return g_data, e_data
    else: raise NotImplementedError(f"{which_result} is not in ['g', 'e', 'both']")

def xpostselection_both_sides(data_tx:np.ndarray, data_rx:np.ndarray, start_tx:int|float, stop_tx:int|float, start_rx:int|float, stop_rx:int|float, which:str="both", which_result="both"):
    assert data_tx.shape[0] == data_rx.shape[0]
    start_tx_idx = int(start_tx // dig_ch_tx.sampling_interval())
    end_tx_idx = int(stop_tx // dig_ch_tx.sampling_interval())
    delete_idx_tx = []
    start_rx_idx = int(start_rx // dig_ch_rx.sampling_interval())
    end_rx_idx = int(stop_rx // dig_ch_rx.sampling_interval())
    delete_idx_rx = []

    for i in range(len(data_tx)):
        single_shot = data_tx[i, start_tx_idx:end_tx_idx]
        if np.dot(single_shot - mean_g_tx, ein_vec_tx) > 1/2:
            delete_idx_tx.append(i)
    for i in range(len(data_rx)):
        single_shot = data_rx[i, start_rx_idx:end_rx_idx]
        if np.dot(single_shot - mean_g_rx, ein_vec_rx) > 1/2:
            delete_idx_rx.append(i)
    delete_idx = np.unique(np.append(delete_idx_tx, delete_idx_rx))
    print(len(delete_idx_tx), len(delete_idx_rx), len(delete_idx))
    gg_data_tx = np.delete(data_tx, delete_idx, axis=0)
    other_tx = data_tx[delete_idx]
    gg_data_rx = np.delete(data_rx, delete_idx, axis=0)
    other_rx = data_rx[delete_idx]
    return gg_data_tx, gg_data_rx, other_tx, other_rx

def append_waveforms(seq:Sequence, append_ports:Seq[Port], waveforms_added:Seq[np.ndarray]|Seq[Seq[np.ndarray]], check:bool=False):
    seq.compile()
    waveform_dict = dict()
    for port in seq.port_list:
        waveform = port.waveform
        windows = port.measurement_windows
        for append_port, waveform_appended in zip(append_ports, waveforms_added):
            if port.name == append_port.name:
                if len(windows) == 1:
                    window = windows[0]
                    try:
                        if window[1] - window[0] == len(waveform_appended):
                            waveform[int(window[0]) : int(window[1])] = waveform_appended
                            if check:
                                print(f"added to : {append_port.name}, {window[0]}~{window[1]}")
                        else:
                            print(f"zeros appended because of a shape mismatch: {append_port.name}, {window[0]}~{window[1]}")
                    except:
                        print(f"addition failed : {append_port.name}, {window[0]}~{window[1]}")
                else:
                    assert len(windows) == len(waveform_appended)
                    for i, window in enumerate(windows):
                        waveform[int(window[0]) : int(window[1])] = waveform_appended[i]
        waveform_dict[port.name] = waveform
    return waveform_dict


def load_sequence_w_append(sequence:Sequence, append_ports:Seq[Port], waveforms_added:Seq[np.ndarray]|Seq[Seq[np.ndarray]], cycles, check:bool=False):
    sequence.compile()
    awg1.stop_all() ;awg1.flush_waveform()
    awg2.stop_all() ;awg2.flush_waveform()
    awg3.stop_all() ;awg3.flush_waveform()
    awg4.stop_all() ;awg4.flush_waveform()
    waveform_dict = append_waveforms(sequence, append_ports, waveforms_added, check=check)
    if readout_port_tx in sequence.port_list:
        awg1.stop_all()
        awg1.flush_waveform()
        awg1.load_waveform(waveform_dict[readout_port_tx.name].real, 0, append_zeros=True)
        awg_readout_tx.queue_waveform(0, trigger="software/hvi", cycles=cycles)
    if readout_port_rx in sequence.port_list:
        awg1.load_waveform(waveform_dict[readout_port_rx.name].real, 1, append_zeros=True)
        awg_readout_rx.queue_waveform(1, trigger="software/hvi", cycles=cycles)
    if qubit_drive_port_tx in sequence.port_list:
        i, q = iq_corrector_q_tx.correct(waveform_dict[qubit_drive_port_tx.name])
        awg2.stop_all()
        awg2.flush_waveform()
        awg2.load_waveform(i, 1, append_zeros=True)
        awg2.load_waveform(q, 2, append_zeros=True)
        awg_Qdrive_I_tx.queue_waveform(1, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_Q_tx.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_I_tx.dc_offset(iq_corrector_q_tx.i_offset)
        awg_Qdrive_Q_tx.dc_offset(iq_corrector_q_tx.q_offset)
    if fogi_port_tx in sequence.port_list:
        i, q = iq_corrector_fogi_tx.correct(waveform_dict[fogi_port_tx.name])
        awg2.load_waveform(i, 3, append_zeros=True)
        awg2.load_waveform(q, 4, append_zeros=True)
        awg_fogi_I_tx.queue_waveform(3, trigger="software/hvi", cycles=cycles)
        awg_fogi_Q_tx.queue_waveform(4, trigger="software/hvi", cycles=cycles)
        awg_fogi_I_tx.dc_offset(iq_corrector_fogi_tx.i_offset)
        awg_fogi_Q_tx.dc_offset(iq_corrector_fogi_tx.q_offset)
    if qubit_drive_port_rx in sequence.port_list:
        i, q = iq_corrector_q_rx.correct(waveform_dict[qubit_drive_port_rx.name])
        awg3.stop_all()
        awg3.flush_waveform()
        awg3.load_waveform(i, 1, append_zeros=True)
        awg3.load_waveform(q, 2, append_zeros=True)
        awg_Qdrive_I_rx.queue_waveform(1, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_Q_rx.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_I_rx.dc_offset(iq_corrector_q_rx.i_offset)
        awg_Qdrive_Q_rx.dc_offset(iq_corrector_q_rx.q_offset)
    if fogi_port_rx in sequence.port_list:
        i, q = iq_corrector_fogi_rx.correct(waveform_dict[fogi_port_rx.name])
        awg3.load_waveform(i, 3, append_zeros=True)
        awg3.load_waveform(q, 4, append_zeros=True)
        awg_fogi_I_rx.queue_waveform(3, trigger="software/hvi", cycles=cycles)
        awg_fogi_Q_rx.queue_waveform(4, trigger="software/hvi", cycles=cycles)
        awg_fogi_I_rx.dc_offset(iq_corrector_fogi_rx.i_offset)
        awg_fogi_Q_rx.dc_offset(iq_corrector_fogi_rx.q_offset)
    if JPA_port_tx in sequence.port_list:
        i, q = iq_corrector_JPA_tx.correct(waveform_dict[JPA_port_tx.name])
        awg1.load_waveform(i, 2, append_zeros=True)
        awg1.load_waveform(q, 3, append_zeros=True)
        awg_JPA_I_tx.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_JPA_Q_tx.queue_waveform(3, trigger="software/hvi", cycles=cycles)
        awg_JPA_I_tx.dc_offset(iq_corrector_JPA_tx.i_offset)
        awg_JPA_Q_tx.dc_offset(iq_corrector_JPA_tx.q_offset)
    if JPA_port_rx in sequence.port_list:
        i, q = iq_corrector_JPA_rx.correct(waveform_dict[JPA_port_rx.name])
        awg4.load_waveform(i, 1, append_zeros=True)
        awg4.load_waveform(q, 2, append_zeros=True)
        awg_JPA_I_rx.queue_waveform(1, trigger="software/hvi", cycles=cycles)
        awg_JPA_Q_rx.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_JPA_I_rx.dc_offset(iq_corrector_JPA_rx.i_offset)
        awg_JPA_Q_rx.dc_offset(iq_corrector_JPA_rx.q_offset)
    if dig_port_tx in sequence.port_list:
        if len(dig_port_tx.measurement_windows) == 0:
            acquire_start = 0
        else:
            acquire_start=dig_port_tx.measurement_windows[0][0]
            acquire_end=dig_port_tx.measurement_windows[-1][-1]
            dig_ch_tx.points_per_cycle(int((acquire_end-acquire_start)/dig_ch_tx.sampling_interval()))
        assert dig_ch_tx.points_per_cycle() % dig_ch_tx.sampling_interval() == 0
        dig_ch_tx.delay(int(acquire_start//dig_ch_tx.sampling_interval()))
        dig_ch_tx.cycles(cycles)
    if dig_port_rx in sequence.port_list:
        if len(dig_port_rx.measurement_windows) == 0:
            acquire_start = 0
        else:
            acquire_start=dig_port_rx.measurement_windows[0][0]
            acquire_end=dig_port_rx.measurement_windows[-1][-1]
            dig_ch_rx.points_per_cycle(int((acquire_end-acquire_start)/dig_ch_rx.sampling_interval()))
        assert dig_ch_rx.points_per_cycle() % dig_ch_rx.sampling_interval() == 0
        dig_ch_rx.delay(int(acquire_start//dig_ch_rx.sampling_interval()))
        dig_ch_rx.cycles(cycles)

def check_sequence_formulti(seq:Sequence, var:Variables, idxlist, port_list=ports):
    for ccc in range(len(idxlist)):
        idx=idxlist[ccc]
        count=0
        for update_command in tqdm(var.update_command_list):
            seq.update_variables(update_command)
            if count==idx:
                seq.draw()
            count=count+1
    raise SystemError

def seq_duration(seq:Sequence):
    port = seq.port_list[0]
    seq.compile()
    return len(port.waveform)

def load_sequence_w_append_comm(sequence:Sequence, append_ports, waveforms_appended, cycles):
    sequence.compile()
    awg1.stop_all() ;awg1.flush_waveform()
    awg2.stop_all() ;awg2.flush_waveform()
    awg3.stop_all() ;awg3.flush_waveform()
    waveform_dict = append_waveforms(sequence, append_ports, waveforms_appended)
    if readout_port_tx in sequence.port_list:
        awg1.stop_all()
        awg1.flush_waveform()
        awg1.load_waveform(waveform_dict[readout_port_tx.name].real, 0, append_zeros=True)
        awg_readout_tx.queue_waveform(0, trigger="software/hvi", cycles=cycles)
    if readout_port_rx in sequence.port_list:
        awg1.load_waveform(waveform_dict[readout_port_rx.name].real, 1, append_zeros=True)
        awg_readout_rx.queue_waveform(1, trigger="software/hvi", cycles=cycles)
    if qubit_drive_port_tx in sequence.port_list:
        i, q = iq_corrector_q_tx.correct(waveform_dict[qubit_drive_port_tx.name])
        awg2.stop_all()
        awg2.flush_waveform()
        awg2.load_waveform(i, 1, append_zeros=True)
        awg2.load_waveform(q, 2, append_zeros=True)
        awg_Qdrive_I_tx.queue_waveform(1, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_Q_tx.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_I_tx.dc_offset(iq_corrector_q_tx.i_offset)
        awg_Qdrive_Q_tx.dc_offset(iq_corrector_q_tx.q_offset)
    if fogi_port_tx in sequence.port_list:
        i, q = iq_corrector_fogi_tx.correct(waveform_dict[fogi_port_tx.name])
        awg2.load_waveform(i, 3, append_zeros=True)
        awg2.load_waveform(q, 4, append_zeros=True)
        awg_fogi_I_tx.queue_waveform(3, trigger="software/hvi", cycles=cycles)
        awg_fogi_Q_tx.queue_waveform(4, trigger="software/hvi", cycles=cycles)
        awg_fogi_I_tx.dc_offset(iq_corrector_fogi_tx.i_offset)
        awg_fogi_Q_tx.dc_offset(iq_corrector_fogi_tx.q_offset)
    if qubit_drive_port_rx in sequence.port_list:
        i, q = iq_corrector_q_rx.correct(waveform_dict[qubit_drive_port_rx.name])
        awg3.stop_all()
        awg3.flush_waveform()
        awg3.load_waveform(i, 1, append_zeros=True)
        awg3.load_waveform(q, 2, append_zeros=True)
        awg_Qdrive_I_rx.queue_waveform(1, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_Q_rx.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_I_rx.dc_offset(iq_corrector_q_rx.i_offset)
        awg_Qdrive_Q_rx.dc_offset(iq_corrector_q_rx.q_offset)
    if fogi_port_rx in sequence.port_list:
        i, q = iq_corrector_fogi_rx.correct(waveform_dict[fogi_port_rx.name])
        awg3.load_waveform(i, 3, append_zeros=True)
        awg3.load_waveform(q, 4, append_zeros=True)
        awg_fogi_I_rx.queue_waveform(3, trigger="software/hvi", cycles=cycles)
        awg_fogi_Q_rx.queue_waveform(4, trigger="software/hvi", cycles=cycles)
        awg_fogi_I_rx.dc_offset(iq_corrector_fogi_rx.i_offset)
        awg_fogi_Q_rx.dc_offset(iq_corrector_fogi_rx.q_offset)
    if JPA_port_tx in sequence.port_list:
        i, q = iq_corrector_JPA_tx.correct(waveform_dict[JPA_port_tx.name])
        awg1.load_waveform(i, 2, append_zeros=True)
        awg1.load_waveform(q, 3, append_zeros=True)
        awg_JPA_I_tx.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_JPA_Q_tx.queue_waveform(3, trigger="software/hvi", cycles=cycles)
        awg_JPA_I_tx.dc_offset(iq_corrector_JPA_tx.i_offset)
        awg_JPA_Q_tx.dc_offset(iq_corrector_JPA_tx.q_offset)
    if JPA_port_rx in sequence.port_list:
        i, q = iq_corrector_JPA_rx.correct(waveform_dict[JPA_port_rx.name])
        awg4.load_waveform(i, 1, append_zeros=True)
        awg4.load_waveform(q, 2, append_zeros=True)
        awg_JPA_I_rx.queue_waveform(1, trigger="software/hvi", cycles=cycles)
        awg_JPA_Q_rx.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_JPA_I_rx.dc_offset(iq_corrector_JPA_rx.i_offset)
        awg_JPA_Q_rx.dc_offset(iq_corrector_JPA_rx.q_offset)
    if dig_port_tx in sequence.port_list:
        if len(dig_port_tx.measurement_windows) == 0:
            acquire_start = 0
        else:
            acquire_start=dig_port_tx.measurement_windows[0][0]
            acquire_end=dig_port_tx.measurement_windows[-1][-1]
            dig_ch_tx.points_per_cycle(int((acquire_end-acquire_start)/dig_ch_tx.sampling_interval()))
        assert dig_ch_tx.points_per_cycle() % dig_ch_tx.sampling_interval() == 0
        dig_ch_tx.delay(int(acquire_start//dig_ch_tx.sampling_interval()))
        dig_ch_tx.cycles(cycles)
    if dig_port_rx in sequence.port_list:
        if len(dig_port_rx.measurement_windows) == 0:
            acquire_start = 0
        else:
            acquire_start=dig_port_rx.measurement_windows[0][0]
            acquire_end=dig_port_rx.measurement_windows[-1][-1]
            dig_ch_rx.points_per_cycle(int((acquire_end-acquire_start)/dig_ch_rx.sampling_interval()))
        assert dig_ch_rx.points_per_cycle() % dig_ch_rx.sampling_interval() == 0
        dig_ch_rx.delay(int(acquire_start//dig_ch_rx.sampling_interval()))
        dig_ch_rx.cycles(cycles)



def append_two_waveforms(seq:Sequence, append_port0:Port, waveform0_appended:np.ndarray, append_port1:Port, waveform1_appended:np.ndarray):
    seq.compile()
    waveform_dict = dict()
    for port in seq.port_list:
        waveform = port.waveform
        windows = port.measurement_windows
        if port.name == append_port0.name:
            for window in windows:
                try:
                    if window[1] - window[0] == len(waveform0_appended):
                        waveform[int(window[0]) : int(window[1])] = waveform0_appended
                    else:
                        waveform[int(window[0]) : int(window[1])] = np.zeros(int(window[1] - window[0]))
                        print(f"zeros appended : {append_port0.name}, {window[0]}~{window[1]}")
                except:
                    print(f"not appended : {append_port0.name}, {window[0]}~{window[1]}")
                    pass
        if port.name == append_port1.name:
            for window in windows:
                try:
                    if window[1] - window[0] == len(waveform1_appended):
                        waveform[int(window[0]) : int(window[1])] = waveform1_appended
                    else:waveform[int(window[0]) : int(window[1])] = np.zeros(int(window[1] - window[0]))
                except:
                    print(f"not appended : {append_port1.name}, {window[0]}~{window[1]}")
                    pass        
        waveform_dict[port.name] = waveform
    return waveform_dict

