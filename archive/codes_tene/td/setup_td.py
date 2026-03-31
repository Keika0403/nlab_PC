import os
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
from qcodes.instrument_drivers.yokogawa.GS200 import GS200
from sequence_parser import Sequence
from sequence_parser.variable import Variable, Variables
from sequence_parser.instruction import *
from setup_td_parameters import *
from qcodes_drivers.iq_corrector import IQCorrector
from tqdm import tqdm

setup_file = __file__

tags = ["TD", "CDY170", "Out2B"]
data_path = f"D:\\K_Sunada\\result\\{tags[1]}"
# data_path_IQ = f"D:\\K_Sunada\\result\\CDY157"
os.makedirs(data_path, exist_ok=True)

# Instruments
station = qc.Station()

awg1:M3202A = M3202A('awg1', chassis=1, slot=2)
awg_readout_I:SD_AWG_CHANNEL = awg1.ch2
awg_JPA_I:SD_AWG_CHANNEL = awg1.ch3
awg_JPA_Q:SD_AWG_CHANNEL = awg1.ch4
station.add_component(awg1)

awg2:M3202A = M3202A('awg2', 1, 4)
awg_Qdrive_I:SD_AWG_CHANNEL = awg2.ch1
awg_Qdrive_Q:SD_AWG_CHANNEL = awg2.ch2
awg_fogi_I:SD_AWG_CHANNEL = awg2.ch3
awg_fogi_Q:SD_AWG_CHANNEL = awg2.ch4
station.add_component(awg2)

hvi_trigger:HVI_Trigger = HVI_Trigger('hvi_trigger', "PXI0::1::BACKPLANE")
station.add_component(hvi_trigger)
hvi_trigger.digitizer_delay(400)
hvi_trigger.trigger_period(100000) # ns

dig:M3102A = M3102A('digitizer', chassis=1, slot=7)
station.add_component(dig)
dig_ch:SD_DIG_CHANNEL = dig.ch2
dig_ch.high_impedance(False)  # 50 Ohms
dig_ch.half_range_50(0.5)  # V_pp / 2
dig_ch.ac_coupling(False)  # dc coupling
dig_ch.sampling_interval(2)  # ns
dig_ch.trigger_mode("software/hvi")
dig_ch.timeout(10000)  # ms                                                           
voltage_step=dig_ch.half_range_50()/(2**15-1)
                   
lo1:E8247 = E8247('lo1', 'TCPIP0::192.168.100.5::inst0::INSTR')
lo1.power(24)
lo1.frequency(readout_lo_freq*1e9)
station.add_component(lo1)

lo2:N51x1 = N51x1('lo2', 'TCPIP0::192.168.100.129::inst0::INSTR')
lo2.frequency(qubit_lo_freq*1e9)
lo2.power(12) #17
station.add_component(lo2)

lo3:N51x1 = N51x1('lo3', 'TCPIP0::192.168.100.8::inst0::INSTR')
lo3.frequency(fogi_lo_freq*1e9) 
lo3.power(16)
station.add_component(lo3)

# from qcodes.instrument_drivers.yokogawa.GS200 import GS200
# yoko:GS200 = GS200("yoko", "TCPIP0::192.168.100.98::inst0::INSTR")
# station.add_component(yoko)
# if yoko.state():
#     print('yoko was on')
#     pass
# else:
#     yoko.ramp_current(0, step=1e-8, delay=0)
#     yoko.on()
#     print('yoko was off')
# assert yoko.state()
# yoko.ramp_current(88e-6, step=1e-8, delay=0) #112

# IQ calibration
iq_corrector_fogi = IQCorrector(
    awg_fogi_I,
    awg_fogi_Q,
    data_path,
    lo_leakage_datetime="2024-11-13T094603",
    rf_power_datetime="2024-11-13T095452",
    len_kernel=41,
    fit_weight=10,
)

iq_corrector_q = IQCorrector(
    awg_Qdrive_I,
    awg_Qdrive_Q,
    data_path,
    lo_leakage_datetime="2024-11-13T093214",
    rf_power_datetime="2024-11-13T094023",
    len_kernel=41,
    fit_weight=10,
)



iq_corrector_JPA = IQCorrector(
    awg_JPA_I,
    awg_JPA_Q,
    data_path,
    lo_leakage_datetime="2024-11-13T091918",
    rf_power_datetime="2024-11-13T092701",
    len_kernel=41,
    fit_weight=10,
)

def fogi_ac_Stark_shift(amp):
    return 0#-0.09728155* amp**2

def chirping(fogi_waveform):
    wave_amp = np.abs(fogi_waveform)
    shift = fogi_ac_Stark_shift(wave_amp)
    phi = [0]
    for i in range(1, len(shift)):
        new_phi = phi[i-1] + shift[i]
        phi.append(new_phi)
    phi = np.array(phi)
    chirped_pulse = fogi_waveform * np.exp(1j*phi * 2*np.pi)
    return chirped_pulse


# functions
def load_sequence(sequence: Sequence, cycles: int, chirp=False):
    sequence.compile()
    if readout_port in sequence.port_list:
        awg1.stop_all()
        awg1.flush_waveform()
        awg1.load_waveform(readout_port.waveform.real, 0, append_zeros=True)
        awg_readout_I.queue_waveform(0, trigger="software/hvi", cycles=cycles)
    if qubit_drive_port in sequence.port_list:
        i, q = iq_corrector_q.correct(qubit_drive_port.waveform)
        awg2.stop_all()
        awg2.flush_waveform()
        awg2.load_waveform(i, 1, append_zeros=True)
        awg2.load_waveform(q, 2, append_zeros=True)
        awg_Qdrive_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_I.dc_offset(iq_corrector_q.i_offset)
        awg_Qdrive_Q.dc_offset(iq_corrector_q.q_offset)
    if fogi_port in sequence.port_list:
        if chirp:
            i, q = iq_corrector_fogi.correct(chirping(fogi_port.waveform))
        else:
            i, q = iq_corrector_fogi.correct(fogi_port.waveform)
        awg2.load_waveform(i, 3, append_zeros=True)
        awg2.load_waveform(q, 4, append_zeros=True)
        awg_fogi_I.queue_waveform(3, trigger="software/hvi", cycles=cycles)
        awg_fogi_Q.queue_waveform(4, trigger="software/hvi", cycles=cycles)
        awg_fogi_I.dc_offset(iq_corrector_fogi.i_offset)
        awg_fogi_Q.dc_offset(iq_corrector_fogi.q_offset)
    if JPA_port in sequence.port_list:
        i, q = iq_corrector_JPA.correct(JPA_port.waveform)
        awg1.load_waveform(i, 1, append_zeros=True)
        awg1.load_waveform(q, 2, append_zeros=True)
        awg_JPA_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
        awg_JPA_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_JPA_I.dc_offset(iq_corrector_JPA.i_offset)
        awg_JPA_Q.dc_offset(iq_corrector_JPA.q_offset)
    if dig_port in sequence.port_list:
        acquire_start=dig_port.measurement_windows[0][0]
        acquire_end=dig_port.measurement_windows[-1][-1]
        dig_ch.delay(int(acquire_start//dig_ch.sampling_interval()))
        dig_ch.points_per_cycle(int((acquire_end-acquire_start)/dig_ch.sampling_interval()))
        assert dig_ch.points_per_cycle() % dig_ch.sampling_interval() == 0
    dig_ch.cycles(cycles)

def run(sequence:Sequence, plot=False, lo_on=True):
    hvi_trigger.output(False)
    assert dig_ch.trigger_mode()=="software/hvi"
    if dig_port in sequence.port_list:
        acquire_start=dig_port.measurement_windows[0][0]
        acquire_end=dig_port.measurement_windows[-1][-1]
        assert (acquire_end - acquire_start) % 4 == 0 
        dig_ch.delay(int(acquire_start//dig_ch.sampling_interval()))
        dig_ch.points_per_cycle(int((acquire_end-acquire_start)/dig_ch.sampling_interval()))
    lo1.output(True)
    if readout_port in sequence.port_list:
        assert dig_ch.cycles()==awg_readout_I.cycles()[0], f'{dig_ch.cycles()}, {awg_readout_I.cycles()}'
        awg_readout_I.start()
    if qubit_drive_port in sequence.port_list:
        assert dig_ch.cycles()==awg_Qdrive_I.cycles()[0], f'{dig_ch.cycles()}, {awg_Qdrive_I.cycles()}'
        if lo_on:lo2.output(True)
        awg_Qdrive_I.start()
        awg_Qdrive_Q.start()
        if fogi_port in sequence.port_list:
            dig_ch.cycles()==awg_fogi_I.cycles()[0], f'{dig_ch.cycles()}, {awg_fogi_I.cycles()}'
            if lo_on:lo3.output(True)
            awg_fogi_I.start()
            awg_fogi_Q.start()
    if JPA_port in sequence.port_list:
        assert dig_ch.cycles()==awg_JPA_I.cycles()[0], f'{dig_ch.cycles()}, {awg_JPA_I.cycles()}'
        awg_JPA_I.start()
        awg_JPA_Q.start()
    dig_ch.start()
    hvi_trigger.output(True)
    data = dig_ch.read()
    hvi_trigger.output(False)
    if plot:
        plt.plot(dig_ch.sampling_interval() * np.arange(len(data[0])), data.mean(axis=0) * voltage_step)
        plt.show()
    return data

def demodulate(data, demodulation_if = readout_if_freq):
    t = np.arange(data.shape[-1]) * dig_ch.sampling_interval()
    return (data * np.exp(2j * np.pi * demodulation_if * t)).mean(axis=-1)

def off():
    awg1.stop_all()
    awg2.stop_all()
    hvi_trigger.output(False)
    dig_ch.stop()
    lo1.output(False)
    lo2.output(False)
    lo3.output(False)

def load_note(data_path, date, name):
    lines = []
    with open(f'{data_path}\\{date}\\{name}', encoding='utf-8') as f:
        lines = f.readlines()  
    return"".join(lines)

def error_print(e):
    print('type:' + str(type(e)))
    print('args:' + str(e.args))
    print('message:' + e.message)
    print('error:' + str(e))