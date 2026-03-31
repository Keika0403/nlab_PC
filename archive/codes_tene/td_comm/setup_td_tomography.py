import numpy as np
from setup_td import *
from sequence_parser import Sequence, Variable, Variables
from plottr.data.datadict_storage import DataDict, DDH5Writer
from sequence_parser.instruction import Delay
from tqdm import tqdm
from scipy import signal as sg
from datataking import search_datadict_miyamura


setup_file_tomo = __file__
tags.append("tomography")

def append_waveform(seq:Sequence, append_port:Port, waveform_appended:np.ndarray):
    seq.compile()
    waveform_dict = dict()
    for port in seq.port_list:
        waveform = port.waveform
        windows = port.measurement_windows
        if port.name == append_port.name:
            for window in windows:
                try:
                    if window[1] - window[0] == len(waveform_appended):
                        waveform[int(window[0]) : int(window[1])] = waveform_appended
                    else:waveform[int(window[0]) : int(window[1])] = np.zeros(int(window[1] - window[0]))
                except:
                    print(f"not appended : {append_port.name}, {window[0]}~{window[1]}")
                    pass
        waveform_dict[port.name] = waveform
    return waveform_dict

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

def load_sequence_w_append(sequence:Sequence, append_ports:Seq[Port], waveforms_added:Seq[np.ndarray]|Seq[Seq[np.ndarray]], cycles, check:bool=False):
    sequence.compile()
    awg1.stop_all() ;awg1.flush_waveform()
    awg2.stop_all() ;awg2.flush_waveform()
    awg3.stop_all() ;awg3.flush_waveform()
    # awg4.stop_all() ;awg4./ush_waveform()
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


def load_sequence_w_two_append(sequence:Sequence, append_ports, waveforms_appended, cycles):
    waveform_dict = append_two_waveforms(sequence, append_ports[0], waveforms_appended[0], append_ports[1], waveforms_appended[1])
    if readout_port in sequence.port_list:
        awg1.stop_all()
        awg1.flush_waveform()
        awg1.load_waveform(waveform_dict[readout_port.name].real, 0, append_zeros=True)
        awg_readout_I.queue_waveform(0, trigger="software/hvi", cycles=cycles)
    if qubit_drive_port in sequence.port_list:
        i, q = iq_corrector_q.correct(waveform_dict[qubit_drive_port.name])
        awg2.stop_all()
        awg2.flush_waveform()
        awg2.load_waveform(i, 1, append_zeros=True)
        awg2.load_waveform(q, 2, append_zeros=True)
        awg_Qdrive_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_I.dc_offset(iq_corrector_q.i_offset)
        awg_Qdrive_Q.dc_offset(iq_corrector_q.q_offset)
    if fogi_port in sequence.port_list:
        i, q = iq_corrector_fogi.correct(waveform_dict[fogi_port.name])
        # i, q = iq_corrector_fogi.correct(chirping(waveform_dict[fogi_port.name]))
        awg2.load_waveform(i, 3, append_zeros=True)
        awg2.load_waveform(q, 4, append_zeros=True)
        awg_fogi_I.queue_waveform(3, trigger="software/hvi", cycles=cycles)
        awg_fogi_Q.queue_waveform(4, trigger="software/hvi", cycles=cycles)
        awg_fogi_I.dc_offset(iq_corrector_fogi.i_offset)
        awg_fogi_Q.dc_offset(iq_corrector_fogi.q_offset)
    if JPA_port in sequence.port_list:
        i, q = iq_corrector_JPA.correct(waveform_dict[JPA_port.name])
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

def load_sequence_w_append_comm(sequence:Sequence, append_ports, waveforms_appended, cycles):
    waveform_dict = append_two_waveforms(sequence, append_ports[0], waveforms_appended[0], append_ports[1], waveforms_appended[1])
    if readout_port_tx in sequence.port_list:
        awg1.stop_all()
        awg1.flush_waveform()
        awg1.load_waveform(waveform_dict[readout_port_tx.name].real, 0, append_zeros=True)
        awg_readout_tx.queue_waveform(0, trigger="software/hvi", cycles=cycles)
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
        # i, q = iq_corrector_fogi.correct(chirping(waveform_dict[fogi_port.name]))
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
        awg_Qdrive_Q_tx.dc_offset(iq_corrector_q_rx.q_offset)
    if fogi_port_rx in sequence.port_list:
        i, q = iq_corrector_fogi_rx.correct(waveform_dict[fogi_port_rx.name])
        # i, q = iq_corrector_fogi.correct(chirping(waveform_dict[fogi_port.name]))
        awg3.load_waveform(i, 3, append_zeros=True)
        awg3.load_waveform(q, 4, append_zeros=True)
        awg_fogi_I_rx.queue_waveform(3, trigger="software/hvi", cycles=cycles)
        awg_fogi_Q_rx.queue_waveform(4, trigger="software/hvi", cycles=cycles)
        awg_fogi_I_rx.dc_offset(iq_corrector_fogi_rx.i_offset)
        awg_fogi_Q_rx.dc_offset(iq_corrector_fogi_rx.q_offset)
    if JPA_port_tx in sequence.port_list:
        i, q = iq_corrector_JPA_tx.correct(waveform_dict[JPA_port_tx.name])
        awg1.load_waveform(i, 1, append_zeros=True)
        awg1.load_waveform(q, 2, append_zeros=True)
        awg_JPA_I_tx.queue_waveform(1, trigger="software/hvi", cycles=cycles)
        awg_JPA_Q_tx.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_JPA_I_tx.dc_offset(iq_corrector_JPA_tx.i_offset)
        awg_JPA_Q_tx.dc_offset(iq_corrector_JPA_tx.q_offset)
    if dig_port_tx in sequence.port_list:
        acquire_start=dig_port_tx.measurement_windows[0][0]
        acquire_end=dig_port_tx.measurement_windows[-1][-1]
        dig_ch_tx.delay(int(acquire_start//dig_ch_tx.sampling_interval()))
        dig_ch_tx.points_per_cycle(int((acquire_end-acquire_start)/dig_ch_tx.sampling_interval()))
        assert dig_ch_tx.points_per_cycle() % dig_ch_tx.sampling_interval() == 0
        dig_ch_tx.cycles(cycles)

def load_sequence_comm(sequence: Sequence, cycles: int, chirp=False):
    sequence.compile()
    if readout_port in sequence.port_list:
        awg1.stop_all()
        awg1.flush_waveform()
        awg1.load_waveform(readout_port.waveform.real, 0, append_zeros=True)
        awg_readout_I.queue_waveform(0, trigger="software/hvi", cycles=cycles)
    if qubit_drive_port_tx in sequence.port_list:
        i, q = iq_corrector_q_tx.correct(qubit_drive_port_tx.waveform)
        awg2.stop_all()
        awg2.flush_waveform()
        awg2.load_waveform(i, 1, append_zeros=True)
        awg2.load_waveform(q, 2, append_zeros=True)
        awg_Qdrive_I_tx.queue_waveform(1, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_Q_tx.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_I_tx.dc_offset(iq_corrector_q_tx.i_offset)
        awg_Qdrive_Q_tx.dc_offset(iq_corrector_q_tx.q_offset)
    if fogi_port_tx in sequence.port_list:
        i, q = iq_corrector_fogi_tx.correct(fogi_port_tx.waveform)
        # i, q = iq_corrector_fogi.correct(chirping(waveform_dict[fogi_port.name]))
        awg2.flush_waveform()
        awg2.load_waveform(i, 3, append_zeros=True)
        awg2.load_waveform(q, 4, append_zeros=True)
        awg_fogi_I_tx.queue_waveform(3, trigger="software/hvi", cycles=cycles)
        awg_fogi_Q_tx.queue_waveform(4, trigger="software/hvi", cycles=cycles)
        awg_fogi_I_tx.dc_offset(iq_corrector_fogi_tx.i_offset)
        awg_fogi_Q_tx.dc_offset(iq_corrector_fogi_tx.q_offset)
    if qubit_drive_port_rx in sequence.port_list:
        i, q = iq_corrector_q_rx.correct(qubit_drive_port_rx.waveform)
        awg3.stop_all()
        awg3.flush_waveform()
        awg3.load_waveform(i, 1, append_zeros=True)
        awg3.load_waveform(q, 2, append_zeros=True)
        awg_Qdrive_I_rx.queue_waveform(1, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_Q_rx.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_Qdrive_I_rx.dc_offset(iq_corrector_q_rx.i_offset)
        awg_Qdrive_Q_tx.dc_offset(iq_corrector_q_rx.q_offset)
    if fogi_port_rx in sequence.port_list:
        i, q = iq_corrector_fogi_rx.correct(fogi_port_rx.waveform)
        # i, q = iq_corrector_fogi.correct(chirping(waveform_dict[fogi_port.name]))
        awg3.flush_waveform()
        awg3.load_waveform(i, 3, append_zeros=True)
        awg3.load_waveform(q, 4, append_zeros=True)
        awg_fogi_I_rx.queue_waveform(3, trigger="software/hvi", cycles=cycles)
        awg_fogi_Q_rx.queue_waveform(4, trigger="software/hvi", cycles=cycles)
        awg_fogi_I_rx.dc_offset(iq_corrector_fogi_rx.i_offset)
        awg_fogi_Q_rx.dc_offset(iq_corrector_fogi_rx.q_offset)
    if JPA_port in sequence.port_list:
        i, q = iq_corrector_JPA.correct(JPA_port.waveform)
        awg1.flush_waveform()
        awg1.load_waveform(i, 1, append_zeros=True)
        awg1.load_waveform(q, 2, append_zeros=True)
        awg_JPA_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
        awg_JPA_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        awg_JPA_I.dc_offset(iq_corrector_JPA.i_offset)
        awg_JPA_Q.dc_offset(iq_corrector_JPA.q_offset)
    # if dig_port in sequence.port_list:
    #     acquire_start=dig_port.measurement_windows[0][0]
    #     acquire_end=dig_port.measurement_windows[-1][-1]
    #     dig_ch.delay(int(acquire_start//dig_ch.sampling_interval()))
    #     dig_ch.points_per_cycle(int((acquire_end-acquire_start)/dig_ch.sampling_interval()))
    #     assert dig_ch.points_per_cycle() % dig_ch.sampling_interval() == 0
    dig_ch_comm.cycles(cycles)


def run_comm(sequence:Sequence, plot=False, lo_on=True):
    hvi_trigger.output(False)
    assert dig_ch_comm.trigger_mode()=="software/hvi"
    if dig_port in sequence.port_list:
        acquire_start=dig_port.measurement_windows[0][0]
        acquire_end=dig_port.measurement_windows[-1][-1]
        assert (acquire_end - acquire_start) % 4 == 0 
        dig_ch_comm.delay(int(acquire_start//dig_ch_comm.sampling_interval()))
        dig_ch_comm.points_per_cycle(int((acquire_end-acquire_start)/dig_ch_comm.sampling_interval()))
    lo1.output(True)
    if readout_port in sequence.port_list:
        assert dig_ch_comm.cycles()==awg_readout_I.cycles()[0], f'{dig_ch_comm.cycles()}, {awg_readout_I.cycles()}'
        awg_readout_I.start()
    if qubit_drive_port_tx in sequence.port_list:
        assert dig_ch_comm.cycles()==awg_Qdrive_I_tx.cycles()[0], f'{dig_ch_comm.cycles()}, {awg_Qdrive_I_tx.cycles()}'
        if lo_on:lo2.output(True)
        awg_Qdrive_I_tx.start()
        awg_Qdrive_Q_tx.start()
    if qubit_drive_port_rx in sequence.port_list:
        assert dig_ch_comm.cycles()==awg_Qdrive_I_rx.cycles()[0], f'{dig_ch_comm.cycles()}, {awg_Qdrive_I_rx.cycles()}'
        if lo_on:lo4.output(True)
        awg_Qdrive_I_rx.start()
        awg_Qdrive_Q_rx.start()
    if fogi_port_tx in sequence.port_list:
        dig_ch_comm.cycles()==awg_fogi_I_tx.cycles()[0], f'{dig_ch_comm.cycles()}, {awg_fogi_I_tx.cycles()}'
        if lo_on:lo3.output(True)
        awg_fogi_I_tx.start()
        awg_fogi_Q_tx.start()
    if fogi_port_rx in sequence.port_list:
        dig_ch_comm.cycles()==awg_fogi_I_rx.cycles()[0], f'{dig_ch_comm.cycles()}, {awg_fogi_I_rx.cycles()}'
        if lo_on:lo5.output(True)
        awg_fogi_I_rx.start()
        awg_fogi_Q_rx.start()
    if JPA_port in sequence.port_list:
        assert dig_ch_comm.cycles()==awg_JPA_I.cycles()[0], f'{dig_ch_comm.cycles()}, {awg_JPA_I.cycles()}'
        awg_JPA_I.start()
        awg_JPA_Q.start()
    dig_ch_comm.start()
    hvi_trigger.output(True)
    data = dig_ch_comm.read()
    hvi_trigger.output(False)
    if plot:
        plt.plot(dig_ch_comm.sampling_interval() * np.arange(len(data[0])), data.mean(axis=0) * voltage_step_comm)
        plt.show()
    return data

def demodulate_comm(data, demodulation_if = readout_if_freq_tx):
    t = np.arange(data.shape[-1]) * dig_ch_tx.sampling_interval()
    return (data * np.exp(2j * np.pi * demodulation_if * t)).mean(axis=-1)

def off_comm():
    awg1.stop_all()
    awg2.stop_all()
    awg3.stop_all()
    hvi_trigger.output(False)
    dig_ch_comm.stop()
    lo1.output(False)
    lo2.output(False)
    lo3.output(False)
    lo4.output(False)
    lo5.output(False)

def postselection_both(data, start_time, stop_time):
    start_idx = start_time // dig_ch.sampling_interval()
    end_idx = stop_time // dig_ch.sampling_interval()
    delete_idx = []
    for i in range(len(data)):
        single_shot = data[i, start_idx:end_idx]
        if np.dot(single_shot - mean_g, ein_vec) > 1/2:
            delete_idx.append(i)
    g_data = np.delete(data, delete_idx, axis=0)
    delete_idx = []
    for i in range(len(data)):
        single_shot = data[i, start_idx:end_idx]
        if np.dot(single_shot - mean_g, ein_vec) < 1/2:
            delete_idx.append(i)
    e_data = np.delete(data, delete_idx, axis=0)
    return g_data, e_data