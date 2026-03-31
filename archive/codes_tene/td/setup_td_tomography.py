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

def load_sequence_w_append(sequence:Sequence, append_port:Port, waveform_appended:np.ndarray, cycles):
    waveform_dict = append_waveform(sequence, append_port, waveform_appended)
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
    dig_ch.cycles(cycles)


def load_sequence_w_two_append(sequence:Sequence, append_ports, waveforms_appended, cycles):
    waveform_dict = append_two_waveforms(sequence, append_ports[0], waveforms_appended[0], append_ports[1], waveforms_appended[1])
    # print(waveform_dict.keys())
    # plt.plot(waveform_dict[readout_port.name])
    # plt.show()
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