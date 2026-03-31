import os

import numpy as np
from setup_td import *
from sequence_parser import Sequence, Variable, Variables
from plottr.data.datadict_storage import DataDict, DDH5Writer, datadict_from_hdf5
from tqdm import tqdm
from setup_td_tomography import *
from time_reverse import *

measurement_name = os.path.basename(__file__)[:-3]

trigger_period = 3000
hvi_trigger.trigger_period(trigger_period) 

##### fogi pulse
pulse_dict = dict(
    data0 = dict(target_freq=10.28, date="2024-05-22", acquire_time="205328"),
    # data1 = dict(target_freq=10.30, date="2024-05-22", acquire_time="205329"),
    # data2 = dict(target_freq=10.32, date="2024-05-22", acquire_time="205330"),
    # data3 = dict(target_freq=10.32, date="2024-02-22", acquire_time="142727"),
    # data4 = dict(target_freq=10.33, date="2024-02-22", acquire_time="142728"),
)
# for k, v in tqdm(pulse_dict.items()):
for k, v in tqdm(pulse_dict.items()):
    _, datadict = search_datadict_miyamura("D:\\K_Sunada\\result\\control_pulses", 
                                            v["date"], acquire_time=v["acquire_time"])
    target_shape_note = load_note("D:\\K_Sunada\\result\\control_pulses", v["date"],
                                f"{_}\\target_shape.md").replace("path", "before") + f"\npath : {_}"
    control_pulse_f = datadict["control_pulse"]["values"].ravel()#[::-1]


fogi_delay = 58
fogi_duration = 1000


##### coherent pusle
ph_if = readout_lo_freq - np.array([10.28e9]) 
photon_amplitude = [0.01] #id=0
const = 2.5e-3 * 2 * np.pi
id = 0


port_list = [fogi_port, readout_port]
acquisition_time = fogi_duration + 80
num_of_cycles = 50000
repetition = 20


data = DataDict(
    time=dict(unit="ns"),
    photon_amp = dict(unit = ""),
    waveform_I=dict(axes=["time", "photon_amp"]),
    waveform_zero_fogi_I=dict(axes=["time", "photon_amp"]),

    waveform_Q=dict(axes=["time",  "photon_amp"]),
    waveform_zero_fogi_Q=dict(axes=["time", "photon_amp"]),

    waveform=dict(axes=["time", "photon_amp"]),
    waveform_zero_fogi=dict(axes=["time",  "photon_amp"]),

    qstate = dict(axes=["photon_amp"]), 
    qstate_zero_fogi = dict(axes=["photon_amp"]),
)
data.validate()

lo1.output(True)
lo2.output(True)
lo3.output(True)

with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())

    for p in tqdm(photon_amplitude):
        ph_amp = p
        x = np.linspace(0, 999, 1000)
        ph_waveform = 1/np.cosh(const * (x - fogi_duration/2))*np.cos(2*np.pi* ph_if[id]*x*1e-9) 
        control_pulse_p=ph_waveform *ph_amp
        # print(control_pulse_p)
        control_pulses = [control_pulse_f, control_pulse_f]
        # plt.plot(x, control_pulse_p)
        # plt.show()

        def abs_sequence(fogi:bool, photon:bool, JPA_direction=1) -> Sequence:
            assert JPA_direction == -1 or 1
            seq = Sequence(port_list=ports)
            seq.call(reset_sequence)
            seq.trigger(ports)
            seq.add(Square(amplitude=JPA_direction*0.7, duration=len(control_pulse_f)+100), JPA_port)
            seq.add(Acquire(acquisition_time), dig_port)
            seq.add(Delay(20), readout_port)
            seq.trigger([readout_port, qubit_drive_port])
            seq.add(Delay(fogi_delay), fogi_port)
            if fogi:
                seq.add(Acquire(len(control_pulse_f)), fogi_port)
            if photon:
                seq.add(Acquire(len(control_pulse_p)), readout_port) 
            seq.trigger([qubit_drive_port,readout_port])
            seq.add(Delay(60), readout_port)
            seq.add(SetDetuning(0), readout_port)
            seq.call(readout_seq)
            return seq

        def seq_Q(fogi_amp, photon_amp=True):
            seq = Sequence(port_list = [qubit_drive_port, fogi_port, readout_port, dig_port, JPA_port])
            seq.call(reset_sequence)
            seq.trigger(ports)
            seq.call(seq_JPA_pi)
            seq.add(Acquire(acquisition_time), dig_port)
            seq.add(Delay(20), readout_port)
            seq.trigger([readout_port, qubit_drive_port])
            seq.add(Delay(fogi_delay), fogi_port)
            if fogi_amp:
                seq.add(Acquire(len(control_pulse_f)), fogi_port)
            if photon_amp:
                seq.add(Acquire(len(control_pulse_p)), readout_port)
            # seq.add(SetDetuning(-(photon_freq - readout_lo_freq ) - readout_if_freq), readout_port)
            # seq.add(Square(amplitude = photon_amp, duration=photon_duration), readout_port)
            seq.trigger([qubit_drive_port,  readout_port])
            seq.add(Delay(60), readout_port)
            seq.add(SetDetuning(0), readout_port)
            seq.call(readout_seq)
            return seq
        # seq_Q(0.5, 0.01).draw()
        # raise SystemError

        waveform_I = []
        waveform_ref_I = []
        waveform_offset_I = []
        waveform_offset_ref_I = []
        qstate= []
        qstate_ref = []

        waveform_Q = []
        waveform_ref_Q = []
        waveform_offset_Q = []
        waveform_offset_ref_Q = []

        waveform = []
        waveform_ref = []
        for _ in tqdm(range(repetition)):
            # load_sequence(seq_I(fogi_amplitude, ph_amp), cycles=num ,  chirp=True)           
            load_sequence_w_two_append(seq_I(fogi_amp=True, photon_amp=True), append_ports=port_list, waveforms_appended=control_pulses, cycles=num_of_cycles)
            data = run(seq_I(fogi_amp=True, photon_amp=True)).mean(axis=0)
            plt.plot(data)
            
            waveform_I = np.append(waveform_I, data[0: acquisition_time//dig_ch.sampling_interval()]* voltage_step)
            qstate = np.append(qstate, demodulate(data[acquisition_time//dig_ch.sampling_interval():len(data)]))

            # load_sequence(seq_I(0, ph_amp), cycles=num_of_cycles,  chirp=True)
            load_sequence_w_two_append(seq_I(fogi_amp=False, photon_amp=True), append_ports=port_list, waveforms_appended=control_pulses, cycles=num_of_cycles)
            data1 = run(seq_I(fogi_amp=False, photon_amp=True)).mean(axis=0)
            plt.plot(data1)
            waveform_zero_fogi_I = np.append(waveform_zero_fogi_I, data1[0: acquisition_time//dig_ch.sampling_interval()]* voltage_step)
            qstate_zero_fogi = np.append(qstate_zero_fogi, demodulate(data1[acquisition_time//dig_ch.sampling_interval():len(data)]))
            
            # load_sequence(seq_I(0, 0), cycles=num_of_cycles,  chirp=True)
            load_sequence_w_two_append(seq_I(fogi_amp=False, photon_amp=False), append_ports=port_list, waveforms_appended=control_pulses, cycles=num_of_cycles)
            data2 = run(seq_I(fogi_amp=False, photon_amp=False)).mean(axis=0)* voltage_step
            plt.plot(data2)
            waveform_offset_I = np.append(waveform_offset_I, data2[0: acquisition_time//dig_ch.sampling_interval()])

            # load_sequence(seq_I(fogi, 0), cycles=num_of_cycles,  chirp=True)
            load_sequence_w_two_append(seq_I(fogi_amp=True, photon_amp=False), append_ports=port_list, waveforms_appended=control_pulses, cycles=num_of_cycles)
            data2_f = run(seq_I(fogi_amp=True, photon_amp=False)).mean(axis=0)* voltage_step
            plt.plot(data2_f)
            # plt.show()
            waveform_offset_I_f = np.append(waveform_offset_I_f, data2_f[0: acquisition_time//dig_ch.sampling_interval()])

            # load_sequence(seq_Q(fogi_amplitude, ph_amp), cycles=num_of_cycles,  chirp=True)
            load_sequence_w_two_append(seq_Q(fogi_amp=True, photon_amp=True), append_ports=port_list, waveforms_appended=control_pulses, cycles=num_of_cycles)
            dataQ = run(seq_Q(fogi_amp=True, photon_amp=True)).mean(axis=0)* voltage_step
            waveform_Q = np.append(waveform_Q, dataQ[0: acquisition_time//dig_ch.sampling_interval()])
            # qstate_Q = np.append(qstate_Q, dataQ[acquisition_time//dig_ch.sampling_interval()-1:len(data)-1].mean(axis=1))

            # load_sequence(seq_Q(0, ph_amp), cycles=num_of_cycles,  chirp=True)
            load_sequence_w_two_append(seq_Q(fogi_amp=False, photon_amp=True), append_ports=port_list, waveforms_appended=control_pulses, cycles=num_of_cycles)
            data1Q = run(seq_Q(fogi_amp=False, photon_amp=True)).mean(axis=0)* voltage_step
            waveform_zero_fogi_Q = np.append(waveform_zero_fogi_Q, data1Q[0: acquisition_time//dig_ch.sampling_interval()])
            # qstate_zero_fogi_Q = np.append(qstate_zero_fogi_Q, data1Q[acquisition_time//dig_ch.sampling_interval()-1:len(data)-1].mean(axis=1))
            
            # load_sequence(seq_Q(0, 0), cycles=num_of_cycles,  chirp=True)
            load_sequence_w_two_append(seq_Q(fogi_amp=False, photon_amp=False), append_ports=port_list, waveforms_appended=control_pulses, cycles=num_of_cycles)
            data2Q = run(seq_Q(fogi_amp=False, photon_amp=False)).mean(axis=0)* voltage_step
            waveform_offset_Q = np.append(waveform_offset_Q, data2Q[0: acquisition_time//dig_ch.sampling_interval()])

            # load_sequence(seq_Q(0, 0), cycles=num_of_cycles,  chirp=True)
            load_sequence_w_two_append(seq_Q(fogi_amp=True, photon_amp=False), append_ports=port_list, waveforms_appended=control_pulses, cycles=num_of_cycles)
            data2Q_f = run(seq_Q(fogi_amp=True, photon_amp=False)).mean(axis=0)* voltage_step
            waveform_offset_Q_f = np.append(waveform_offset_Q_f, data2Q_f[0: acquisition_time//dig_ch.sampling_interval()])

        waveform_I = waveform_I.reshape(int(repetition), acquisition_time//dig_ch.sampling_interval())
        waveform_zero_fogi_I = waveform_zero_fogi_I.reshape(int(repetition), acquisition_time//dig_ch.sampling_interval())
        waveform_offset_I = waveform_offset_I.reshape(int(repetition), acquisition_time//dig_ch.sampling_interval())
        waveform_offset_I_f = waveform_offset_I_f.reshape(int(repetition), acquisition_time//dig_ch.sampling_interval())
        waveform_Q = waveform_Q.reshape(int(repetition), acquisition_time//dig_ch.sampling_interval())
        waveform_zero_fogi_Q = waveform_zero_fogi_Q.reshape(int(repetition), acquisition_time//dig_ch.sampling_interval())
        waveform_offset_Q = waveform_offset_Q.reshape(int(repetition), acquisition_time//dig_ch.sampling_interval())
        waveform_offset_Q_f = waveform_offset_Q_f.reshape(int(repetition), acquisition_time//dig_ch.sampling_interval())
        
        waveform_I = np.array(waveform_I).mean(axis=0)
        waveform_zero_fogi_I = np.array(waveform_zero_fogi_I).mean(axis=0)
        waveform_offset_I = np.array(waveform_offset_I).mean(axis=0)
        waveform_offset_I_f = np.array(waveform_offset_I_f).mean(axis=0)
        waveform_Q = np.array(waveform_Q).mean(axis=0)
        waveform_zero_fogi_Q = np.array(waveform_zero_fogi_Q).mean(axis=0)
        waveform_offset_Q = np.array(waveform_offset_Q).mean(axis=0)
        waveform_offset_Q_f = np.array(waveform_offset_Q_f).mean(axis=0)
        qstate = np.array(qstate).mean(axis=0)
        qstate_zero_fogi =np.array(qstate_zero_fogi).mean(axis=0)
    #   print(waveform.shape)    
        # raise SystemError
        writer.add_data(
            time = dig_ch.sampling_interval()*np.arange(len(waveform_I)),
            photon_amp = ph_amp,
            waveform_I = waveform_I - waveform_offset_I_f,
            waveform_zero_fogi_I = waveform_zero_fogi_I - waveform_offset_I,
            waveform_Q = waveform_Q - waveform_offset_Q_f,
            waveform_zero_fogi_Q = waveform_zero_fogi_Q - waveform_offset_Q,
            waveform = (waveform_I - waveform_offset_I_f + waveform_Q- waveform_offset_Q_f)/2,
            waveform_zero_fogi = (waveform_zero_fogi_I- waveform_offset_I + waveform_zero_fogi_Q- waveform_offset_Q)/2,
            qstate = qstate,
            qstate_zero_fogi=qstate_zero_fogi,
        )
    awg2.flush_waveform()
    awg1.flush_waveform()
    dig_ch.stop()