import os

import numpy as np
from setup_td import *
from sequence_parser import Sequence, Variable, Variables
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm
from setup_td_tomography import *
from time_reverse import *

measurement_name = os.path.basename(__file__)[:-3]


trigger_period = 3000
hvi_trigger.trigger_period(trigger_period) 


fogi_freqs = np.linspace(5.3835, 5.386, 1)  #5.391
fogi_amplitude = 0.6
fogi_duration = 500
JPA_pump.params["duration"] = fogi_duration + 80
JPA_pump_pi.params["duration"] = fogi_duration + 80
fogi_delay = 36


# photon_amplitude =(1 + 0.5 *np.linspace(0, 8, 9))
photon_amplitude = [0.1175, 0.13, 0.142, 0.155, 0.167, 0.179, 0.1915,0.2035,  0.215, 0.227, 0.239]#[0.179, 0.1915,0.2035]#[0.215, 0.227, 0.239]#[0.251, 0.263, 0.275, 0.286, 0.298, 0.310, 0.322, 0.334, 0.346, 0.358]
#(10dB)[0.0794, 0.0832, 0.0868, 0.0906, 0.0943, 0.0981, 0.1018, 0.1056, 0.1094, 0.1132, 0.1171]#square


ph_if = [74.04e6, 77.88e6, 78.85e6, 78.85e6, 78.89e6, 80.00e6, 75.71e6, 80.00e6, 85.00e6]


amp = [5.0130e-04, 8.1523e-04, 0.00122458, 0.00150458, 0.00215715, 0.00246627, 0.00290790, 0.00604601]
mode =[0.0017, 0.0022, 0.0027, 0.0032, 0.0037, 0.0042, 0.0047, 0.0052, 0.0057, 0.0062, 0.0067]#[0.0042, 0.0047, 0.0052]#[0.0057, 0.0062, 0.0067]#[0.0072, 0.0077, 0.0082, 0.0087, 0.0092, 0.0097, 0.0102, 0.0107, 0.0112, 0.0117, 0.0122]

print(len(photon_amplitude), len(mode))
x= np.linspace(0, 499, 500)

## photon shape ### 

acquisition_time = fogi_duration + 80
num_of_cycles = 50000
repetition = 10


data = DataDict(
    time=dict(unit="ns"),
    fogi_frequency = dict(unit="GHz"),
    mode = dict(unit = ""),
    waveform_I=dict(axes=["time", "fogi_frequency", "mode"]),
    waveform_zero_fogi_I=dict(axes=["time", "fogi_frequency", "mode"]),

    waveform_Q=dict(axes=["time", "fogi_frequency", "mode"]),
    waveform_zero_fogi_Q=dict(axes=["time", "fogi_frequency", "mode"]),

    waveform=dict(axes=["time", "fogi_frequency", "mode"]),
    waveform_zero_fogi=dict(axes=["time", "fogi_frequency", "mode"]),

    qstate = dict(axes=["fogi_frequency", "mode"]), 
    qstate_zero_fogi = dict(axes=["fogi_frequency", "mode"]),
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

    for i in range(len(photon_amplitude)):
        ph_amp = photon_amplitude[i]
        m = mode[i]
        y_env = amp[4]*np.exp(-m*(x+26))*1e2
        time_reversed_waveform = (y_env*np.cos(2*np.pi* ph_if[4]*(x*1e-9)))[::-1]
        control_pulse=time_reversed_waveform *ph_amp

        def seq_I(fogi_amp, photon_amp=True):
            seq = Sequence(port_list = [qubit_drive_port, fogi_port, readout_port, dig_port, JPA_port])
            seq.call(reset_sequence)
            seq.trigger(ports)
            seq.call(seq_JPA)
            seq.add(Acquire(acquisition_time), dig_port)
            seq.add(Delay(20), readout_port)
            seq.trigger([readout_port, qubit_drive_port])
            seq.add(Delay(fogi_delay), fogi_port)
            seq.add(Square(amplitude=fogi_amp, duration=fogi_duration), fogi_port)
            if photon_amp:
                seq.add(Acquire(len(control_pulse)), readout_port) 
            # seq.add(SetDetuning(-(photon_freq - readout_lo_freq ) - readout_if_freq), readout_port)
            # seq.add(Square(amplitude = photon_amp, duration=photon_duration), readout_port)
            seq.trigger([qubit_drive_port,  readout_port])
            seq.add(Delay(60), readout_port)
            seq.add(SetDetuning(0), readout_port)
            seq.call(readout_seq)
            return seq
        # seq_I(0.5, 0.01).draw()
        # raise SystemError

        def seq_Q(fogi_amp, photon_amp=True):
            seq = Sequence(port_list = [qubit_drive_port, fogi_port, readout_port, dig_port, JPA_port])
            seq.call(reset_sequence)
            seq.trigger(ports)
            seq.call(seq_JPA_pi)
            seq.add(Acquire(acquisition_time), dig_port)
            seq.add(Delay(20), readout_port)
            seq.trigger([readout_port, qubit_drive_port])
            seq.add(Delay(fogi_delay), fogi_port)
            seq.add(Square(amplitude=fogi_amp, duration=fogi_duration), fogi_port)
            if photon_amp:
                seq.add(Acquire(len(control_pulse)), readout_port)
            # seq.add(SetDetuning(-(photon_freq - readout_lo_freq ) - readout_if_freq), readout_port)
            # seq.add(Square(amplitude = photon_amp, duration=photon_duration), readout_port)
            seq.trigger([qubit_drive_port,  readout_port])
            seq.add(Delay(60), readout_port)
            seq.add(SetDetuning(0), readout_port)
            seq.call(readout_seq)
            return seq
        # seq_Q(0.5, 0.01).draw()
        # raise SystemError
        
        for f in tqdm(fogi_freqs):
            fogi_freq = f
            fogi_port.if_freq = fogi_freq - fogi_lo_freq

            waveform_I = []
            waveform_zero_fogi_I = []
            waveform_offset_I = []
            waveform_offset_I_f = []
            qstate= []
            qstate_zero_fogi = []

            waveform_Q = []
            waveform_zero_fogi_Q = []
            waveform_offset_Q = []
            waveform_offset_Q_f = []

            waveform = []
            waveform_zero_fogi = []
            for _ in tqdm(range(repetition)):
                # load_sequence(seq_I(fogi_amplitude, ph_amp), cycles=num ,  chirp=True)           
                load_sequence_w_append(seq_I(fogi_amplitude, photon_amp=True), append_port=readout_port, waveform_appended=control_pulse, cycles=num_of_cycles)
                data = run(seq_I(fogi_amplitude, photon_amp=True)).mean(axis=0)
                waveform_I = np.append(waveform_I, data[0: acquisition_time//dig_ch.sampling_interval()]* voltage_step)
                qstate = np.append(qstate, demodulate(data[acquisition_time//dig_ch.sampling_interval():len(data)]))

                # load_sequence(seq_I(0, ph_amp), cycles=num_of_cycles,  chirp=True)
                load_sequence_w_append(seq_I(0, photon_amp=True), append_port=readout_port, waveform_appended=control_pulse, cycles=num_of_cycles)
                data1 = run(seq_I(0, photon_amp=True)).mean(axis=0)
                waveform_zero_fogi_I = np.append(waveform_zero_fogi_I, data1[0: acquisition_time//dig_ch.sampling_interval()]* voltage_step)
                qstate_zero_fogi = np.append(qstate_zero_fogi, demodulate(data1[acquisition_time//dig_ch.sampling_interval():len(data)]))
                
                # load_sequence(seq_I(0, 0), cycles=num_of_cycles,  chirp=True)
                load_sequence_w_append(seq_I(0, photon_amp=False), append_port=readout_port, waveform_appended=0, cycles=num_of_cycles)
                data2 = run(seq_I(0, photon_amp=False)).mean(axis=0)* voltage_step
                waveform_offset_I = np.append(waveform_offset_I, data2[0: acquisition_time//dig_ch.sampling_interval()])

                # load_sequence(seq_I(fogi, 0), cycles=num_of_cycles,  chirp=True)
                load_sequence_w_append(seq_I(fogi_amplitude, photon_amp=False), append_port=readout_port, waveform_appended=0, cycles=num_of_cycles)
                data2_f = run(seq_I(fogi_amplitude, photon_amp=False)).mean(axis=0)* voltage_step
                waveform_offset_I_f = np.append(waveform_offset_I_f, data2_f[0: acquisition_time//dig_ch.sampling_interval()])

                # load_sequence(seq_Q(fogi_amplitude, ph_amp), cycles=num_of_cycles,  chirp=True)
                load_sequence_w_append(seq_Q(fogi_amplitude, photon_amp=True), append_port=readout_port, waveform_appended=control_pulse, cycles=num_of_cycles)
                dataQ = run(seq_Q(fogi_amplitude, photon_amp=True)).mean(axis=0)* voltage_step
                waveform_Q = np.append(waveform_Q, dataQ[0: acquisition_time//dig_ch.sampling_interval()])
                # qstate_Q = np.append(qstate_Q, dataQ[acquisition_time//dig_ch.sampling_interval()-1:len(data)-1].mean(axis=1))

                # load_sequence(seq_Q(0, ph_amp), cycles=num_of_cycles,  chirp=True)
                load_sequence_w_append(seq_Q(0, photon_amp=True), append_port=readout_port, waveform_appended=control_pulse, cycles=num_of_cycles)
                data1Q = run(seq_Q(0, photon_amp=True)).mean(axis=0)* voltage_step
                waveform_zero_fogi_Q = np.append(waveform_zero_fogi_Q, data1Q[0: acquisition_time//dig_ch.sampling_interval()])
                # qstate_zero_fogi_Q = np.append(qstate_zero_fogi_Q, data1Q[acquisition_time//dig_ch.sampling_interval()-1:len(data)-1].mean(axis=1))
                
                # load_sequence(seq_Q(0, 0), cycles=num_of_cycles,  chirp=True)
                load_sequence_w_append(seq_Q(0, photon_amp=False), append_port=readout_port, waveform_appended=0, cycles=num_of_cycles)
                data2Q = run(seq_Q(0, photon_amp=False)).mean(axis=0)* voltage_step
                waveform_offset_Q = np.append(waveform_offset_Q, data2Q[0: acquisition_time//dig_ch.sampling_interval()])

                # load_sequence(seq_Q(0, 0), cycles=num_of_cycles,  chirp=True)
                load_sequence_w_append(seq_Q(fogi_amplitude, photon_amp=False), append_port=readout_port, waveform_appended=0, cycles=num_of_cycles)
                data2Q_f = run(seq_Q(fogi_amplitude, photon_amp=False)).mean(axis=0)* voltage_step
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
                mode = m,
                fogi_frequency = fogi_freq,
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