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


fogi_freq = 5.3835#np.linspace(5.381, 5.386, 11) #5.384
fogi_amplitude = 0.6
# fogi_duration = 500
fogi_delay = 30
JPA_pump.params["duration"] = 1000 + 80
JPA_pump_pi.params["duration"] = 1000 + 80

ph_lengths = [400, 500, 600, 700, 800, 900, 999]
photon_amplitude =[1.362]# [0.282, 0.515, 0.728, 0.891, 1.029, 1.151, 1.261, 1.362] #tr
# photon_amplitude =[0.338, 0.617, 0.872, 1.068, 1.233] #square


ph_if = [74.04e6, 77.88e6, 78.85e6, 78.85e6, 78.89e6, 80.00e6, 75.71e6, 80.00e6, 85.00e6]
data_id = 4
# time_reversed_waveform = np.array(square(ph_if[4], 0.05, 500) )

# time_reversed_waveform = square(ph_if[data_id], 0.05, 500)
## photon shape ### 
id = 4
duration = 520
header = "D:/K_sunada/result/CDY154/"
data = "/2024-01-17/2024-01-17T143735_84bc5e10-70_JPA_photon_generation"
dd = datadict_from_hdf5(header+data+"/data")
x = dd['time']['values'][id][0:duration]
y = dd['waveform']['values'][id][0:duration]
y_shift =  y[13:duration-7]* np.exp(-1j*2 *np.pi* ph_if[id]*(x[13:duration-7]*1e-9))
y_LPF = np.array(np.abs(LPF(y_shift, 500e6, 15e6, 40e6, 5, 40))*2)

x = x[0+13:250+13]
y = y[0+13:250+13]*1e2
amp = [5.0130e-04, 8.1523e-04, 0.00122458, 0.00150458, 0.00215715, 0.00246627, 0.00290790, 0.00604601]
gamma = [0.00147027, 0.00264687, 0.00458680, 0.00656348, 0.00972759	, 0.01271420, 0.01645340, 0.02609383]
x = np.linspace(0, 499, 500)
y_env = amp[id]*np.exp(-(gamma[id]/2)*(x+26))*1e2
time_reversed_waveform = (y_env*np.cos(2*np.pi* ph_if[4]*(x*1e-9)))[::-1]
## photon shape ### 


acquisition_time = 1000 + 80
num_of_cycles = 50000
repetition = 15


data = DataDict(
    time=dict(unit="ns"),
    ph_length = dict(unit="GHz"),
    photon_amp = dict(unit = ""),
    waveform_I=dict(axes=["time", "ph_length", "photon_amp"]),
    waveform_zero_fogi_I=dict(axes=["time", "ph_length", "photon_amp"]),

    waveform_Q=dict(axes=["time", "ph_length", "photon_amp"]),
    waveform_zero_fogi_Q=dict(axes=["time", "ph_length", "photon_amp"]),

    waveform=dict(axes=["time", "ph_length", "photon_amp"]),
    waveform_zero_fogi=dict(axes=["time", "ph_length", "photon_amp"]),

    qstate = dict(axes=["ph_length", "photon_amp"]), 
    qstate_zero_fogi = dict(axes=["ph_length", "photon_amp"]),
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
        for l in tqdm(ph_lengths):
            fogi_port.if_freq = fogi_freq - fogi_lo_freq
            length = int(l)
            fogi_duration = length
            x = np.linspace(0, length-1, length)
            y_env = amp[id]*np.exp(-(gamma[id]/2)*(x+26))*1e2
            time_reversed_waveform = (y_env*np.cos(2*np.pi* ph_if[4]*(x*1e-9)))[::-1]
            control_pulse=time_reversed_waveform *p

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
                photon_amp = ph_amp,
                ph_length = length,
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