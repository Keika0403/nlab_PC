import os

import numpy as np
from setup_td import *
from sequence_parser import Sequence, Variable, Variables
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm
from setup_td_tomography import *
from time_reverse import *

measurement_name = os.path.basename(__file__)[:-3]


trigger_period = 3200
hvi_trigger.trigger_period(trigger_period) 
cd = "CDY168"


##### fogi pulse
pulse_dict = dict(
    data0 = dict(target_freq=9.35, date="2024-10-13", acquire_time="232754"),
    data1 = dict(target_freq=9.35, date="2024-10-13", acquire_time="232850"),
    data2 = dict(target_freq=9.35, date="2024-10-13", acquire_time="233101"),
    data3 = dict(target_freq=9.35, date="2024-10-13", acquire_time="233151"),

)


for k, v in tqdm(pulse_dict.items()):
    _, datadict = search_datadict_miyamura(f"D:\\K_Sunada\\result\\control_pulses\\{cd}", 
                                            v["date"], acquire_time=v["acquire_time"])
    target_shape_note = load_note(f"D:\\K_Sunada\\result\\control_pulses\\{cd}", v["date"],
                                f"{_}\\target_shape.md").replace("path", "before") + f"\npath : {_}"
    control_pulse_f = datadict["control_pulse"]["values"].ravel()

    fogi_duration = 1000
    fogi_delay = np.linspace(-100, 100, 6)#61

    ##### coherent pusle
    ph_if = readout_lo_freq*1e9 - v["target_freq"]*1e9
    x = np.linspace(0, 999, 1000)
    ph_amp = 0.07
    const = 2.5e-3 * 2 * np.pi
    ph_waveform = (const/2)**0.5/np.cosh(const * (x - fogi_duration/2))*np.cos(2*np.pi* ph_if*x*1e-9)
    # ph_waveform = (6*const**3/(np.pi**2))**0.5*(x - fogi_duration/2)/np.cosh(const * (x - fogi_duration/2))*np.cos(2*np.pi* ph_if*x*1e-9)
    # ph_waveform = (45*(2*const)**5/64/(np.pi**4))**0.5*((x - fogi_duration/2)**2-(np.pi/2/const)**2/3)/np.cosh(const * (x - fogi_duration/2))*np.cos(2*np.pi* ph_if*x*1e-9)
    # ph_waveform = (175*(2*const)**7)**0.5/24/(np.pi**3)*((x - fogi_duration/2)**3-(7*(x - fogi_duration/2)*(np.pi/(2*const))**2/5))/np.cosh(const * (x - fogi_duration/2))*np.cos(2*np.pi* ph_if*x*1e-9)


    port_list = [fogi_port, readout_port]
    acquisition_time = fogi_duration + 500
    num_of_cycles = 50000
    repetition = 20
    # plt.plot(x, ph_waveform)
    # plt.show()
    # raise SystemError

    data = DataDict(
        time=dict(unit="ns"),
        fogi_delay=dict(unit="ns"),
        waveform_I=dict(axes=["time", "fogi_delay"]),
        waveform_ref_I=dict(axes=["time", "fogi_delay"]),

        waveform_Q=dict(axes=["time", "fogi_delay"]),
        waveform_ref_Q=dict(axes=["time", "fogi_delay"]),

        waveform=dict(axes=["time", "fogi_delay"]),
        waveform_ref=dict(axes=["time", "fogi_delay"])

        # qstate = dict(axes=["fogi_delay"]), 
        # qstate_ref = dict(axes=["fogi_delay"]),
    )
    data.validate()

    lo1.output(True)
    lo2.output(True)
    lo3.output(True)

    try:
        with DDH5Writer(data, data_path, name=measurement_name) as writer:
            writer.add_tag(tags)
            writer.backup_file([__file__, setup_file, setup_parameters_file, setup_file_tomo])
            writer.save_text("wiring.md", wiring)
            # writer.save_text("target_shape.md", target_shape_note)
            writer.save_dict("station_snapshot.json", station.snapshot())

            for t in tqdm(fogi_delay):
                delay = t

                print(ph_if)
                control_pulse_p=ph_waveform *ph_amp
                control_pulses = [control_pulse_f, control_pulse_p]

                def abs_sequence(fogi:bool, photon:bool, JPA_direction=1) -> Sequence:
                    assert JPA_direction == -1 or 1
                    seq = Sequence(port_list=ports)
                    seq.call(reset_sequence)
                    seq.trigger(ports)
                    seq.add(Square(amplitude=JPA_direction*1.4, duration=len(control_pulse_f)+1000), JPA_port)
                    seq.add(Delay(500), readout_port)
                    seq.add(Delay(500), fogi_port)
                    seq.add(Delay(250), dig_port)
                    seq.add(Acquire(acquisition_time), dig_port)
                    seq.add(Delay(delay), fogi_port)
                    if fogi:
                        seq.add(Acquire(len(control_pulse_f)), fogi_port)
                    if photon:
                        seq.add(Acquire(len(control_pulse_p)), readout_port) 
                    seq.trigger(ports)
                    seq.add(SetDetuning(0), readout_port)
                    # seq.call(readout_seq)
                    return seq


                waveform_I = []
                waveform_ref_I = []
                waveform_offset_I = []
                waveform_offset_ref_I = []
                # qstate= []
                # qstate_ref = []

                waveform_Q = []
                waveform_ref_Q = []
                waveform_offset_Q = []
                waveform_offset_ref_Q = []

                waveform = []
                waveform_ref = []
                # delay = 0
                # abs_sequence(fogi=True, photon=True, JPA_direction=1).draw()
                # raise SystemError
                for _ in  tqdm(range(repetition)):
                    for state in ["abs_i", "abs_q", "ref_i", "ref_q", 
                                "offset_abs_i", "offset_abs_q", "offset_ref_i", "offset_ref_q",]:
                        awg1.flush_waveform()
                        awg2.flush_waveform()
                        if state=="abs_i":
                            seq = abs_sequence(fogi=True, photon=True, JPA_direction=1)
                            seq.draw()
                            raise SystemError
                        if state=="abs_q":
                            seq = abs_sequence(fogi=True, photon=True, JPA_direction=-1)
                        if state=="ref_i":
                            seq = abs_sequence(fogi=False, photon=True, JPA_direction=1)
                        if state=="ref_q":
                            seq = abs_sequence(fogi=False, photon=True, JPA_direction=-1)
                        if state=="offset_abs_i":
                            seq = abs_sequence(fogi=True, photon=False, JPA_direction=1)
                        if state=="offset_abs_q":
                            seq = abs_sequence(fogi=True, photon=False, JPA_direction=-1)
                        if state=="offset_ref_i":
                            seq = abs_sequence(fogi=False, photon=False, JPA_direction=1)
                        if state=="offset_ref_q":
                            seq = abs_sequence(fogi=False, photon=False, JPA_direction=-1)
                        load_sequence_w_two_append(seq, append_ports=port_list, waveforms_appended=control_pulses, cycles=num_of_cycles)
                        data = run(seq).mean(axis=0) * voltage_step
                        if state=="abs_i": waveform_I = np.append(waveform_I, data[0: acquisition_time//dig_ch.sampling_interval()])
                        # if state=="abs_i": qstate = np.append(qstate, demodulate(data[acquisition_time//dig_ch.sampling_interval():len(data)]))
                        if state=="abs_q": waveform_Q = np.append(waveform_Q, data[0: acquisition_time//dig_ch.sampling_interval()])
                        if state=="ref_i": waveform_ref_I = np.append(waveform_ref_I, data[0: acquisition_time//dig_ch.sampling_interval()])
                        # if state=="ref_i": qstate_ref = np.append(qstate_ref, demodulate(data[acquisition_time//dig_ch.sampling_interval():len(data)]))
                        if state=="ref_q": waveform_ref_Q = np.append(waveform_ref_Q, data[0: acquisition_time//dig_ch.sampling_interval()])
                        if state=="offset_abs_i": waveform_offset_I = np.append(waveform_offset_I, data[0: acquisition_time//dig_ch.sampling_interval()])
                        if state=="offset_abs_q": waveform_offset_Q = np.append(waveform_offset_Q, data[0: acquisition_time//dig_ch.sampling_interval()])
                        if state=="offset_ref_i": waveform_offset_ref_I = np.append(waveform_offset_ref_I, data[0: acquisition_time//dig_ch.sampling_interval()])
                        if state=="offset_ref_q": waveform_offset_ref_Q = np.append(waveform_offset_ref_Q, data[0: acquisition_time//dig_ch.sampling_interval()])
                waveform_I = waveform_I.reshape(int(repetition), acquisition_time//dig_ch.sampling_interval())
                waveform_ref_I = waveform_ref_I.reshape(int(repetition), acquisition_time//dig_ch.sampling_interval())
                waveform_offset_I = waveform_offset_I.reshape(int(repetition), acquisition_time//dig_ch.sampling_interval())
                waveform_offset_ref_I = waveform_offset_ref_I.reshape(int(repetition), acquisition_time//dig_ch.sampling_interval())
                waveform_Q = waveform_Q.reshape(int(repetition), acquisition_time//dig_ch.sampling_interval())
                waveform_ref_Q = waveform_ref_Q.reshape(int(repetition), acquisition_time//dig_ch.sampling_interval())
                waveform_offset_Q = waveform_offset_Q.reshape(int(repetition), acquisition_time//dig_ch.sampling_interval())
                waveform_offset_ref_Q = waveform_offset_ref_Q.reshape(int(repetition), acquisition_time//dig_ch.sampling_interval())
                

                waveform_I = np.array(waveform_I).mean(axis=0)
                waveform_ref_I = np.array(waveform_ref_I).mean(axis=0)
                waveform_offset_I = np.array(waveform_offset_I).mean(axis=0)
                waveform_offset_ref_I = np.array(waveform_offset_ref_I).mean(axis=0)
                waveform_Q = np.array(waveform_Q).mean(axis=0)
                waveform_ref_Q = np.array(waveform_ref_Q).mean(axis=0)
                waveform_offset_Q = np.array(waveform_offset_Q).mean(axis=0)
                waveform_offset_ref_Q = np.array(waveform_offset_ref_Q).mean(axis=0)
                # qstate = np.array(qstate).mean(axis=0)
                # qstate_ref =np.array(qstate_ref).mean(axis=0)

                waveform_I = (waveform_I - waveform_offset_I)/2
                waveform_Q = (waveform_Q - waveform_offset_Q)/2
                waveform = (waveform_I + waveform_Q) / 2
                waveform_ref_I = (waveform_ref_I - waveform_offset_ref_I)/2
                waveform_ref_Q = (waveform_ref_Q - waveform_offset_ref_Q)/2
                waveform_ref = (waveform_ref_I + waveform_ref_Q) / 2

                writer.add_data(
                    time = dig_ch.sampling_interval()*np.arange(len(waveform_I)),
                    fogi_delay = delay,
                    waveform_I = waveform_I,
                    waveform_ref_I = waveform_ref_I,
                    waveform_Q = waveform_Q,
                    waveform_ref_Q = waveform_ref_Q,
                    waveform = waveform,
                    waveform_ref = waveform_ref
                    # qstate = qstate,
                    # qstate_ref=qstate_ref,
                )
    finally:
        off()
        print('finished')
