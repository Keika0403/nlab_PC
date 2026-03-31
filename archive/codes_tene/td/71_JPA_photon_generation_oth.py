from setup_td_tomography import *

measurement_name = os.path.basename(__file__)[:-3]
tags.append("photon")

cycles = 50000
repetition = 20
hvi_trigger.digitizer_delay(400)
hvi_trigger.trigger_period(100000)
cd = "CDY167"

pulse_dict = dict(
    data0 = dict(target_freq=9.36, date="2024-10-07", acquire_time="130854"),
    data1 = dict(target_freq=9.36, date="2024-10-07", acquire_time="131102"),
    # data2 = dict(target_freq=9.35, date="2024-10-04", acquire_time="171749"),
    # data3 = dict(target_freq=9.35, date="2024-10-04", acquire_time="171916"),
)   

for k, v in tqdm(pulse_dict.items()):
    _, datadict = search_datadict_miyamura(f"D:\\K_Sunada\\result\\control_pulses\\{cd}", 
                                            v["date"], acquire_time=v["acquire_time"])
    target_shape_note = load_note(f"D:\\K_Sunada\\result\\control_pulses\\{cd}", v["date"],
                                f"{_}\\target_shape_othogonal.md").replace("path", "before") + f"\npath : {_}"
    control_pulse = datadict["control_pulse"]["values"].ravel()

    def fogi_sequence(half_pi:bool, JPA_direction=1) -> Sequence:
        assert JPA_direction == -1 or 1
        seq = Sequence(port_list=ports)
        seq.add(Square(amplitude=JPA_direction*1.0, duration=len(control_pulse)+100), JPA_port)
        if half_pi:
            seq.call(ge_half_pi_seq)
        else:
            seq.add(VirtualZ(np.pi), qubit_drive_port)
            seq.call(ge_half_pi_seq)
            seq.add(VirtualZ(np.pi), qubit_drive_port)
        seq.call(ef_pi_seq)
        seq.trigger(ports_wo_JPA)
        seq.add(Acquire(len(control_pulse)), fogi_port)
        seq.add(Acquire(len(control_pulse) + 500), dig_port)
        return seq

    data = DataDict(
        time=dict(unit="ns"),
        waveform=dict(axes=["time"], unit="V"),
        g_plus_e_I=dict(axes=["time"], unit="V"),
        g_minus_e_I=dict(axes=["time"], unit="V"),
        g_plus_e_Q=dict(axes=["time"], unit="V"),
        g_minus_e_Q=dict(axes=["time"], unit="V"),
    )
    data.validate()

    try:
        with DDH5Writer(data, data_path, name=measurement_name) as writer:
            writer.add_tag(tags)
            writer.backup_file([__file__, setup_file, setup_parameters_file, setup_file_tomo])
            writer.save_text("wiring.md", wiring)
            writer.save_text("target_shape.md", target_shape_note)
            writer.save_dict("station_snapshot.json", station.snapshot())
            g_plus_e_I=[]
            g_plus_e_Q=[]
            g_minus_e_I=[]
            g_minus_e_Q=[]
            for _ in  tqdm(range(repetition)):
                for state in ["0+1_i", "0+1_q", "0-1_i", "0-1_q"]:
                    awg1.flush_waveform()
                    awg2.flush_waveform()
                    if state=="0+1_i":
                        seq = fogi_sequence(half_pi=True, JPA_direction=1)
                    if state=="0+1_q":
                        seq = fogi_sequence(half_pi=True, JPA_direction=-1)
                    if state=="0-1_i":
                        seq = fogi_sequence(half_pi=False, JPA_direction=1)
                    if state=="0-1_q":
                        seq = fogi_sequence(half_pi=False, JPA_direction=-1)
                    load_sequence_w_append(seq, append_port=fogi_port, waveform_appended=control_pulse, cycles=cycles)
                    data = run(seq).mean(axis=0) * voltage_step
                    if state=="0+1_i": g_plus_e_I = np.append(g_plus_e_I, data)
                    if state=="0+1_q": g_plus_e_Q = np.append(g_plus_e_Q, data)
                    if state=="0-1_i": g_minus_e_I = np.append(g_minus_e_I, data)
                    if state=="0-1_q": g_minus_e_Q = np.append(g_minus_e_Q, data)
            g_plus_e_I = g_plus_e_I.reshape(int(repetition), int(dig_ch.points_per_cycle())).mean(axis=0)
            g_plus_e_Q = g_plus_e_Q.reshape(int(repetition), int(dig_ch.points_per_cycle())).mean(axis=0)
            g_minus_e_I = g_minus_e_I.reshape(int(repetition), int(dig_ch.points_per_cycle())).mean(axis=0)
            g_minus_e_Q = g_minus_e_Q.reshape(int(repetition), int(dig_ch.points_per_cycle())).mean(axis=0)
            waveform_I = (g_plus_e_I - g_minus_e_I)/2
            waveform_Q = (g_plus_e_Q - g_minus_e_Q)/2
            waveform = (waveform_I + waveform_Q) / 2
            writer.add_data(
                time=np.arange(len(waveform)) * dig_ch.sampling_interval(),
                waveform=waveform,
                g_plus_e_I=g_plus_e_I,
                g_minus_e_I=g_minus_e_I,
                g_plus_e_Q=g_plus_e_Q,
                g_minus_e_Q=g_minus_e_Q,
            )
    finally:
        off()
        print('finished')
