from setup_td_tomography import *

measurement_name = os.path.basename(__file__)[:-3]
tags.append("photon")

cd = "CDY160"
cycles = 50000
repetition = 5
hvi_trigger.digitizer_delay(400)
hvi_trigger.trigger_period(100000)

pulse_dict_tx = dict(
    data0 = dict(target_freq=9.266, date="2024-06-23", acquire_time="110009"),
    # data1 = dict(target_freq=9.269, date="2024-06-17", Sacquire_time="152721"),
    # data2 = dict(target_freq=9.2715, date="2024-06-17", acquire_time="152722"),
    # data3 = dict(target_freq=10.26, date="2024-05-23", acquire_tim5e="112726"),
    # data4 = dict(target_freq=10.35, date="2024-05-13", acquire_time="121745"),
)
pulse_dict_rx = dict(
    data0 = dict(target_freq=9.266, date="2024-06-24", acquire_time="141521"),
    # data1 = dict(target_freq=9.269, date="2024-06-17", acquire_time="152721"),
    # data2 = dict(target_freq=9.2715, date="2024-06-17", acquire_time="152722"),
    # data3 = dict(target_freq=10.26, date="2024-05-23", acquire_tim5e="112726"),
    # data4 = dict(target_freq=10.35, date="2024-05-13", acquire_time="121745"),
)

for k, v in tqdm(pulse_dict_tx.items()):
    _, datadict_tx = search_datadict_miyamura(f"D:\\K_Sunada\\result\\control_pulses\\{cd}\\", 
                                            v["date"], acquire_time=v["acquire_time"])
    target_shape_note_tx = load_note(f"D:\\K_Sunada\\result\\control_pulses\\{cd}\\", v["date"],
                                f"{_}\\target_shape.md").replace("path", "before") + f"\npath : {_}"
    fogi_pulse_tx = datadict_tx["control_pulse"]["values"].ravel()

    for k, v in tqdm(pulse_dict_rx.items()):
        _, datadict_rx = search_datadict_miyamura(f"D:\\K_Sunada\\result\\control_pulses\\{cd}\\", 
                                                v["date"], acquire_time=v["acquire_time"])
        target_shape_note_rx = load_note(f"D:\\K_Sunada\\result\\control_pulses\\{cd}\\", v["date"],
                                    f"{_}\\target_shape.md").replace("path", "before") + f"\npath : {_}"
        fogi_pulse_rx = datadict_rx["control_pulse"]["values"].ravel()

        fogi_ports = [fogi_port_tx, fogi_port_rx]
        fogi_pulses = [fogi_pulse_tx, fogi_pulse_rx]

        def comm_sequence(half_pi:bool, abs:bool, JPA_direction=1) -> Sequence:
            assert JPA_direction == -1 or 1
            seq = Sequence(port_list=ports)
            seq.add(Square(amplitude=JPA_direction*0.41, duration=len(fogi_pulse_tx)+500), JPA_port)
            if half_pi:
                seq.call(ge_half_pi_seq_tx)
            else:
                seq.add(VirtualZ(np.pi), qubit_drive_port_tx)
                seq.call(ge_half_pi_seq_tx)
                seq.add(VirtualZ(np.pi), qubit_drive_port_tx)
            seq.call(ef_pi_seq_tx)
            seq.trigger(ports_wo_JPA)
            seq.add(Acquire(len(fogi_pulse_tx)), fogi_port_tx)
            if abs:
                seq.add(Delay(300), fogi_port_rx)
                seq.add(Acquire(len(fogi_pulse_rx)), fogi_port_rx)
            seq.add(Acquire(len(fogi_pulse_tx) + 500), dig_port)
            return seq
        
        # comm_sequence(half_pi=True, abs=True, JPA_direction=1).draw()
        # raise SystemError

        data = DataDict(
            time=dict(unit="ns"),

            waveform_ref=dict(axes=["time"], unit="V"),
            g_plus_e_I_ref=dict(axes=["time"], unit="V"),
            g_minus_e_I_ref=dict(axes=["time"], unit="V"),
            g_plus_e_Q_ref=dict(axes=["time"], unit="V"),
            g_minus_e_Q_ref=dict(axes=["time"], unit="V"),

            waveform_abs=dict(axes=["time"], unit="V"),
            g_plus_e_I_abs=dict(axes=["time"], unit="V"),
            g_minus_e_I_abs=dict(axes=["time"], unit="V"),
            g_plus_e_Q_abs=dict(axes=["time"], unit="V"),
            g_minus_e_Q_abs=dict(axes=["time"], unit="V"),
        )
        data.validate()

        try:
            with DDH5Writer(data, data_path, name=measurement_name) as writer:
                writer.add_tag(tags)
                writer.backup_file([__file__, setup_file, setup_parameters_file, setup_file_tomo])
                writer.save_text("wiring.md", wiring)
                writer.save_text("target_shape_tx.md", target_shape_note_tx)
                writer.save_text("target_shape_rx.md", target_shape_note_rx)
                writer.save_dict("station_snapshot.json", station.snapshot())
                g_plus_e_I_ref=[]
                g_plus_e_Q_ref=[]
                g_minus_e_I_ref=[]
                g_minus_e_Q_ref=[]
                g_plus_e_I_abs=[]
                g_plus_e_Q_abs=[]
                g_minus_e_I_abs=[]
                g_minus_e_Q_abs=[]
                for _ in  tqdm(range(repetition)):
                    for state in ["0+1_i_ref", "0+1_q_ref", "0-1_i_ref", "0-1_q_ref", "0+1_i_abs", "0+1_q_abs", "0-1_i_abs", "0-1_q_abs"]:
                        awg1.flush_waveform()
                        awg2.flush_waveform()
                        awg3.flush_waveform()
                        if state=="0+1_i_ref":
                            seq = comm_sequence(half_pi=True, abs=False, JPA_direction=1)
                        if state=="0+1_q_ref":
                            seq = comm_sequence(half_pi=True, abs=False, JPA_direction=-1)
                        if state=="0-1_i_ref":
                            seq = comm_sequence(half_pi=False, abs=False, JPA_direction=1)
                        if state=="0-1_q_ref":
                            seq = comm_sequence(half_pi=False, abs=False, JPA_direction=-1)
                        if state=="0+1_i_abs":
                            seq = comm_sequence(half_pi=True, abs=True, JPA_direction=1)
                        if state=="0+1_q_abs":
                            seq = comm_sequence(half_pi=True, abs=True, JPA_direction=-1)
                        if state=="0-1_i_abs":
                            seq = comm_sequence(half_pi=False, abs=True, JPA_direction=1)
                        if state=="0-1_q_abs":
                            seq = comm_sequence(half_pi=False, abs=True, JPA_direction=-1)
                        load_sequence_w_append_comm(seq, append_ports=fogi_ports, waveforms_appended=fogi_pulses, cycles=cycles)
                        data = run_comm(seq).mean(axis=0) * voltage_step_comm
                        if state=="0+1_i_ref": g_plus_e_I_ref = np.append(g_plus_e_I_ref, data)
                        if state=="0+1_q_ref": g_plus_e_Q_ref = np.append(g_plus_e_Q_ref, data)
                        if state=="0-1_i_ref": g_minus_e_I_ref = np.append(g_minus_e_I_ref, data)
                        if state=="0-1_q_ref": g_minus_e_Q_ref = np.append(g_minus_e_Q_ref, data)
                        if state=="0+1_i_abs": g_plus_e_I_abs = np.append(g_plus_e_I_abs, data)
                        if state=="0+1_q_abs": g_plus_e_Q_abs = np.append(g_plus_e_Q_abs, data)
                        if state=="0-1_i_abs": g_minus_e_I_abs = np.append(g_minus_e_I_abs, data)
                        if state=="0-1_q_abs": g_minus_e_Q_abs = np.append(g_minus_e_Q_abs, data)
                g_plus_e_I_ref = g_plus_e_I_ref.reshape(int(repetition), int(dig_ch_comm.points_per_cycle())).mean(axis=0)
                g_plus_e_Q_ref = g_plus_e_Q_ref.reshape(int(repetition), int(dig_ch_comm.points_per_cycle())).mean(axis=0)
                g_minus_e_I_ref = g_minus_e_I_ref.reshape(int(repetition), int(dig_ch_comm.points_per_cycle())).mean(axis=0)
                g_minus_e_Q_ref = g_minus_e_Q_ref.reshape(int(repetition), int(dig_ch_comm.points_per_cycle())).mean(axis=0)
                g_plus_e_I_abs = g_plus_e_I_abs.reshape(int(repetition), int(dig_ch_comm.points_per_cycle())).mean(axis=0)
                g_plus_e_Q_abs = g_plus_e_Q_abs.reshape(int(repetition), int(dig_ch_comm.points_per_cycle())).mean(axis=0)
                g_minus_e_I_abs = g_minus_e_I_abs.reshape(int(repetition), int(dig_ch_comm.points_per_cycle())).mean(axis=0)
                g_minus_e_Q_abs = g_minus_e_Q_abs.reshape(int(repetition), int(dig_ch_comm.points_per_cycle())).mean(axis=0)
                waveform_I_ref = (g_plus_e_I_ref - g_minus_e_I_ref)/2
                waveform_Q_ref = (g_plus_e_Q_ref - g_minus_e_Q_ref)/2
                waveform_ref = (waveform_I_ref + waveform_Q_ref) / 2
                waveform_I_abs = (g_plus_e_I_abs - g_minus_e_I_abs)/2
                waveform_Q_abs = (g_plus_e_Q_abs - g_minus_e_Q_abs)/2
                waveform_abs = (waveform_I_abs + waveform_Q_abs) / 2
                writer.add_data(
                    time=np.arange(len(waveform_ref)) * dig_ch_comm.sampling_interval(),
                    waveform_ref=waveform_ref,
                    g_plus_e_I_ref=g_plus_e_I_ref,
                    g_minus_e_I_ref=g_minus_e_I_ref,
                    g_plus_e_Q_ref=g_plus_e_Q_ref,
                    g_minus_e_Q_ref=g_minus_e_Q_ref,
                    waveform_abs=waveform_abs,
                    g_plus_e_I_abs=g_plus_e_I_abs,
                    g_minus_e_I_abs=g_minus_e_I_abs,
                    g_plus_e_Q_abs=g_plus_e_Q_abs,
                    g_minus_e_Q_abs=g_minus_e_Q_abs,
                )
        finally:
            off_comm()
            print('finished')
