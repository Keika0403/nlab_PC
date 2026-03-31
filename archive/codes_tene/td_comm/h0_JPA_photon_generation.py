from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]
tags.append("photon")

cycles = 50000
repetition = 10
hvi_trigger.digitizer_delay(400)
hvi_trigger.trigger_period(3000)
cd = "CDK184"
JPA_amp = 0.72

pulse_dict = dict(
    # data0 = dict(target_freq=9.38, date="2025-08-14", acquire_time="114630"),
    data1 = dict(target_freq=9.38, date="2025-08-15", acquire_time="192449"),
    # data2 = dict(target_freq=9.38, date="2025-08-14", acquire_time="114633"),
    # data3 = dict(target_freq=9.38, date="2025-08-14", acquire_time="114634"),
    # data4 = dict(target_freq=9.38, date="2025-08-14", acquire_time="114636"),
    # data5 = dict(target_freq=9.38, date="2025-08-14", acquire_time="114637"),
    # data6 = dict(target_freq=9.38, date="2025-08-14", acquire_time="114639"),
    data7 = dict(target_freq=9.38, date="2025-08-15", acquire_time="192504"),

    # data0 = dict(target_freq=9.38, date="2025-08-14", acquire_time="121001"),
    # data1 = dict(target_freq=9.38, date="2025-08-14", acquire_time="121002"),
    # data2 = dict(target_freq=9.38, date="2025-08-14", acquire_time="121004"),
    # data3 = dict(target_freq=9.38, date="2025-08-14", acquire_time="121005"),
    # data4 = dict(target_freq=9.38, date="2025-08-14", acquire_time="121007"),
    # data5 = dict(target_freq=9.38, date="2025-08-14", acquire_time="121009"),
    # data6 = dict(target_freq=9.38, date="2025-08-14", acquire_time="121011"),
    # data7 = dict(target_freq=9.38, date="2025-08-14", acquire_time="121013"),
)

for k, v in tqdm(pulse_dict.items()):
    _, datadict = search_datadict_miyamura(f"D:\\K_Sunada\\result\\control_pulses\\{cd}", 
                                            v["date"], acquire_time=v["acquire_time"])
    target_shape_note = load_note(f"D:\\K_Sunada\\result\\control_pulses\\{cd}", v["date"],
                                f"{_}\\target_shape.md").replace("path", "before") + f"\npath : {_}"
    control_pulse = datadict["control_pulse"]["values"].ravel()

    if measure_which == "tx":
        fogi_port = fogi_port_tx
        def fogi_sequence(half_pi:bool, JPA_direction=1) -> Sequence:
            assert JPA_direction == -1 or 1
            seq = Sequence(port_list=ports_tx)
            seq.call(reset_sequence_tx)
            seq.trigger(ports_tx)
            seq.add(Square(amplitude=JPA_direction*JPA_amp, duration=len(control_pulse)+100), JPA_port_tx)
            if half_pi:
                seq.call(ge_half_pi_seq_tx)
            else:
                seq.add(VirtualZ(np.pi), qubit_drive_port_tx)
                seq.call(ge_half_pi_seq_tx)
                seq.add(VirtualZ(np.pi), qubit_drive_port_tx)
            seq.call(ef_pi_seq_tx)
            seq.trigger(ports_tx_wo_JPA)
            seq.add(Acquire(len(control_pulse)), fogi_port_tx)
            seq.add(Acquire(len(control_pulse) + 500), dig_port_tx)
            return seq
    elif measure_which == "txQrx":
        fogi_port = fogi_port_rx
        def fogi_sequence(half_pi:bool, JPA_direction=1) -> Sequence:
            assert JPA_direction == -1 or 1
            seq = Sequence(port_list=ports_txQrx)
            seq.call(reset_sequence_rx)
            seq.trigger(ports_txQrx)
            seq.add(Square(amplitude=JPA_direction*JPA_amp, duration=len(control_pulse)+100), JPA_port_tx)
            if half_pi:
                seq.call(ge_half_pi_seq_rx)
            else:
                seq.add(VirtualZ(np.pi), qubit_drive_port_rx)
                seq.call(ge_half_pi_seq_rx)
                seq.add(VirtualZ(np.pi), qubit_drive_port_rx)
            seq.call(ef_pi_seq_rx)
            seq.trigger(ports_txQrx_wo_JPA)
            seq.add(Acquire(len(control_pulse)), fogi_port_rx)
            seq.add(Acquire(len(control_pulse) + 500), dig_port_tx)
            return seq
    elif measure_which == "rx":
        fogi_port = fogi_port_rx
        def fogi_sequence(half_pi:bool, JPA_direction=1) -> Sequence:
            assert JPA_direction == -1 or 1
            seq = Sequence(port_list=ports_rx)
            seq.call(reset_sequence_rx)
            seq.trigger(ports_rx)
            seq.add(Square(amplitude=JPA_direction*JPA_amp, duration=len(control_pulse)+100), JPA_port_tx)
            if half_pi:
                seq.call(ge_half_pi_seq_rx)
            else:
                seq.add(VirtualZ(np.pi), qubit_drive_port_rx)
                seq.call(ge_half_pi_seq_rx)
                seq.add(VirtualZ(np.pi), qubit_drive_port_rx)
            seq.call(ef_pi_seq_rx)
            seq.trigger(ports_rx_wo_JPA)
            seq.add(Acquire(len(control_pulse)), fogi_port_rx)
            seq.add(Acquire(len(control_pulse) + 500), dig_port_rx)
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
            writer.backup_file([__file__, setup_file, setup_parameters_file])
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
                    awg3.flush_waveform()
                    if state=="0+1_i":
                        seq = fogi_sequence(half_pi=True, JPA_direction=1)
                        # seq.draw()
                        # raise SystemError
                    if state=="0+1_q":
                        seq = fogi_sequence(half_pi=True, JPA_direction=-1)
                    if state=="0-1_i":
                        seq = fogi_sequence(half_pi=False, JPA_direction=1)
                    if state=="0-1_q":
                        seq = fogi_sequence(half_pi=False, JPA_direction=-1)
                    if measure_which == "tx":
                        load_sequence_w_append(seq, append_ports=[fogi_port_tx], waveforms_added=[control_pulse], cycles=cycles)
                    elif measure_which == "txQrx":
                        load_sequence_w_append(seq, append_ports=[fogi_port_rx], waveforms_added=[control_pulse], cycles=cycles)
                    data = run(seq, which=measure_which[:2]).mean(axis=0) * voltage_step_tx
                    if state=="0+1_i": g_plus_e_I = np.append(g_plus_e_I, data)
                    if state=="0+1_q": g_plus_e_Q = np.append(g_plus_e_Q, data)
                    if state=="0-1_i": g_minus_e_I = np.append(g_minus_e_I, data)
                    if state=="0-1_q": g_minus_e_Q = np.append(g_minus_e_Q, data)
            g_plus_e_I = g_plus_e_I.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
            g_plus_e_Q = g_plus_e_Q.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
            g_minus_e_I = g_minus_e_I.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
            g_minus_e_Q = g_minus_e_Q.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
            waveform_I = (g_plus_e_I - g_minus_e_I)/2
            waveform_Q = (g_plus_e_Q - g_minus_e_Q)/2
            waveform = (waveform_I + waveform_Q) / 2
            writer.add_data(
                time=np.arange(len(waveform)) * dig_ch_tx.sampling_interval(),
                waveform=waveform,
                g_plus_e_I=g_plus_e_I,
                g_minus_e_I=g_minus_e_I,
                g_plus_e_Q=g_plus_e_Q,
                g_minus_e_Q=g_minus_e_Q,
            )
    finally:
        off()
        print('finished')
