from setup_td import *
from itertools import product

measurement_name = os.path.basename(__file__)[:-3]
tags.append("photon")

cycles = 50000
repetition = 5
hvi_trigger.digitizer_delay(400)
hvi_trigger.trigger_period(30000)
cd = "CDK184"
JPA_amp = 0.7

# fogi_rx_delays = np.linspace(158, 278, 21)
fogi_rx_delays = np.linspace(218, 278, 1)
# fogi_rx_delays = np.linspace(218, 278, 1)
# fogi_rx_delays = np.linspace(150+76, 310, 1)
# (0, 0)

pulse_dict_tx = dict(
    # data0 = dict(target_freq=9.38, date="2025-08-14", acquire_time="114630"),
    data1 = dict(target_freq=9.38, date="2025-08-14", acquire_time="114631"),
    # data2 = dict(target_freq=9.38, date="2025-08-14", acquire_time="114633"),
    # data3 = dict(target_freq=9.38, date="2025-08-14", acquire_time="114634"),
    # data4 = dict(target_freq=9.38, date="2025-08-14", acquire_time="114636"),
    # data5 = dict(target_freq=9.38, date="2025-08-14", acquire_time="114637"),
    # data6 = dict(target_freq=9.38, date="2025-08-14", acquire_time="114639"),
    # data7 = dict(target_freq=9.38, date="2025-08-14", acquire_time="114640"),
)
pulse_dict_rx = dict(
    # data0 = dict(target_freq=9.38, date="2025-08-14", acquire_time="122215"),
    data1 = dict(target_freq=9.38, date="2025-08-14", acquire_time="122217"),
    # data2 = dict(target_freq=9.38, date="2025-08-14", acquire_time="122219"),
    # data3 = dict(target_freq=9.38, date="2025-08-14", acquire_time="122221"),
    # data4 = dict(target_freq=9.38, date="2025-08-14", acquire_time="122222"),
    # data5 = dict(target_freq=9.38, date="2025-08-14", acquire_time="122224"),
    # data6 = dict(target_freq=9.38, date="2025-08-14", acquire_time="122226"),
    # data7 = dict(target_freq=9.38, date="2025-08-14", acquire_time="122228"),
)

for (k_tx, v_tx), (k_rx, v_rx) in tqdm(product(pulse_dict_tx.items(), pulse_dict_rx.items())):
    _, datadict_tx = search_datadict_miyamura(f"D:\\K_Sunada\\result\\control_pulses\\{cd}", 
                                            v_tx["date"], acquire_time=v_tx["acquire_time"])
    target_shape_note_tx = load_note(f"D:\\K_Sunada\\result\\control_pulses\\{cd}", v_tx["date"],
                                f"{_}\\target_shape.md").replace("path", "before") + f"\npath : {_}"
    control_pulse_tx = datadict_tx["control_pulse"]["values"].ravel()

    _, datadict_rx = search_datadict_miyamura(f"D:\\K_Sunada\\result\\control_pulses\\{cd}", 
                                            v_rx["date"], acquire_time=v_rx["acquire_time"])
    target_shape_note_rx = load_note(f"D:\\K_Sunada\\result\\control_pulses\\{cd}", v_rx["date"],
                                f"{_}\\target_shape.md").replace("path", "before") + f"\npath : {_}"
    control_pulse_rx = datadict_rx["control_pulse"]["values"].ravel()

    # You should exclude dig_port_rx & readout_port_rx from ports
    def comm_sequence(delay:int, half_pi:bool, fogi_rx:bool, JPA_direction=1) -> Sequence:
        assert JPA_direction == -1 or 1
        seq = Sequence(port_list=ports)
        seq.call(reset_sequence_tx)
        seq.call(reset_sequence_rx)
        seq.trigger(ports)
        seq.add(Delay(104), qubit_drive_port_tx)
        if half_pi:
            seq.call(ge_half_pi_seq_tx)
        else:
            seq.add(VirtualZ(np.pi), qubit_drive_port_tx)
            seq.call(ge_half_pi_seq_tx)
            seq.add(VirtualZ(np.pi), qubit_drive_port_tx)
        seq.call(ef_pi_seq_tx)
        seq.trigger([qubit_drive_port_tx, fogi_port_tx])
        seq.add(Square(amplitude=JPA_direction*JPA_amp, duration=len(control_pulse_tx)+500), JPA_port_tx)
        seq.add(Acquire(len(control_pulse_tx)), fogi_port_tx)
        seq.add(Acquire(len(control_pulse_tx)+500), dig_port_tx)
        seq.add(Delay(delay), qubit_drive_port_rx)
        seq.trigger(ports_rx)
        if fogi_rx:
            seq.add(Acquire(len(control_pulse_rx)), fogi_port_rx)
            seq.trigger([qubit_drive_port_rx, fogi_port_rx])
            seq.call(ef_pi_seq_rx)
        return seq

    data = DataDict(
        time=dict(unit="ns"),
        delay=dict(unit="ns"),
        waveform=dict(axes=["time", "delay"], unit="V"),
        waveform_wo_fogi_rx=dict(axes=["time", "delay"], unit="V"),
    )
    data.validate()

    try:
        with DDH5Writer(data, data_path, name=measurement_name) as writer:
            writer.add_tag(tags)
            writer.backup_file([__file__, setup_file, setup_parameters_file])
            writer.save_text("wiring.md", wiring)
            writer.save_text("target_shape_tx.md", target_shape_note_tx)
            writer.save_text("target_shape_rx.md", target_shape_note_rx)
            writer.save_dict("station_snapshot.json", station.snapshot())

            for t in tqdm(fogi_rx_delays):
                g_plus_e_I=[]
                g_plus_e_Q=[]
                g_minus_e_I=[]
                g_minus_e_Q=[]

                g_plus_e_I_wo_fogi_rx=[]
                g_plus_e_Q_wo_fogi_rx=[]
                g_minus_e_I_wo_fogi_rx=[]
                g_minus_e_Q_wo_fogi_rx=[]

                for _ in  tqdm(range(repetition)):
                    for state in ["0+1_i", "0+1_q", "0-1_i", "0-1_q",
                                  "0+1_i_wo_fogi_rx", "0+1_q_wo_fogi_rx", "0-1_i_wo_fogi_rx", "0-1_q_wo_fogi_rx"]:
                        awg1.flush_waveform()
                        awg2.flush_waveform()
                        awg3.flush_waveform()
                        if state=="0+1_i":
                            seq = comm_sequence(delay=t, half_pi=True, JPA_direction=1, fogi_rx=True)
                            load_sequence_w_append(seq, append_ports=[fogi_port_tx, fogi_port_rx], waveforms_added=[control_pulse_tx, control_pulse_rx], cycles=cycles)
                            # seq.draw()
                            # raise SystemError
                        if state=="0+1_q":
                            seq = comm_sequence(delay=t, half_pi=True, JPA_direction=-1, fogi_rx=True)
                            load_sequence_w_append(seq, append_ports=[fogi_port_tx, fogi_port_rx], waveforms_added=[control_pulse_tx, control_pulse_rx], cycles=cycles)
                        if state=="0-1_i":
                            seq = comm_sequence(delay=t, half_pi=False, JPA_direction=1, fogi_rx=True)
                            load_sequence_w_append(seq, append_ports=[fogi_port_tx, fogi_port_rx], waveforms_added=[control_pulse_tx, control_pulse_rx], cycles=cycles)
                        if state=="0-1_q":
                            seq = comm_sequence(delay=t, half_pi=False, JPA_direction=-1, fogi_rx=True)
                            load_sequence_w_append(seq, append_ports=[fogi_port_tx, fogi_port_rx], waveforms_added=[control_pulse_tx, control_pulse_rx], cycles=cycles)
                        if state=="0+1_i_wo_fogi_rx":
                            seq = comm_sequence(delay=t, half_pi=True, fogi_rx=False, JPA_direction=1)
                            load_sequence_w_append(seq, append_ports=[fogi_port_tx], waveforms_added=[control_pulse_tx], cycles=cycles)
                        if state=="0+1_q_wo_fogi_rx":
                            seq = comm_sequence(delay=t, half_pi=True, fogi_rx=False, JPA_direction=-1)
                            load_sequence_w_append(seq, append_ports=[fogi_port_tx], waveforms_added=[control_pulse_tx], cycles=cycles)
                        if state=="0-1_i_wo_fogi_rx":
                            seq = comm_sequence(delay=t, half_pi=False, fogi_rx=False, JPA_direction=1)
                            load_sequence_w_append(seq, append_ports=[fogi_port_tx], waveforms_added=[control_pulse_tx], cycles=cycles)
                        if state=="0-1_q_wo_fogi_rx":
                            seq = comm_sequence(delay=t, half_pi=False, fogi_rx=False, JPA_direction=-1)
                            load_sequence_w_append(seq, append_ports=[fogi_port_tx], waveforms_added=[control_pulse_tx], cycles=cycles)
                        
                        data = run(seq, which=measure_which[:2]).mean(axis=0) * voltage_step_tx
                        if state=="0+1_i": g_plus_e_I = np.append(g_plus_e_I, data)
                        if state=="0+1_q": g_plus_e_Q = np.append(g_plus_e_Q, data)
                        if state=="0-1_i": g_minus_e_I = np.append(g_minus_e_I, data)
                        if state=="0-1_q": g_minus_e_Q = np.append(g_minus_e_Q, data)
                        if state=="0+1_i_wo_fogi_rx": g_plus_e_I_wo_fogi_rx = np.append(g_plus_e_I_wo_fogi_rx, data)
                        if state=="0+1_q_wo_fogi_rx": g_plus_e_Q_wo_fogi_rx = np.append(g_plus_e_Q_wo_fogi_rx, data)
                        if state=="0-1_i_wo_fogi_rx": g_minus_e_I_wo_fogi_rx = np.append(g_minus_e_I_wo_fogi_rx, data)
                        if state=="0-1_q_wo_fogi_rx": g_minus_e_Q_wo_fogi_rx = np.append(g_minus_e_Q_wo_fogi_rx, data)
                g_plus_e_I = g_plus_e_I.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
                g_plus_e_Q = g_plus_e_Q.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
                g_minus_e_I = g_minus_e_I.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
                g_minus_e_Q = g_minus_e_Q.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
                waveform_I = (g_plus_e_I - g_minus_e_I)/2
                waveform_Q = (g_plus_e_Q - g_minus_e_Q)/2
                waveform = (waveform_I + waveform_Q) / 2

                g_plus_e_I_wo_fogi_rx = g_plus_e_I_wo_fogi_rx.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
                g_plus_e_Q_wo_fogi_rx = g_plus_e_Q_wo_fogi_rx.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
                g_minus_e_I_wo_fogi_rx = g_minus_e_I_wo_fogi_rx.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
                g_minus_e_Q_wo_fogi_rx = g_minus_e_Q_wo_fogi_rx.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
                waveform_I_wo_fogi_rx = (g_plus_e_I_wo_fogi_rx - g_minus_e_I_wo_fogi_rx)/2
                waveform_Q_wo_fogi_rx = (g_plus_e_Q_wo_fogi_rx - g_minus_e_Q_wo_fogi_rx)/2
                waveform_wo_fogi_rx = (waveform_I_wo_fogi_rx + waveform_Q_wo_fogi_rx) / 2
                
                writer.add_data(
                    time=np.arange(len(waveform)) * dig_ch_tx.sampling_interval(),
                    delay=t-200,
                    waveform=waveform,
                    waveform_wo_fogi_rx=waveform_wo_fogi_rx
                )
    finally:
        off()
        print('finished')
