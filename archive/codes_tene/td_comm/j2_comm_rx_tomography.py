from setup_td import *
from setup_td_tomography import *
from time_reverse import *

measurement_name = os.path.basename(__file__)[:-3]
tags.append("single_shot")

cd = "CDY184"
cycles = 50000
repetition = 5
hvi_trigger.digitizer_delay(400)
hvi_trigger.trigger_period(100000)
fogi_delay = 30

pulse_dict_tx = dict(
    data0 = dict(target_freq=9.38, date="2025-08-12", acquire_time="124708"),
    # data1 = dict(target_freq=9.38, date="2025-08-12", acquire_time="123940"),
    # data2 = dict(target_freq=9.38, date="2025-08-12", acquire_time="123942"),
    # data3 = dict(target_freq=9.38, date="2025-08-12", acquire_time="124713"),
)
pulse_dict_rx = dict(
    data0 = dict(target_freq=9.38, date="2025-08-12", acquire_time="111523"),
    # data1 = dict(target_freq=9.38, date="2025-08-12", acquire_time="111525"),
    # data2 = dict(target_freq=9.38, date="2025-08-12", acquire_time="111526"),
    # data3 = dict(target_freq=9.38, date="2025-08-12", acquire_time="111528"),
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

        def comm_sequence(state) -> Sequence:
            seq = Sequence(port_list=ports)
            if state=="0+1":
                seq.add(VirtualZ(0), qubit_drive_port_tx, copy=False)
                seq.call(ge_half_pi_seq_tx)
                seq.add(VirtualZ(0), qubit_drive_port_tx, copy=False)
            if state=="0+i":
                seq.add(VirtualZ(np.pi/2), qubit_drive_port_tx, copy=False)
                seq.call(ge_half_pi_seq_tx)
                seq.add(VirtualZ(-np.pi/2), qubit_drive_port_tx, copy=False)
            if state=="0-1":
                seq.add(VirtualZ(np.pi), qubit_drive_port_tx, copy=False)
                seq.call(ge_half_pi_seq_tx)
                seq.add(VirtualZ(-np.pi), qubit_drive_port_tx, copy=False)
            if state=="0-i":
                seq.add(VirtualZ(-np.pi/2), qubit_drive_port_tx, copy=False)
                seq.call(ge_half_pi_seq_tx)
                seq.add(VirtualZ(np.pi/2), qubit_drive_port_tx, copy=False)
            seq.trigger(ports)
            seq.call(ef_pi_seq_tx)
            seq.trigger(ports)
            seq.add(Acquire(len(fogi_pulse_tx)), fogi_port_tx)
            seq.add(Delay(fogi_delay), fogi_port_rx)
            seq.add(Acquire(len(fogi_pulse_rx)), fogi_port_rx)
            seq.trigger(ports)
            seq.call(ef_pi_seq_rx)
            return seq

        def sequence_x(state) -> Sequence:
            seq = Sequence(port_list=ports)
            seq.call(readout_single_shot_seq)
            seq.trigger(ports)
            # seq.call(comm_sequence(state))
            seq.trigger(ports)
            seq.add(Delay(0), qubit_drive_port_rx)
            seq.trigger(ports)
            seq.add(VirtualZ(0), qubit_drive_port_rx, copy=False)
            seq.trigger(ports)
            seq.call(ge_half_pi_seq_rx)
            seq.trigger(ports)
            seq.call(readout_single_shot_seq)
            return seq

        # sequence_x.draw()
        # raise SystemError

        def sequence_y(state) -> Sequence:
            seq = Sequence(port_list=ports)
            seq.call(readout_single_shot_seq)
            seq.trigger(ports)
            # seq.call(comm_sequence(state))
            seq.trigger(ports)
            seq.add(Delay(0), qubit_drive_port_rx)
            seq.trigger(ports)
            seq.add(VirtualZ(np.pi/2), qubit_drive_port_rx, copy=False)
            seq.trigger(ports)
            seq.call(ge_half_pi_seq_rx)
            seq.trigger(ports)
            seq.call(readout_single_shot_seq)
            return seq

        def sequence_z(state) -> Sequence:
            seq = Sequence(port_list=ports)
            seq.call(readout_single_shot_seq)
            seq.trigger(ports)
            # seq.call(comm_sequence(state))
            seq.trigger(ports)
            seq.add(Delay(30), qubit_drive_port_rx)
            seq.trigger(ports)
            seq.call(readout_single_shot_seq)
            return seq

        data = DataDict(
            signal_x_0=dict(),
            signal_y_0=dict(),
            signal_z_0=dict(),
            signal_x_pi_2=dict(),
            signal_y_pi_2=dict(),
            signal_z_pi_2=dict(),
            signal_x_pi=dict(),
            signal_y_pi=dict(),
            signal_z_pi=dict(),
            signal_x_pi_2_mi=dict(),
            signal_y_pi_2_mi=dict(),
            signal_z_pi_2_mi=dict(),
        )
        data.validate()

        try:
            with DDH5Writer(data, data_path, name=measurement_name) as writer:
                writer.add_tag(tags)
                writer.backup_file([__file__, setup_file])
                writer.save_text("wiring.md", wiring)
                writer.save_dict("station_snapshot.json", station.snapshot())
                signal_xs = []
                signal_ys = []
                signal_zs = []
                for state in ["0+1", "0+i", "0-1", "0-i"]:
                # for state in [ "0-i",  "0-i",  "0-i", "0-i"]:
                    result_x = []
                    result_y = []
                    result_z = []
                    seq_x = sequence_x(state)
                    seq_y = sequence_y(state)
                    seq_z = sequence_z(state)
                    seq_x.draw()
                    seq_y.draw()
                    seq_z.draw()
                    raise SystemError
                    for _ in tqdm(range(repetition)):
                        load_sequence_w_append_comm(seq_x, append_ports=fogi_ports, waveforms_appended=fogi_pulses, cycles=cycles)
                        # seq_x.draw()
                        pulse_x = run_comm(seq_x, plot=0) * voltage_step_comm
                        result_x.append(pulse_x)
                        post_selection_start_x = int(dig_port.measurement_windows[1][0] // 2 )
                        
                        load_sequence_w_append_comm(seq_y, append_ports=fogi_ports, waveforms_appended=fogi_pulses, cycles=cycles)
                        # seq_y.draw()
                        pulse_y = run_comm(seq_y, plot=0) * voltage_step_comm
                        result_y.append(pulse_y)
                        post_selection_start_y = int(dig_port.measurement_windows[1][0] // 2 )

                        load_sequence_w_append_comm(seq_z, append_ports=fogi_ports, waveforms_appended=fogi_pulses, cycles=cycles)
                        # seq_z.draw()
                        pulse_z = run_comm(seq_z, plot=0) * voltage_step_comm
                        result_z.append(pulse_z)
                        post_selection_start_z = int(dig_port.measurement_windows[1][0] // 2 )
                    result_x = np.array(result_x).reshape(int(repetition*cycles), len(pulse_x[0]))
                    result_y = np.array(result_y).reshape(int(repetition*cycles), len(pulse_y[0]))
                    result_z = np.array(result_z).reshape(int(repetition*cycles), len(pulse_z[0]))
                    # plt.plot(result_z.mean(axis=0))

                    ##### anlyzation
                    delete_idx_x = []
                    delete_idx_y = []
                    delete_idx_z = []
                    for i in range(len(result_z)):
                        single_shot_x = result_x[i][0:len(mean_g)]
                        single_shot_y = result_y[i][0:len(mean_g)]
                        single_shot_z = result_z[i][0:len(mean_g)]
                        if np.dot(single_shot_x - mean_g, ein_vec) > 1/2: # eliminate e state
                            delete_idx_x.append(i)
                        if np.dot(single_shot_y - mean_g, ein_vec) > 1/2: # eliminate e state
                            delete_idx_y.append(i)
                        if np.dot(single_shot_z - mean_g, ein_vec) > 1/2: # eliminate e state
                            delete_idx_z.append(i)
                    result_x = np.delete(result_x, delete_idx_x, axis=0)[:, post_selection_start_x:post_selection_start_x+len(mean_g)]
                    result_y = np.delete(result_y, delete_idx_y, axis=0)[:, post_selection_start_y:post_selection_start_y+len(mean_g)]
                    result_z = np.delete(result_z, delete_idx_z, axis=0)[:, post_selection_start_z:post_selection_start_z+len(mean_g)]
                    # print(result_x.shape, result_y.shape)
                    # print(result_z.shape)
                    # plt.plot(result_z.mean(axis=0))
                    # plt.show()
                    signal_x = np.dot(result_x - mean_g, ein_vec)
                    signal_y = np.dot(result_y - mean_g, ein_vec)
                    signal_z = np.dot(result_z - mean_g, ein_vec)
                    print(signal_x.shape, signal_y.shape, signal_z.shape)
                    signal_xs.append(signal_x)
                    signal_ys.append(signal_y)
                    signal_zs.append(signal_z)
                    
                writer.add_data(
                    signal_x_0=signal_xs[0],
                    signal_y_0=signal_ys[0],
                    signal_z_0=signal_zs[0],
                    signal_x_pi_2=signal_xs[1],
                    signal_y_pi_2=signal_ys[1],
                    signal_z_pi_2=signal_zs[1],
                    signal_x_pi=signal_xs[2],
                    signal_y_pi=signal_ys[2],
                    signal_z_pi=signal_zs[2],
                    signal_x_pi_2_mi=signal_xs[3],
                    signal_y_pi_2_mi=signal_ys[3],
                    signal_z_pi_2_mi=signal_zs[3],
                )
        finally:
            off_comm()
            print("finished")
            