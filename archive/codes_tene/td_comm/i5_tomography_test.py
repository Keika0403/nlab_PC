from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]
tags.append("single_shot")

shot_count = 50000
repetition = 10
hvi_trigger.trigger_period(100000)

sequence_x = Sequence(port_list=[qubit_drive_port, readout_port, JPA_port, dig_port])
sequence_x.call(readout_single_shot_seq)
sequence_x.trigger([qubit_drive_port, readout_port, JPA_port, dig_port])
sequence_x.call(ge_half_pi_seq)
sequence_x.trigger([qubit_drive_port, readout_port, JPA_port, dig_port])
sequence_x.add(Delay(30), qubit_drive_port)
sequence_x.trigger([qubit_drive_port, readout_port, JPA_port, dig_port])
sequence_x.add(ResetPhase(np.pi), qubit_drive_port, copy=False)
sequence_x.call(ge_half_pi_seq)
sequence_x.call(readout_single_shot_seq)

sequence_y = Sequence(port_list=[qubit_drive_port, readout_port, JPA_port, dig_port])
sequence_y.call(readout_single_shot_seq)
sequence_y.trigger([qubit_drive_port, readout_port, JPA_port, dig_port])
sequence_y.call(ge_half_pi_seq)
sequence_y.trigger([qubit_drive_port, readout_port, JPA_port, dig_port])
sequence_y.add(Delay(30), qubit_drive_port)
sequence_y.trigger([qubit_drive_port, readout_port, JPA_port, dig_port])
sequence_y.add(ResetPhase(np.pi/2), qubit_drive_port, copy=False)
sequence_y.call(ge_half_pi_seq)
sequence_y.call(readout_single_shot_seq)

sequence_z = Sequence(port_list=[qubit_drive_port, readout_port, JPA_port, dig_port])
sequence_z.call(readout_single_shot_seq)
sequence_z.trigger([qubit_drive_port, readout_port, JPA_port, dig_port])
sequence_z.call(ge_half_pi_seq)
sequence_z.trigger([qubit_drive_port, readout_port, JPA_port, dig_port])
sequence_z.add(Delay(60), qubit_drive_port)
sequence_z.trigger([qubit_drive_port, readout_port, JPA_port, dig_port])
sequence_z.call(readout_single_shot_seq)

# sequence_z.draw()
# raise SystemError
data = DataDict(
    signal_x=dict(),
    signal_y=dict(),
    signal_z=dict(),

)
data.validate()

try:
    with DDH5Writer(data, data_path, name=measurement_name) as writer:
        writer.add_tag(tags)
        writer.backup_file([__file__, setup_file])
        writer.save_text("wiring.md", wiring)
        writer.save_dict("station_snapshot.json", station.snapshot())
        result_x = []
        result_y = []
        result_z = []
        for _ in tqdm(range(repetition)):
            load_sequence(sequence_x, cycles=shot_count)
            pulse_x = run(sequence_x, plot=0) * voltage_step
            result_x.append(pulse_x)
            post_selection_start_x = int(dig_port.measurement_windows[1][0] // 2 )
            
            load_sequence(sequence_y, cycles=shot_count)
            pulse_y = run(sequence_y, plot=0) * voltage_step
            result_y.append(pulse_y)
            post_selection_start_y = int(dig_port.measurement_windows[1][0] // 2 )

            load_sequence(sequence_z, cycles=shot_count)
            pulse_z = run(sequence_z, plot=0) * voltage_step
            result_z.append(pulse_z)
            post_selection_start_z = int(dig_port.measurement_windows[1][0] // 2 )
        result_x = np.array(result_x).reshape(int(repetition*shot_count), len(pulse_x[0]))
        result_y = np.array(result_y).reshape(int(repetition*shot_count), len(pulse_y[0]))
        result_z = np.array(result_z).reshape(int(repetition*shot_count), len(pulse_z[0]))
        # plt.plot(result_z.mean(axis=0))
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
        writer.add_data(
            signal_x=signal_x,
            signal_y=signal_y,
            signal_z=signal_z,
        )
finally:
    off()
    print("finished")
    