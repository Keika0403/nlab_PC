from setup_td_tomography import *

measurement_name = os.path.basename(__file__)[:-3]
tags.append("single_shot")

shot_count = 40000
repetition = 1
hvi_trigger.trigger_period(300000)
hvi_trigger.digitizer_delay(400)

qubit_drive_port = qubit_drive_port_rx
fogi_port = fogi_port_rx
readout_port = readout_port_rx
readout_single_shot_seq = readout_ss_seq_rx
JPA_port = JPA_port_rx
dig_port = dig_port_rx
ge_pi_seq = ge_pi_seq_rx
voltage_step = voltage_step_rx
mean_g = mean_g_rx
mean_e = mean_e_rx
ein_vec = ein_vec_rx

sequence_g = Sequence(port_list=[qubit_drive_port, fogi_port, readout_port, JPA_port, dig_port])
sequence_g.add(Delay(48), qubit_drive_port)
sequence_g.trigger([qubit_drive_port, readout_port, JPA_port, dig_port])
sequence_g.call(readout_single_shot_seq)
sequence_g.trigger([readout_port, JPA_port, dig_port])
sequence_g.call(readout_single_shot_seq)
# sequence_g.draw()

sequence_e = Sequence(port_list=[qubit_drive_port, fogi_port, readout_port, JPA_port, dig_port])
sequence_e.call(ge_pi_seq)
sequence_e.trigger([qubit_drive_port, readout_port, JPA_port, dig_port])
sequence_e.call(readout_single_shot_seq)
sequence_e.trigger([readout_port, JPA_port, dig_port])
sequence_e.call(readout_single_shot_seq)
# sequence_e.draw()

# raise SystemError

data = DataDict(
    signal_g=dict(),
    signal_e=dict(),
)
data.validate()

try:
    with DDH5Writer(data, data_path, name=measurement_name) as writer:
        writer.add_tag(tags)
        writer.backup_file([__file__, setup_parameters_file, setup_file])
        writer.save_text("wiring.md", wiring)
        writer.save_dict("station_snapshot.json", station.snapshot())
        result_g = []
        result_e = []
        for _ in tqdm(range(repetition)):
            load_sequence(sequence_g, cycles=shot_count)
            pulse_g = run(sequence_g, which=measure_which[:2]) * voltage_step
            result_g.append(pulse_g)
            post_selection_start_g = int(dig_port.measurement_windows[1][0] // 2 - dig_port.measurement_windows[0][0] // 2)
            
            load_sequence(sequence_e, cycles=shot_count)
            pulse_e = run(sequence_e, which=measure_which[:2]) * voltage_step
            result_e.append(pulse_e)
            post_selection_start_e = int(dig_port.measurement_windows[1][0] // 2 - dig_port.measurement_windows[0][0] // 2)
        result_g = np.array(result_g).reshape(int(repetition*shot_count), len(pulse_g[0]))
        result_e = np.array(result_e).reshape(int(repetition*shot_count), len(pulse_e[0]))
        # plt.plot(result_g.mean(axis=0))
        # plt.plot(result_e.mean(axis=0))
        # plt.plot(mean_g)
        # plt.show()
        delete_idx_g = []
        delete_idx_e = []
        for i in range(len(result_g)):
            single_shot_g = result_g[i][0:len(mean_g)]
            single_shot_e = result_e[i][0:len(mean_e)]
            if np.dot(single_shot_g - mean_g, ein_vec) > 1/2: # eliminate e state
                delete_idx_g.append(i)
            if np.dot(single_shot_e - mean_g, ein_vec) < 1/2: # eliminate g state
                delete_idx_e.append(i)
        result_g = np.delete(result_g, delete_idx_g, axis=0)[:, post_selection_start_g:post_selection_start_g+len(mean_g)]
        result_e = np.delete(result_e, delete_idx_e, axis=0)[:, post_selection_start_e:post_selection_start_e+len(mean_e)]
        # s11_g = demodulate(result_g)
        # s11_e = demodulate(result_e)
        signal_g = np.dot(result_g - mean_g, ein_vec)
        signal_e = np.dot(result_e - mean_g, ein_vec)
        writer.add_data(
            signal_g=signal_g,
            signal_e=signal_e,
        )
finally:
    off()
    print("finished")
    