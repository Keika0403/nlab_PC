from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]
tags.append("single_shot")


shot_count = 40000
repetition = 1
hvi_trigger.trigger_period(300000)
hvi_trigger.digitizer_delay(400)

if measure_which == "tx":
    sequence_g = Sequence(port_list=ports_tx)
    sequence_g.call(readout_ss_seq_tx)
    sequence_g.trigger(ports_tx)
    sequence_g.add(Delay(ge_pi_pulse_tx.params["duration"]), qubit_drive_port_tx)
    sequence_g.trigger(ports_tx)
    sequence_g.call(readout_ss_seq_tx)
    # sequence_g.draw()
    # raise SystemError

    sequence_e = Sequence(port_list=ports_tx)
    sequence_e.call(readout_ss_seq_tx)
    sequence_e.trigger(ports_tx)
    sequence_e.call(ge_pi_seq_tx)
    sequence_e.trigger(ports_tx)
    sequence_e.call(readout_ss_seq_tx)
    # sequence_e.draw()
elif measure_which == "txQrx":
    sequence_g = Sequence(port_list=ports_txQrx)
    sequence_g.call(readout_ss_seq_tx)
    sequence_g.trigger(ports_txQrx)
    sequence_g.add(Delay(ge_pi_pulse_rx.params["duration"]), qubit_drive_port_rx)
    sequence_g.trigger(ports_txQrx)
    sequence_g.call(readout_ss_seq_tx)
    # sequence_g.draw()

    sequence_e = Sequence(port_list=ports_txQrx)
    sequence_e.call(readout_ss_seq_tx)
    sequence_e.trigger(ports_txQrx)
    sequence_e.call(ge_pi_seq_rx)
    sequence_e.trigger(ports_txQrx)
    sequence_e.call(readout_ss_seq_tx)
    # sequence_e.draw()
elif measure_which == "rx":
    sequence_g = Sequence(port_list=ports_rx)
    sequence_g.call(readout_ss_seq_rx)
    sequence_g.trigger(ports_rx)
    sequence_g.add(Delay(ge_pi_pulse_rx.params["duration"]), qubit_drive_port_rx)
    sequence_g.trigger(ports_rx)
    sequence_g.call(readout_ss_seq_rx)
    # sequence_g.draw()

    sequence_e = Sequence(port_list=ports_rx)
    sequence_e.call(readout_ss_seq_rx)
    sequence_e.trigger(ports_rx)
    sequence_e.call(ge_pi_seq_rx)
    sequence_e.trigger(ports_rx)
    sequence_e.call(readout_ss_seq_rx)


# raise SystemError
data = DataDict(
    signal_g=dict(),
    signal_e=dict(),
)
data.validate()

try:
    with DDH5Writer(data, data_path, name=measurement_name) as writer:
        writer.add_tag(tags)
        writer.backup_file([__file__, setup_file])
        writer.save_text("wiring.md", wiring)
        writer.save_dict("station_snapshot.json", station.snapshot())
        result_g = []
        result_e = []
        if measure_which[:2] == "tx":
            for _ in tqdm(range(repetition)):
                load_sequence(sequence_g, cycles=shot_count)
                pulse_g = run(sequence_g, which=measure_which[:2]) * voltage_step_tx
                result_g.append(pulse_g)
                post_selection_start_g = int(dig_port_tx.measurement_windows[1][0] // 2 )
                
                load_sequence(sequence_e, cycles=shot_count)
                pulse_e = run(sequence_e, which=measure_which[:2]) * voltage_step_tx
                result_e.append(pulse_e)
                post_selection_start_e = int(dig_port_tx.measurement_windows[1][0] // 2 )
            result_g = np.array(result_g).reshape(int(repetition*shot_count), len(pulse_g[0]))
            result_e = np.array(result_e).reshape(int(repetition*shot_count), len(pulse_e[0]))
            # plt.plot(result_g.mean(axis=0))
            # plt.plot(result_e.mean(axis=0))
            # plt.show()
            delete_idx_g = []
            delete_idx_e = []
            # post_selection_start_g = int(sequence_g.get_waveform_information()[dig_port_tx.name]['measurement_windows'][1][0] // 2 )
            # post_selection_start_e = int(sequence_e.get_waveform_information()[dig_port_tx.name]['measurement_windows'][1][0] // 2 )
            for i in range(len(result_g)):
                single_shot_g = result_g[i][0:len(mean_g_tx)]
                single_shot_e = result_e[i][0:len(mean_e_tx)]
                if np.dot(single_shot_g - mean_g_tx, ein_vec_tx) > 1/2: # eliminate e state
                    delete_idx_g.append(i)
                if np.dot(single_shot_e - mean_g_tx, ein_vec_tx) > 1/2: # eliminate e state
                    delete_idx_e.append(i)
            result_g = np.delete(result_g, delete_idx_g, axis=0)[:, post_selection_start_g:post_selection_start_g+len(mean_g_tx)]
            result_e = np.delete(result_e, delete_idx_e, axis=0)[:, post_selection_start_e:post_selection_start_e+len(mean_e_tx)]
            print(result_g.shape, result_e.shape)
            # plt.plot(result_g.mean(axis=0))
            # plt.plot(result_e.mean(axis=0))
            # plt.show()
            # raise SystemError
            # s11_g = demodulate(result_g)
            # s11_e = demodulate(result_e)
            signal_g = np.dot(result_g - mean_g_tx, ein_vec_tx)
            signal_e = np.dot(result_e - mean_g_tx, ein_vec_tx)
            writer.add_data(
                signal_g=signal_g,
                signal_e=signal_e,
            )
        elif measure_which[:2] == "rx":
            for _ in tqdm(range(repetition)):
                load_sequence(sequence_g, cycles=shot_count)
                pulse_g = run(sequence_g, which=measure_which[:2]) * voltage_step_rx
                result_g.append(pulse_g)
                post_selection_start_g = int(dig_port_rx.measurement_windows[1][0] // 2 )
                
                load_sequence(sequence_e, cycles=shot_count)
                pulse_e = run(sequence_e, which=measure_which[:2]) * voltage_step_rx
                result_e.append(pulse_e)
                post_selection_start_e = int(dig_port_rx.measurement_windows[1][0] // 2 )
            result_g = np.array(result_g).reshape(int(repetition*shot_count), len(pulse_g[0]))
            result_e = np.array(result_e).reshape(int(repetition*shot_count), len(pulse_e[0]))
            # plt.plot(result_g.mean(axis=0))
            # plt.plot(result_e.mean(axis=0))
            # plt.show()
            delete_idx_g = []
            delete_idx_e = []
            # post_selection_start_g = int(sequence_g.get_waveform_information()[dig_port_rx.name]['measurement_windows'][1][0] // 2 )
            # post_selection_start_e = int(sequence_e.get_waveform_information()[dig_port_rx.name]['measurement_windows'][1][0] // 2 )
            for i in range(len(result_g)):
                single_shot_g = result_g[i][0:len(mean_g_rx)]
                single_shot_e = result_e[i][0:len(mean_e_rx)]
                if np.dot(single_shot_g - mean_g_rx, ein_vec_rx) > 1/2: # eliminate e state
                    delete_idx_g.append(i)
                if np.dot(single_shot_e - mean_g_rx, ein_vec_rx) > 1/2: # eliminate e state
                    delete_idx_e.append(i)
            result_g = np.delete(result_g, delete_idx_g, axis=0)[:, post_selection_start_g:post_selection_start_g+len(mean_g_rx)]
            result_e = np.delete(result_e, delete_idx_e, axis=0)[:, post_selection_start_e:post_selection_start_e+len(mean_e_rx)]
            print(result_g.shape, result_e.shape)
            # plt.plot(result_g.mean(axis=0))
            # plt.plot(result_e.mean(axis=0))
            # plt.show()
            # raise SystemError
            # s11_g = demodulate(result_g)
            # s11_e = demodulate(result_e)
            signal_g = np.dot(result_g - mean_g_rx, ein_vec_rx)
            signal_e = np.dot(result_e - mean_g_rx, ein_vec_rx)
            writer.add_data(
                signal_g=signal_g,
                signal_e=signal_e,
            )
finally:
    off()
    print("finished")
    