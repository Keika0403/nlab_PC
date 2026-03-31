from setup_td import *
# from TD_photon_calibration import calibrate_shape
from itertools import product

measurement_name = os.path.basename(__file__)[:-3]

# photon_frequency = 9.35
delay_relative= 18
assert measure_which == "both"
cd = "CDK184"

cycles = 10000
repetition = 1
hvi_trigger.digitizer_delay(400)
hvi_trigger.trigger_period(200000)
lo1.power(24)
lo1.frequency(readout_lo_freq_tx*1e9)
JPA_port_tx.if_freq = (readout_freq_tx - readout_lo_freq_tx) * 2


# _, datadict = search_datadict_miyamura("D:\\miyamura\\result\\control_pulses", 
#                                         "2024-08-12", acquire_time="113309")
# control_pulse_tx = datadict["control_pulse"]["values"].ravel()

# _, datadict = search_datadict_miyamura("D:\\miyamura\\result\\control_pulses", 
#                                         "2024-08-12", acquire_time="113318")
# control_pulse_rx = datadict["control_pulse"]["values"].ravel()

# control_pulse_tx, photon_tx = calibrate_shape(control_pulse_tx, "tx", photon_frequency)
# control_pulse_rx, photon_rx = calibrate_shape(control_pulse_rx, "txQrx", photon_frequency, reverse=True)

pulse_dict_tx = dict(
    data0 = dict(target_freq=9.38, date="2025-08-12", acquire_time="124708"),
    data1 = dict(target_freq=9.38, date="2025-08-12", acquire_time="123940"),
    data2 = dict(target_freq=9.38, date="2025-08-12", acquire_time="123942"),
    data3 = dict(target_freq=9.38, date="2025-08-12", acquire_time="124713"),
)
pulse_dict_rx = dict(
    data0 = dict(target_freq=9.38, date="2025-08-12", acquire_time="111523"),
    data1 = dict(target_freq=9.38, date="2025-08-12", acquire_time="111525"),
    data2 = dict(target_freq=9.38, date="2025-08-12", acquire_time="111526"),
    data3 = dict(target_freq=9.38, date="2025-08-12", acquire_time="111528"),
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


    def exp(prepare_seq:Sequence, state:str):
        def entangle_sequence(delay_relative:int,) -> Sequence:
            if delay_relative > 0 or ge_pi_pulse_tx.params["duration"] + ef_half_pi_pulse_tx.params["duration"] + delay_relative >= 0:
                delay = delay_relative + ge_pi_pulse_tx.params["duration"] + ef_half_pi_pulse_tx.params["duration"]
                assert delay%2 == 0
                seq = Sequence(port_list=ports)
                seq.call(prepare_seq)
                seq.call(ef_half_pi_seq_tx)
                seq.trigger(ports_tx)
                seq.add(Acquire(len(control_pulse_tx)), fogi_port_tx)
                seq.add(Delay(delay), qubit_drive_port_rx)
                seq.trigger(ports_rx)
                seq.add(Acquire(len(control_pulse_rx)), fogi_port_rx)
                seq.trigger(ports)
                seq.call(ef_pi_seq_rx)
            elif ge_pi_pulse_tx.params["duration"] + ef_half_pi_pulse_tx.params["duration"] + delay_relative < 0:
                delay = -ge_pi_pulse_tx.params["duration"] - ef_half_pi_pulse_tx.params["duration"] - delay_relative
                assert delay%2 == 0 and delay>0, f"{delay}"
                seq = Sequence(port_list=ports)
                seq.add(Delay(delay), qubit_drive_port_tx)
                seq.call(prepare_seq)
                seq.call(ef_half_pi_seq_tx)
                seq.trigger(ports_tx)
                seq.add(Acquire(len(control_pulse_tx)), fogi_port_tx)
                seq.trigger(ports_rx)
                seq.add(Acquire(len(control_pulse_rx)), fogi_port_rx)
                seq.trigger(ports)
                seq.call(ef_pi_seq_rx)
            else: AssertionError, f"{delay_relative}"
            return seq

        def seq_qst(delay_relative:int, POVM:str)->Sequence:
            sequence = Sequence(ports)
            sequence.call(readout_ss_seq_tx)
            sequence.call(readout_ss_seq_rx)
            sequence.trigger(ports)
            sequence.call(entangle_sequence(delay_relative))
            sequence.trigger(ports)
            if POVM == "g":
                sequence.add(Delay(ge_pi_pulse_rx.params["duration"]), qubit_drive_port_rx)
                sequence.add(Delay(ef_pi_pulse_rx.params["duration"]), qubit_drive_port_rx)
            elif POVM == "e":
                sequence.add(Delay(ef_pi_pulse_rx.params["duration"]), qubit_drive_port_rx)
                sequence.call(ge_pi_seq_rx)
            elif POVM == "f":
                sequence.call(ef_pi_seq_rx)
                sequence.call(ge_pi_seq_rx)
            elif POVM == "ige":
                sequence.add(Delay(ef_pi_pulse_rx.params["duration"]), qubit_drive_port_rx)
                sequence.add(VirtualZ(-np.pi/2), qubit_drive_port_rx)
                sequence.call(ge_half_pi_seq_rx)
                sequence.add(VirtualZ(np.pi/2), qubit_drive_port_rx)
            elif POVM == "+ge":
                sequence.add(Delay(ef_pi_pulse_rx.params["duration"]), qubit_drive_port_rx)
                sequence.add(VirtualZ(np.pi), qubit_drive_port_rx)
                sequence.call(ge_half_pi_seq_rx)
                sequence.add(VirtualZ(-np.pi), qubit_drive_port_rx)
            elif POVM == "+gf":
                sequence.add(VirtualZ(np.pi), qubit_drive_port_rx)
                sequence.call(ef_pi_seq_rx)
                sequence.add(VirtualZ(np.pi), qubit_drive_port_rx)
                sequence.add(VirtualZ(np.pi), qubit_drive_port_rx)
                sequence.call(ge_half_pi_seq_rx)
                sequence.add(VirtualZ(-np.pi), qubit_drive_port_rx)
            elif POVM == "igf":
                sequence.add(VirtualZ(np.pi), qubit_drive_port_rx)
                sequence.call(ef_pi_seq_rx)
                sequence.add(VirtualZ(np.pi), qubit_drive_port_rx)
                sequence.add(VirtualZ(-np.pi/2), qubit_drive_port_rx)
                sequence.call(ge_half_pi_seq_rx)
                sequence.add(VirtualZ(np.pi/2), qubit_drive_port_rx)
            elif POVM == "+ef":
                sequence.add(VirtualZ(np.pi), qubit_drive_port_rx)
                sequence.call(ef_half_pi_seq_rx)
                sequence.add(VirtualZ(-np.pi), qubit_drive_port_rx)
                sequence.call(ge_pi_seq_rx)
            elif POVM == "ief":
                sequence.add(VirtualZ(-np.pi/2), qubit_drive_port_rx)
                sequence.call(ef_half_pi_seq_rx)
                sequence.add(VirtualZ(np.pi/2), qubit_drive_port_rx)
                sequence.call(ge_pi_seq_rx)
            else:raise NotImplementedError
            sequence.trigger(ports)
            sequence.call(readout_ss_seq_rx)
            return sequence

        # seq_qst(delay_relative, "e").draw()
        # print(dig_port_rx.measurement_windows)

        # raise SystemError

        data = DataDict(
            label=dict(),
            signal=dict(axes=["label"])
        )
        data.validate()

        try:
            with DDH5Writer(data, data_path, name=state + "_" + measurement_name) as writer:
                writer.add_tag(tags)
                writer.backup_file([__file__, setup_file, setup_parameters_file])
                writer.save_text("wiring.md", wiring)
                writer.save_dict("station_snapshot.json", station.snapshot())
                for i, p in tqdm(enumerate(["g", "+ge", "ige", "e", "+gf", "igf", "f", "+ef", "ief"])):
                    result=[]
                    result_tx, result_rx = [], []
                    sequence = seq_qst(delay_relative, POVM=p)
                    for _ in  range(repetition):
                        # sequence.draw()
                        load_sequence_w_append(sequence, append_ports=[fogi_port_tx, fogi_port_rx], waveforms_added=[control_pulse_tx, control_pulse_rx], cycles=cycles, check=False)
                        tx, rx = run(sequence, which=measure_which, lo_on=False)
                        result_tx = np.append(result_tx, tx)
                        result_rx = np.append(result_rx, rx)
                        # result = np.append(result, run(sequence, which=measure_which, lo_on=False) * voltage_step_tx)
                    # result = result.reshape(int(repetition*cycles), int(dig_ch_tx.points_per_cycle()))
                    sequence.compile()
                    result_tx = result_tx.reshape(int(repetition*cycles), int(dig_ch_tx.points_per_cycle()))
                    result_rx = result_rx.reshape(int(repetition*cycles), int(dig_ch_rx.points_per_cycle()))
                    windows_tx = dig_port_tx.measurement_windows       
                    ss_start_tx = 0
                    ss_end_tx = int(windows_tx[0][1] - windows_tx[0][0])
                    # ss2_start_tx = int(windows_tx[1][0] - windows_tx[0][0])
                    # ss2_end_tx = int(windows_tx[1][1] - windows_tx[0][0])
                    windows_rx = dig_port_rx.measurement_windows    
                    ss_start_rx = 0
                    ss_end_rx = int(windows_rx[0][1] - windows_rx[0][0])
                    ss2_start_rx = int(windows_rx[1][0] - windows_rx[0][0])
                    ss2_end_rx = int(windows_rx[1][1] - windows_rx[0][0])
                    _, result_gg_rx, _, other_rx = postselection_both_sides(
                        result_tx, result_rx, ss_start_tx, ss_end_tx, ss_start_rx, ss_end_rx
                    )
                    # print(windows_rx)
                    # print(result_rx.shape, result_gg_rx.shape)

                    result_gg_rx = result_gg_rx[:, int(ss2_start_rx//2):int(ss2_end_rx//2)]
                    other_rx = other_rx[:, int(ss2_start_rx//2):int(ss2_end_rx//2)]
                    signal_g = np.dot(result_gg_rx - mean_g_rx, ein_vec_rx)
                    signal_e = np.dot(other_rx - mean_g_rx, ein_vec_rx)
                    writer.add_data(
                        label=i, 
                        signal=np.array(list(signal_g)+list(np.inf*(np.ones(signal_e.shape))))
                    )
        finally:
            # off()
            print('finished')

    lo1.output(True)
    lo2.output(True)
    lo3.output(True)
    lo4.output(True)
    lo5.output(True)
    lo6.output(True)

    state = "g"
    prepare_seq = Sequence(ports_tx)
    prepare_seq.add(Delay(ge_pi_pulse_tx.params["duration"]), qubit_drive_port_tx)
    exp(prepare_seq, state)

    state = "+ge"
    prepare_seq = Sequence(ports_tx)
    prepare_seq.call(ge_half_pi_seq_tx)
    exp(prepare_seq, state)

    state = "-ge"
    prepare_seq = Sequence(ports_tx)
    prepare_seq.add(VirtualZ(np.pi), qubit_drive_port_tx)
    prepare_seq.call(ge_half_pi_seq_tx)
    prepare_seq.add(VirtualZ(np.pi), qubit_drive_port_tx)
    exp(prepare_seq, state)

    state = "+ige"
    prepare_seq = Sequence(ports_tx)
    prepare_seq.add(VirtualZ(np.pi/2), qubit_drive_port_tx)
    prepare_seq.call(ge_half_pi_seq_tx)
    prepare_seq.add(VirtualZ(-np.pi/2), qubit_drive_port_tx)
    exp(prepare_seq, state)

    state = "-ige"
    prepare_seq = Sequence(ports_tx)
    prepare_seq.add(VirtualZ(-np.pi/2), qubit_drive_port_tx)
    prepare_seq.call(ge_half_pi_seq_tx)
    prepare_seq.add(VirtualZ(np.pi/2), qubit_drive_port_tx)
    exp(prepare_seq, state)

    state = "e"
    prepare_seq = Sequence(ports_tx)
    prepare_seq.call(ge_pi_seq_tx)
    exp(prepare_seq, state)

    off()

    # photon_data = DataDict(
    #     time=dict(unit="ns"),
    #     photon_tx=dict(axes=["time"]),
    #     photon_rx=dict(axes=["time"])
    # )
    # photon_data.validate()

    # with DDH5Writer(photon_data, data_dir, name="photons_"+measurement_name) as photon_writer:
    #     photon_writer.add_data(
    #         time=2*np.arange(len(photon_tx)),
    #         photon_tx=photon_tx,
    #         photon_rx=photon_rx,
    #     )
