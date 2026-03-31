from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]
tags.append("single_shot")
tags.append("readout_threshold")

shot_count = 10000
repetition = 1
hvi_trigger.trigger_period(300000)
hvi_trigger.digitizer_delay(400)

if measure_which == "tx":
    sequence_g = Sequence(port_list=ports_tx)
    sequence_g.call(readout_ss_seq_tx)
    # sequence_g.draw()

    sequence_e = Sequence(port_list=ports_tx)
    sequence_e.call(ge_pi_seq_tx)
    sequence_e.trigger(ports_tx)
    sequence_e.call(readout_ss_seq_tx)
    # sequence_e.draw()

elif measure_which == "txQrx":
    sequence_g = Sequence(port_list=ports_txQrx)
    sequence_g.call(readout_ss_seq_tx)
    # sequence_g.draw()

    sequence_e = Sequence(port_list=ports_txQrx)
    sequence_e.call(ge_pi_seq_rx)
    sequence_e.trigger(ports)
    sequence_e.call(readout_ss_seq_tx)
    # sequence_e.draw()

elif measure_which == "rx":
    sequence_g = Sequence(port_list=ports_rx)
    sequence_g.call(readout_ss_seq_rx)
    # sequence_g.draw()

    sequence_e = Sequence(port_list=ports_rx)
    sequence_e.call(ge_pi_seq_rx)
    sequence_e.trigger(ports_rx)
    sequence_e.call(readout_ss_seq_rx)
    # sequence_e.draw()

data = DataDict(
    time=dict(unit="ns"),
    pulse_g=dict(axes=["time"]),
    pulse_e=dict(axes=["time"]),
    signal_g=dict(),
    signal_e=dict(),
)
data.validate()

try:
    with DDH5Writer(data, data_path, name=measurement_name) as writer:
        writer.add_tag(tags)
        writer.backup_file([__file__, setup_file, setup_parameters_file])
        writer.save_text("wiring.md", wiring)
        writer.save_dict("station_snapshot.json", station.snapshot())
        result_g = []
        result_e = []
        for _ in tqdm(range(repetition)):
            load_sequence(sequence_g, cycles=shot_count)
            pulse_g = run(sequence_g, which=measure_which) * voltage_step_tx
            result_g.append(pulse_g)
            
            load_sequence(sequence_e, cycles=shot_count)
            pulse_e = run(sequence_e, which=measure_which) * voltage_step_tx
            result_e.append(pulse_e)
        result_g = np.array(result_g).reshape(int(repetition*shot_count), len(pulse_g[0]))
        result_e = np.array(result_e).reshape(int(repetition*shot_count), len(pulse_e[0]))
        s11_g = demodulate(result_g)
        s11_e = demodulate(result_e)
        writer.add_data(
            time=dig_ch_tx.sampling_interval() * np.arange(len(pulse_g[0])),
            pulse_g=result_g.mean(axis=0),
            pulse_e=result_e.mean(axis=0),
            signal_g=s11_g,
            signal_e=s11_e,
        )

finally:
    off()
    print("finished")
