from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]

shot_count = 2000
hvi_trigger.trigger_period(300000)

var = Variables()
phase = Variable("phase", np.linspace(0, 2 * np.pi, 51), "rad")
# phase = Variable("phase", np.linspace(-np.pi, np.pi, 51), "rad")
var.add(phase)
var.compile()


if measure_which == "tx":
    JPA_phase_tx.params["phase"]=phase
    sequence_g = Sequence(port_list=ports_tx)
    sequence_g.call(readout_seq_tx)
    sequence_g.call(seq_JPA_tx)
    # sequence_g.draw()

    sequence_e = Sequence(port_list=ports_tx)
    sequence_e.call(ge_pi_seq_tx)
    sequence_e.trigger([qubit_drive_port_tx, readout_port_tx, JPA_port_tx, dig_port_tx])
    sequence_e.call(readout_seq_tx)
    sequence_e.call(seq_JPA_tx)
    # sequence_e.draw()
elif measure_which == "txQrx":
    JPA_phase_tx.params["phase"]=phase
    sequence_g = Sequence(port_list=ports_txQrx)
    sequence_g.call(readout_seq_tx)
    sequence_g.call(seq_JPA_tx)
    # sequence_g.draw()

    sequence_e = Sequence(port_list=ports_txQrx)
    sequence_e.call(ge_pi_seq_rx)
    sequence_e.trigger([qubit_drive_port_rx, readout_port_tx, JPA_port_tx, dig_port_tx])
    sequence_e.call(readout_seq_tx)
    sequence_e.call(seq_JPA_tx)
    # sequence_e.draw()
elif measure_which == "rx":
    JPA_phase_rx.params["phase"]=phase
    sequence_g = Sequence(port_list=ports_rx)
    sequence_g.call(readout_seq_rx)
    sequence_g.call(seq_JPA_rx)
    

    sequence_e = Sequence(port_list=ports_rx)
    sequence_e.call(ge_pi_seq_rx)
    sequence_e.trigger([qubit_drive_port_rx, readout_port_rx, JPA_port_rx, dig_port_rx])
    sequence_e.call(readout_seq_rx)
    sequence_e.call(seq_JPA_rx)
    

data = DataDict(
    phase=dict(unit="rad"),
    distance=dict(axes=["phase"]),
)
data.validate()

try:
    with DDH5Writer(data, data_path, name=measurement_name) as writer:
        writer.add_tag(tags)
        writer.backup_file([__file__, setup_file])
        writer.save_text("wiring.md", wiring)
        writer.save_dict("station_snapshot.json", station.snapshot())
        for update_command in tqdm(var.update_command_list):
            sequence_g.update_variables(update_command)
            
            load_sequence(sequence_g, cycles=shot_count)
            s11_g = demodulate(run(sequence_g, which=measure_which).mean(axis=0)) * voltage_step_tx
            sequence_e.update_variables(update_command)
            # sequence_e.draw()
            # raise SystemError
            load_sequence(sequence_e, cycles=shot_count)
            s11_e = demodulate(run(sequence_e, which=measure_which).mean(axis=0)) * voltage_step_tx
            writer.add_data(
                phase=sequence_g.variable_dict["phase"][0].value/np.pi,
                distance=s11_e - s11_g,
            )
finally:
    off()
    print("finished")