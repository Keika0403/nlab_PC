from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]

shot_count = 10000

var = Variables()
phase = Variable("phase", np.linspace(0, 2 * np.pi, 41), "rad")
var.add(phase)
var.compile()
JPA_phase.params["phase"]=phase

sequence_g = Sequence(port_list=[qubit_drive_port, fogi_port, readout_port, JPA_port, dig_port])
sequence_g.trigger([qubit_drive_port, fogi_port, readout_port, JPA_port, dig_port])
sequence_g.call(readout_seq)
sequence_g.call(seq_JPA)
# sequence_g.add(Acquire(width_readout_pulse),dig_port)
# sequence_g.draw()

sequence_e = Sequence(port_list=[qubit_drive_port, fogi_port, readout_port, JPA_port, dig_port])
sequence_e.call(ge_pi_seq)
sequence_e.trigger([qubit_drive_port, readout_port, JPA_port, dig_port])
sequence_e.call(readout_seq)
sequence_e.call(seq_JPA)
# sequence_e.add(Acquire(width_readout_pulse),dig_port)
# sequence_e.draw()

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
            s11_g = demodulate(run(sequence_g).mean(axis=0)) * voltage_step

            sequence_e.update_variables(update_command)
            load_sequence(sequence_e, cycles=shot_count)
            s11_e = demodulate(run(sequence_e).mean(axis=0)) * voltage_step
            
            writer.add_data(
                phase=sequence_g.variable_dict["phase"][0].value/np.pi,
                distance=s11_e-s11_g,
            )
finally:
    off()
    print("finished")