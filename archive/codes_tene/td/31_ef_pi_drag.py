from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]
tags.append("ef_pi")

cycles = 4000
num_of_pairs = 17
beta = Variable('beta', np.linspace(-1.5, 1.5, 51), unit=' ')

var = Variables()
var.add(beta)
var.compile()
ef_pi_pulse_drag.params['beta']=beta

seq = Sequence(port_list=[qubit_drive_port, readout_port, dig_port])
seq.call(ge_pi_seq)
seq.add(SetDetuning(anharmonicity), qubit_drive_port)
for _ in range(num_of_pairs):
    seq.add(HalfDRAG(ef_pi_pulse, beta=beta), qubit_drive_port)
    seq.add(VirtualZ(np.pi), qubit_drive_port)
    seq.add(HalfDRAG(ef_pi_pulse, beta=beta), qubit_drive_port)
    seq.add(VirtualZ(np.pi), qubit_drive_port)
seq.trigger([qubit_drive_port, readout_port, dig_port])
seq.call(readout_seq)
seq.add(Acquire(100), dig_port)

seq1 = Sequence(port_list=[qubit_drive_port, readout_port, dig_port])
seq1.call(ge_pi_seq)
seq1.add(SetDetuning(anharmonicity), qubit_drive_port)
for _ in range(num_of_pairs):
    seq1.add(HalfDRAG(ef_half_pi_pulse, beta=beta), qubit_drive_port)
    seq1.add(VirtualZ(np.pi), qubit_drive_port)
    seq1.add(HalfDRAG(ef_half_pi_pulse, beta=beta), qubit_drive_port)
    seq1.add(VirtualZ(np.pi), qubit_drive_port)
seq1.trigger([qubit_drive_port, readout_port, dig_port])
seq1.call(readout_seq)
seq1.add(Acquire(100), dig_port)

data = DataDict(
    beta=dict(unit="ns"),
    s11=dict(axes=["beta"]),
)
data.validate()
try:
    with DDH5Writer(data, data_path, name=measurement_name) as writer:
        writer.add_tag(tags)
        writer.backup_file([__file__, setup_file, setup_parameters_file])
        writer.save_text("wiring.md", wiring)
        writer.save_dict("station_snapshot.json", station.snapshot())
        for update_command in tqdm(var.update_command_list):
            # seq.update_variables(update_command)
            # load_sequence(seq, cycles=cycles)
            seq1.update_variables(update_command)
            load_sequence(seq1, cycles=cycles)

            writer.add_data(
            #    beta=seq.variable_dict["beta"][0].value,
            #    s11=demodulate(run(seq).mean(axis=0) * voltage_step),
                beta=seq1.variable_dict["beta"][0].value,
                s11=demodulate(run(seq1).mean(axis=0) * voltage_step),
            )
except:print('An error occurred')
finally:
    off()
    print('finished')