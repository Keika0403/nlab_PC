from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]
tags.append("ef_pi")

num_of_cycles = 3000

delay = Variable("delay", np.linspace(0, 1000,  51), "ns")
variables = Variables([delay])
detuning = -0.005

if measure_which == "tx":
    lo2.frequency(qubit_lo_freq_tx*1e9)
    sequence = Sequence(ports_tx)
    sequence.call(ge_pi_seq_tx)
    sequence.add(SetDetuning(detuning + anharmonicity_tx), qubit_drive_port_tx)
    sequence.add(ef_half_pi_pulse_drag_tx, qubit_drive_port_tx)
    sequence.add(Delay(delay), qubit_drive_port_tx)
    sequence.add(VirtualZ(np.pi), qubit_drive_port_tx)
    sequence.add(ef_half_pi_pulse_drag_tx, qubit_drive_port_tx)
    sequence.add(VirtualZ(np.pi), qubit_drive_port_tx)
    sequence.trigger([qubit_drive_port_tx, readout_port_tx, dig_port_tx])
    sequence.call(readout_seq_tx)
    sequence.add(Acquire(100), dig_port_tx)
elif measure_which == "txQrx":
    lo4.frequency(qubit_lo_freq_rx*1e9)
    sequence = Sequence(port_list=ports_txQrx)
    sequence.call(ge_pi_seq_rx)
    sequence.add(SetDetuning(detuning + anharmonicity_rx), qubit_drive_port_rx)
    sequence.add(ef_half_pi_pulse_drag_rx, qubit_drive_port_rx)
    sequence.add(Delay(delay), qubit_drive_port_rx)
    sequence.add(VirtualZ(np.pi), qubit_drive_port_rx)
    sequence.add(ef_half_pi_pulse_drag_rx, qubit_drive_port_rx)
    sequence.add(VirtualZ(np.pi), qubit_drive_port_rx)
    sequence.trigger(ports_txQrx)
    # sequence.call(ge_pi_seq_rx)
    sequence.call(readout_seq_tx)
    sequence.add(Acquire(100), dig_port_tx)
elif measure_which == "rx":
    lo4.frequency(qubit_lo_freq_rx*1e9)
    sequence = Sequence(ports_rx)
    sequence.call(ge_pi_seq_rx)
    sequence.add(SetDetuning(detuning + anharmonicity_rx), qubit_drive_port_rx)
    sequence.add(ef_half_pi_pulse_drag_rx, qubit_drive_port_rx)
    sequence.add(Delay(delay), qubit_drive_port_rx)
    sequence.add(VirtualZ(np.pi), qubit_drive_port_rx)
    sequence.add(ef_half_pi_pulse_drag_rx, qubit_drive_port_rx)
    sequence.add(VirtualZ(np.pi), qubit_drive_port_rx)
    sequence.trigger([qubit_drive_port_rx, readout_port_rx, dig_port_rx])
    sequence.call(readout_seq_rx)
    sequence.add(Acquire(100), dig_port_rx)


data = DataDict(
    delay=dict(unit="ns"),
    s11=dict(axes=["delay"]),
)
data.validate()

try:
    with DDH5Writer(data, data_path, name=measurement_name) as writer:
        writer.add_tag(tags)
        writer.backup_file([__file__, setup_file, setup_parameters_file])
        writer.save_text("wiring.md", wiring)
        writer.save_dict("station_snapshot.json", station.snapshot())
        for update_command in tqdm(variables.update_command_list):
            sequence.update_variables(update_command)
            # sequence.draw()
            # raise SystemError
            load_sequence(sequence, cycles=num_of_cycles)
            writer.add_data(
                delay=sequence.variable_dict['delay'][0].value,
                s11=demodulate(run(sequence, which=measure_which[:2]).mean(axis=0) * voltage_step_tx))
finally:
    off()
    print('finished')
