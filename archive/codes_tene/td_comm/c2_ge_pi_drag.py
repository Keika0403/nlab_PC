from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]
tags.append("ge_pi")

num_of_cycles = 3000
num_of_pairs = 17
beta = Variable('beta', np.linspace(-2, 2, 101), unit=' ')

var = Variables()
var.add(beta)
var.compile()

if measure_which == "tx":
    # ge_pi_pulse_drag_tx.params['beta']=beta
    seq = Sequence(port_list=ports_tx)
    for _ in range(num_of_pairs):
        seq.add(HalfDRAG(ge_pi_pulse_tx, beta=beta), qubit_drive_port_tx)
        seq.add(VirtualZ(np.pi), qubit_drive_port_tx)
        seq.add(HalfDRAG(ge_pi_pulse_tx, beta=beta), qubit_drive_port_tx)
        seq.add(VirtualZ(np.pi), qubit_drive_port_tx)
    seq.trigger(ports_tx)
    seq.call(readout_seq_tx)
elif measure_which == "txQrx":
    ge_pi_pulse_drag_rx.params['beta']=beta
    seq = Sequence(port_list=ports_txQrx)
    for _ in range(num_of_pairs):
        seq.add(HalfDRAG(ge_pi_pulse_rx, beta=beta), qubit_drive_port_rx)
        seq.add(VirtualZ(np.pi), qubit_drive_port_rx)
        seq.add(HalfDRAG(ge_pi_pulse_rx, beta=beta), qubit_drive_port_rx)
        seq.add(VirtualZ(np.pi), qubit_drive_port_rx)
    seq.trigger(ports_txQrx)
    seq.call(readout_seq_tx)
elif measure_which == "rx":
    ge_pi_pulse_drag_rx.params['beta']=beta
    seq = Sequence(port_list=ports_rx)
    for _ in range(num_of_pairs):
        seq.add(HalfDRAG(ge_pi_pulse_rx, beta=beta), qubit_drive_port_rx)
        seq.add(VirtualZ(np.pi), qubit_drive_port_rx)
        seq.add(HalfDRAG(ge_pi_pulse_rx, beta=beta), qubit_drive_port_rx)
        seq.add(VirtualZ(np.pi), qubit_drive_port_rx)
    seq.trigger(ports_rx)
    seq.call(readout_seq_rx)

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
            seq.update_variables(update_command)
            load_sequence(seq, cycles=num_of_cycles)
            # print(f"beta:{seq.variable_dict['beta'][0].value}")
            writer.add_data(
                beta=seq.variable_dict["beta"][0].value,
                s11=demodulate(run(seq, which=measure_which[:2]).mean(axis=0) * voltage_step_tx),
            )
            awg2.flush_waveform()
            awg1.flush_waveform()
            dig_ch_tx.stop()
except:print('An error occurred')
finally:
    off()
    print('finished')