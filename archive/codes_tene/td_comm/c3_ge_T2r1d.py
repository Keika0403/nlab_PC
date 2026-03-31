from setup_td import *
from tqdm import tqdm

measurement_name = os.path.basename(__file__)[:-3]
tags.append("ge_pi")

# delay = np.arange(0, 1000, 51)
# detuning = 0.005
num_of_cycles = 3000

delay = Variable("delay", np.linspace(0, 1000, 51), "ns")
variables = Variables([delay])
detuning = -5e6

if measure_which == "tx":
    lo2.frequency(qubit_lo_freq_tx*1e9+detuning)
    sequence = Sequence(ports_tx)
    sequence.call(ge_half_pi_seq_tx)
    sequence.add(Delay(delay), qubit_drive_port_tx)
    sequence.add(VirtualZ(np.pi), qubit_drive_port_tx)
    sequence.call(ge_half_pi_seq_tx)
    sequence.call(readout_seq_tx)
elif measure_which == "txQrx":
    lo4.frequency(qubit_lo_freq_rx*1e9+detuning)
    sequence = Sequence(port_list=ports_txQrx)
    sequence.add(SetDetuning(detuning), qubit_drive_port_rx)
    sequence.call(ge_half_pi_seq_rx)
    sequence.add(Delay(delay), qubit_drive_port_rx)
    sequence.add(VirtualZ(pi), qubit_drive_port_rx)
    sequence.call(ge_half_pi_seq_rx)
    sequence.add(VirtualZ(pi), qubit_drive_port_rx)
    sequence.trigger(ports_txQrx)
    sequence.call(readout_seq_tx)
elif measure_which == "rx":
    lo4.frequency(qubit_lo_freq_rx*1e9+detuning)
    sequence = Sequence(ports_rx)
    sequence.call(ge_half_pi_seq_rx)
    sequence.add(Delay(delay), qubit_drive_port_rx)
    sequence.add(VirtualZ(np.pi), qubit_drive_port_rx)
    sequence.call(ge_half_pi_seq_rx)
    sequence.call(readout_seq_rx)

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
            data = run(sequence, which=measure_which[:2]).mean(axis=0) * voltage_step_tx
            spara=demodulate(data)
            writer.add_data(
                delay=sequence.variable_dict['delay'][0].value,
                s11=spara,
            )
            awg2.flush_waveform()
            awg1.flush_waveform()
            dig_ch_tx.stop()
finally:
    off()
    print('finished')