from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]

anhms = np.linspace(-0.38,-0.34,101)
fogi_amplitude=0.2
num_of_cycles = 3000

var = Variables()
anhm = Variable('anharmonicity', value_array=anhms, unit='GHz')
var.add(anhm)
var.compile()

if measure_which == "tx":
    seq = Sequence(port_list=[qubit_drive_port_tx, fogi_port_tx, readout_port_tx, dig_port_tx])
    seq.call(ge_pi_seq_tx)
    seq.add(SetDetuning(anhm), qubit_drive_port_tx)
    seq.trigger([qubit_drive_port_tx, fogi_port_tx])
    seq.add(FlatTop(Gaussian(amplitude=fogi_amplitude, fwhm=10, duration=20), top_duration=1000), fogi_port_tx)
    seq.add(FlatTop(Gaussian(amplitude=0.06, fwhm=10, duration=20), top_duration=1000), qubit_drive_port_tx)
    seq.trigger([qubit_drive_port_tx, fogi_port_tx, readout_port_tx, dig_port_tx])
    seq.call(readout_seq_tx)
    # seq.add(Acquire(100), dig_port)
elif measure_which == "txQrx":
    seq = Sequence(port_list=[qubit_drive_port_rx, fogi_port_rx, readout_port_tx, dig_port_tx])
    seq.call(ge_pi_seq_rx)
    seq.add(SetDetuning(anhm), qubit_drive_port_rx)
    seq.trigger([qubit_drive_port_rx, fogi_port_rx])
    seq.add(FlatTop(Gaussian(amplitude=fogi_amplitude, fwhm=10, duration=20), top_duration=1000), fogi_port_rx)
    seq.add(FlatTop(Gaussian(amplitude=0.02, fwhm=10, duration=20), top_duration=1000), qubit_drive_port_rx)
    seq.trigger([qubit_drive_port_rx, fogi_port_rx, readout_port_tx, dig_port_tx])
    seq.call(readout_seq_tx)
    # seq.add(Acquire(100), dig_port)
elif measure_which == "rx":
    seq = Sequence(port_list=[qubit_drive_port_rx, fogi_port_rx, readout_port_rx, dig_port_rx])
    seq.call(ge_pi_seq_rx)
    seq.add(SetDetuning(anhm), qubit_drive_port_rx)
    seq.trigger([qubit_drive_port_rx, fogi_port_rx])
    seq.add(FlatTop(Gaussian(amplitude=fogi_amplitude, fwhm=10, duration=20), top_duration=1000), fogi_port_rx)
    seq.add(FlatTop(Gaussian(amplitude=0.04, fwhm=10, duration=20), top_duration=1000), qubit_drive_port_rx)
    seq.trigger([qubit_drive_port_rx, fogi_port_rx, readout_port_rx, dig_port_rx])
    seq.call(readout_seq_rx)
    # seq.add(Acquire(100), dig_port)

data = DataDict(
    anhm=dict(unit="GHz"),
    s11=dict(axes=["anhm"], unit=""),
)
data.validate()

try:
    with DDH5Writer(data, data_path, name=measurement_name) as writer:
        writer.add_tag(tags)
        writer.backup_file([__file__, setup_file])
        writer.save_text("wiring.md", wiring)
        writer.save_dict("station_snapshot.json", station.snapshot())
        for update_command in tqdm(var.update_command_list):
            seq.update_variables(update_command)
            # seq.draw()
            # raise SystemError
            load_sequence(seq, cycles=num_of_cycles)
            data = run(seq).mean(axis=0)*voltage_step_tx
            spara=demodulate(data)
            # print(f"anharmonicity:{seq.variable_dict['anharmonicity'][0].value}")
            writer.add_data(
                anhm=seq.variable_dict['anharmonicity'][0].value,
                s11=spara,
            )
            awg3.flush_waveform()
            awg2.flush_waveform()
            awg1.flush_waveform()
            dig_ch_tx.stop()
except:print('An error occurred')
finally:
    off()
    print('finished')