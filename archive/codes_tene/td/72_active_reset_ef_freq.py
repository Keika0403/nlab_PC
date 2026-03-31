from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]

anhms = np.linspace(-0.39,-0.36,61)
fogi_amplitude=0.5
num_of_cycles = 10000

var = Variables()
anhm = Variable('anharmonicity', value_array=anhms, unit='GHz')
var.add(anhm)
var.compile()

seq = Sequence(port_list=[qubit_drive_port, fogi_port, readout_port, dig_port])
seq.call(ge_pi_seq)
seq.add(SetDetuning(anhm), qubit_drive_port)
seq.trigger([qubit_drive_port, fogi_port])
seq.add(FlatTop(Gaussian(amplitude=fogi_amplitude, fwhm=10, duration=20), top_duration=1100), fogi_port)
seq.add(FlatTop(Gaussian(amplitude=0.00559, fwhm=10, duration=20), top_duration=1100), qubit_drive_port)
seq.trigger([qubit_drive_port, fogi_port, readout_port, dig_port])
seq.call(readout_seq)
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
            load_sequence(seq, cycles=num_of_cycles)
            data = run(seq).mean(axis=0)*voltage_step
            spara=demodulate(data)
            # print(f"anharmonicity:{seq.variable_dict['anharmonicity'][0].value}")
            writer.add_data(
                anhm=seq.variable_dict['anharmonicity'][0].value,
                s11=spara,
            )
            awg2.flush_waveform()
            awg1.flush_waveform()
            dig_ch.stop()
except:print('An error occurred')
finally:
    off()
    print('finished')