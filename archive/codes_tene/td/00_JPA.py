from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]

cycles = 10000
hvi_trigger.digitizer_delay(400)
currents = np.linspace(70e-6, 100e-6, 61)

var = Variables()
amp = Variable("amplitude", np.append([0], np.linspace(1.0, 1.4, 9)), "V")
var.add(amp)
var.compile()

seq_JPA = Sequence(port_list=[JPA_port])
seq_JPA.add(Square(amplitude=amp, duration=10000), JPA_port, copy=False)

seq = Sequence(port_list=[dig_port, JPA_port, readout_port])
seq.call(seq_JPA)
seq.add(Square(amplitude=0.35, duration=10000), readout_port)
seq.add(Acquire(100), dig_port)

data = DataDict(
    current=dict(unit="A"),
    amplitude=dict(unit="V"),
    gain=dict(axes=["current", "amplitude"], unit="dB")
)
data.validate()

try:
    with DDH5Writer(data, data_path, name=measurement_name) as writer:
        writer.add_tag(tags)
        writer.backup_file([__file__, setup_file, setup_parameters_file,])
        writer.save_text("wiring.md", wiring)
        writer.save_dict("station_snapshot.json", station.snapshot())
        for current in tqdm(currents):
            yoko.ramp_current(current, 1e-8, 0)
            for update_command in var.update_command_list:
                seq.update_variables(update_command)
                load_sequence(seq, cycles=cycles)
                waveform = run(seq).mean(axis=0) * voltage_step
                if seq.variable_dict["amplitude"][0].value == 0:
                    base = demodulate(waveform)
                    continue
                gain = 20 * np.log10(np.abs(demodulate(waveform) / base))
                writer.add_data(
                    amplitude = seq.variable_dict["amplitude"][0].value,
                    current=current,
                    gain=gain,
                )
                awg1.flush_waveform()
                dig_ch.stop()
except:print('An error occurred')
finally:
    off()
    print('finished')