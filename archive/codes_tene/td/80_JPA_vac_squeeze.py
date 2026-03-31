from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]

shot_number = 50000
hvi_trigger.digitizer_delay(400)
hvi_trigger.trigger_period(10000)

var = Variables()
amplitude = Variable("amplitude", np.linspace(0.5, 1.5, 31), "V")

var.add(amplitude)
var.compile()
seq_JPA = Sequence(port_list=[JPA_port])
seq_JPA.add(Square(amplitude=amplitude, duration=2000), JPA_port, copy=False)
seq = Sequence(port_list=[dig_port, JPA_port])
seq.call(seq_JPA)
seq.add(Acquire(80), dig_port)

data = DataDict(
    shot_number=dict(),
    pump_amplitude=dict(unit="V"),
    s11=dict(axes=["shot_number", "pump_amplitude"]),
)
data.validate()

dig_ch.cycles(shot_number)
try:
    with DDH5Writer(data, data_path, name=measurement_name) as writer:
        writer.add_tag(tags)
        writer.backup_file([__file__, setup_file])
        writer.save_text("wiring.md", wiring)
        writer.save_dict("station_snapshot.json", station.snapshot())
        for update_command in tqdm(var.update_command_list):
            seq.update_variables(update_command)
            seq.draw()
            raise System
            load_sequence(seq, cycles=shot_number)
            s11 = demodulate(run(seq)) * voltage_step
            writer.add_data(
                pump_amplitude=seq.variable_dict["amplitude"][0].value,
                shot_number=np.arange(shot_number),
                s11=s11,
            )
            awg1.flush_waveform()
            dig_ch.stop()
except:print('An error occurred')
finally:
    off()
    print('finished')