from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]

shot_count = 50000
repetition = 1
hvi_trigger.digitizer_delay(400)
hvi_trigger.trigger_period(10000)
# lo1.frequency(10.27e9)
# JPA_port_tx.if_freq = (10.152-10.27) * 2
# target_frequency = 10.152

lo6.frequency(10.29e9)
JPA_port_tx.if_freq = (10.414-10.29) * 2
target_frequency = 10.414


var = Variables()
amplitude = Variable("amplitude", np.linspace(0.6, 1.0, 21), "V")

var.add(amplitude)
var.compile()
if measure_which[:2] == "tx":
    seq_JPA = Sequence(port_list=[JPA_port_tx])
    seq_JPA.add(Square(amplitude=amplitude, duration=2000), JPA_port_tx, copy=False)
    seq = Sequence(port_list=[dig_port_tx, JPA_port_tx])
    seq.call(seq_JPA)
    seq.add(Acquire(80), dig_port_tx)
    target_frequency = readout_freq_tx
    JPA_port_tx.if_freq = (target_frequency * 2 - readout_lo_freq_tx * 2)
if measure_which == "rx":
    seq_JPA = Sequence(port_list=[JPA_port_rx])
    seq_JPA.add(Square(amplitude=amplitude, duration=2000), JPA_port_rx, copy=False)
    seq = Sequence(port_list=[dig_port_rx, JPA_port_rx])
    seq.call(seq_JPA)
    seq.add(Acquire(80), dig_port_rx)
    target_frequency = readout_freq_rx
    JPA_port_rx.if_freq = (target_frequency * 2 - readout_lo_freq_rx* 2)

data = DataDict(
    pump_amplitude=dict(unit="V"),
    s11=dict(axes=["pump_amplitude"]),
)
data.validate()
try:
    with DDH5Writer(data, data_path, name=measurement_name) as writer:
        writer.add_tag(tags)
        writer.backup_file([__file__, setup_file])
        writer.save_text("wiring.md", wiring)
        writer.save_dict("station_snapshot.json", station.snapshot())
        for update_command in tqdm(var.update_command_list):
            result = []
            for _ in range(repetition):
                seq.update_variables(update_command)
                load_sequence(seq, cycles=shot_count)
                s11 = demodulate(run(seq, which=measure_which), demodulation_if=demo_if) * voltage_step_tx
                result = np.append(result, s11)
            writer.add_data(
                pump_amplitude=seq.variable_dict["amplitude"][0].value,
                # shot_count=np.arange(shot_count),
                s11=result,
            )
            awg1.flush_waveform()
            dig_ch_tx.stop()
except Exception as e:
    print(e)#
finally:
    off()
    print('finished')