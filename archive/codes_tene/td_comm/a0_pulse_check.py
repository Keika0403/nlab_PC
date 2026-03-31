from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]
num_of_cycles = 5000
hvi_trigger.digitizer_delay(0)
hvi_trigger.trigger_period(10000)

readout_port = readout_port_rx
dig_port = dig_port_rx
dig_ch = dig_ch_rx

# readout_port = readout_port_tx
# dig_port = dig_port_tx
# dig_ch = dig_ch_tx

# readout_port.if_freq=0.01
seq = Sequence(port_list=[readout_port, dig_port, JPA_port_rx])
seq.add(Square(amplitude=0.35, duration=1000), JPA_port_rx, copy=False)


seq.add(Square(amplitude=1, duration=1000), readout_port)
seq.add(Acquire(2000), dig_port)
# seq.draw()

data = DataDict(
    time=dict(unit="ns"),
    waveform=dict(axes=["time"], unit="V"),
)
data.validate()
try:
    with DDH5Writer(data, data_path, name=measurement_name) as writer:
        writer.add_tag(tags)
        writer.backup_file([__file__, setup_file])
        writer.save_text("wiring.md", wiring)
        writer.save_dict("station_snapshot.json", station.snapshot())
        # seq.draw()
        # raise SystemError
        load_sequence(seq, cycles=num_of_cycles)
        waveform = run(seq, which=measure_which[:2]).mean(axis=0) * voltage_step_tx
        writer.add_data(
            time=dig_ch.sampling_interval() * np.arange(len(waveform)),
            waveform=waveform,
        )
        awg4.flush_waveform()
        awg2.flush_waveform()
        awg1.flush_waveform()
        dig_ch.stop()
# except:print('An error occurred')
finally:
    off()
    print('finished')