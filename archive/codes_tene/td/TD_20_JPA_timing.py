from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]

cycles = 50000
hvi_trigger.digitizer_delay(0)
hvi_trigger.trigger_period(10000)

seq = Sequence(port_list=[dig_port, JPA_port])
# seq.call(seq_JPA)
seq.add(Square(amplitude=1., duration=1000), JPA_port)
seq.add(Acquire(2000), dig_port)

data = DataDict(
    time=dict(unit="ns"),
    mean=dict(axes=["time"]),
    std=dict(axes=["time"]),
)
data.validate()

dig_ch.cycles(cycles)
try:
    with DDH5Writer(data, data_path, name=measurement_name) as writer:
        writer.add_tag(tags)
        writer.backup_file([__file__, setup_file])
        writer.save_text("wiring.md", wiring)
        writer.save_dict("station_snapshot.json", station.snapshot())
        load_sequence(seq, cycles=cycles)
        data = run(seq) * voltage_step
        mean = data.mean(axis=0)
        stderr = data.std(axis=0)
        writer.add_data(
            time=dig_ch.sampling_interval()*np.arange(len(mean)),
            mean=mean,
            std=stderr,
        )
        awg1.flush_waveform()
        dig_ch.stop()
except:print('An error occurred')
finally:
    off()
    print('finished')
