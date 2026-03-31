from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]
num_of_cycles = 5000
trigger_period = 10000

hvi_trigger.digitizer_delay(0)
hvi_trigger.trigger_period(trigger_period)

dig_acquire.params["duration"] = 200
# readout_port.if_freq=0.01
seq = Sequence(port_list=[readout_port, dig_port])
seq.add(Square(amplitude=0.9, duration=1000), readout_port)
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
        load_sequence(seq, cycles=num_of_cycles)
        waveform = run(seq).mean(axis=0) * voltage_step
        writer.add_data(
            time=dig_ch.sampling_interval() * np.arange(len(waveform)),
            waveform=waveform,
        )
        awg2.flush_waveform()
        awg1.flush_waveform()
        dig_ch.stop()
except Exception as e:
    print(e)#print('An error occurred:', Exception)
finally:
    off()
    print('finished')