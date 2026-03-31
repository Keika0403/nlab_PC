from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]

cycles = 3000
readout_probe_freqs = np.linspace(9.13, 9.63, 251) #9.38
hvi_trigger.trigger_period(10000)
electrical_delay = -1/(9.1525-9.1327)

if measure_which == "tx":
    sequence_g = Sequence(ports_tx)
    sequence_g.call(readout_seq_tx)

    sequence_e = Sequence(ports_tx)
    sequence_e.call(ge_pi_seq_tx)
    sequence_e.trigger(ports_tx)
    sequence_e.call(readout_seq_tx)
elif measure_which == "txQrx":
    sequence_g = Sequence(ports_txQrx)
    sequence_g.call(readout_seq_tx)

    sequence_e = Sequence(ports_txQrx)
    sequence_e.call(ge_pi_seq_rx)
    sequence_e.trigger(ports_txQrx)
    sequence_e.call(readout_seq_tx)
elif measure_which == "rx":
    sequence_g = Sequence(ports_rx)
    sequence_g.call(readout_seq_rx)

    sequence_e = Sequence(ports_rx)
    sequence_e.call(ge_pi_seq_rx)
    sequence_e.trigger(ports_rx)
    sequence_e.call(readout_seq_rx)


data = DataDict(
    probe_frequency=dict(unit="GHz"),
    s11_g=dict(axes=["probe_frequency"]),
    s11_e=dict(axes=["probe_frequency"]),
    s11_eg_div=dict(axes=["probe_frequency"])
)
data.validate()
try:
    with DDH5Writer(data, data_path, name=measurement_name) as writer:
        writer.add_tag(tags)
        writer.backup_file([__file__, setup_file])
        writer.save_text("wiring.md", wiring)
        writer.save_dict("station_snapshot.json", station.snapshot())
        # sequence_g.draw()
        # raise SystemError
        
        for probe_freq in tqdm(readout_probe_freqs):
            readout_lo_freq_tx = probe_freq + readout_if_freq_tx
            lo1.frequency(readout_lo_freq_tx*1e9)
            # readout_lo_freq_tx = probe_freq - readout_if_freq_tx
            # lo6.frequency(readout_lo_freq_tx*1e9)

            # draw(sequence_g)
            load_sequence(sequence_g, cycles=cycles)
            pulse_g = run(sequence_g, plot=0, which=measure_which).mean(axis=0)

            load_sequence(sequence_e, cycles=cycles)
            pulse_e = run(sequence_e, plot=0, which=measure_which).mean(axis=0)


            # s11_e = demodulate(pulse_e, demodulation_if=if_freq) #* np.exp(2j*np.pi*(IF)*electrical_delay)
            # print(readout_lo_freq_tx, if_freq)
            writer.add_data(
                probe_frequency=probe_freq,
                # probe_frequency=readout_lo_freq_rx + if_freq,
                s11_g=demodulate(pulse_g, demodulation_if=readout_if_freq_tx) * np.exp(-2j*np.pi*(readout_lo_freq_tx - readout_if_freq_tx)*electrical_delay),
                s11_e=demodulate(pulse_e, demodulation_if=readout_if_freq_tx)* np.exp(-2j*np.pi*(readout_lo_freq_tx - readout_if_freq_tx)*electrical_delay),
                s11_eg_div=demodulate(pulse_e, demodulation_if=readout_if_freq_tx)/demodulate(pulse_g, demodulation_if=readout_if_freq_tx)
            )
except Exception as e:
    print(e)#
finally:
    off()
    print("finished")