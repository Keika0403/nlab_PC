from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]

cycles = 2000
readout_if_freqs = np.linspace(0.08, 0.16, 401)
hvi_trigger.trigger_period(10000)
electrical_delay = -2/(10.14-10.13)+1/(10.136-10.1303)+1/(9.3789-9.3333)

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
    # s11_e=dict(axes=["probe_frequency"])
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
        
        for if_freq in tqdm(readout_if_freqs):
            print(readout_port_tx.if_freq)
            # readout_port_tx.if_freq = if_freq
            readout_port_rx.if_freq = if_freq
            # draw(sequence_g)
            load_sequence(sequence_g, cycles=cycles)
            pulse_g = run(sequence_g, plot=0, which=measure_which).mean(axis=0)

            # load_sequence(sequence_e, cycles=cycles)
            # pulse_e = run(sequence_e, plot=0, which=measure_which).mean(axis=0)


            # s11_e = demodulate(pulse_e, demodulation_if=if_freq) #* np.exp(2j*np.pi*(IF)*electrical_delay)
            # print(readout_lo_freq_tx, if_freq)
            writer.add_data(
                probe_frequency=readout_lo_freq_tx - if_freq,
                # probe_frequency=readout_lo_freq_rx + if_freq,
                s11_g=demodulate(pulse_g, demodulation_if=if_freq) * np.exp(-2j*np.pi*(readout_lo_freq_tx - if_freq)*electrical_delay),
                # s11_e=demodulate(pulse_e, demodulation_if=if_freq)* np.exp(-2j*np.pi*(readout_lo_freq_tx - if_freq)*electrical_delay)
            )
except Exception as e:
    print(e)#
finally:
    off()
    print("finished")