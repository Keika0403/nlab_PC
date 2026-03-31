from setup_td import *


measurement_name = os.path.basename(__file__)[:-3]
fogi_freq =  [6.65]
fogi_amplitude =[0.38]
num_of_cycles = 10000

duration = np.linspace(0, 1000, 51)


def fogi_time_sweep(duration, a):
    seq = Sequence(port_list = [qubit_drive_port_tx, fogi_port_tx, readout_port_tx, dig_port_tx])
    seq.call(ge_pi_seq_tx)
    seq.call(ef_pi_seq_tx)
    seq.trigger([qubit_drive_port_tx, fogi_port_tx])
    seq.add(Square(amplitude=a, duration=duration), fogi_port_tx)
    seq.trigger([qubit_drive_port_tx, fogi_port_tx, readout_port_tx, dig_port_tx])
    seq.call(readout_seq_tx)
    return seq

# seq.draw()
# raise SystemError


data = DataDict(
    fogi_duration=dict(unit="ns"),
    fogi_amp = dict(unit="V"),
    s11=dict(axes=["fogi_duration", "fogi_amp"]),
)
data.validate()


dig_ch_tx.cycles(num_of_cycles)
lo1.output(True)
lo2.output(True)
lo3.output(True)
with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())
    for i in range(len(fogi_freq)):
        fogi_port_tx.if_freq = fogi_freq[i] - fogi_lo_freq_tx
        amp=fogi_amplitude[i]
        for d in tqdm(duration):
            seq = fogi_time_sweep(d, amp)
            load_sequence(seq, cycles=num_of_cycles)#, chirp=True)
            data = run(seq, which=measure_which[:2]).mean(axis=0)*voltage_step_tx
            spara=demodulate(data)
            writer.add_data(
                fogi_duration=d,
                fogi_amp = amp,
                s11=spara,
            )

        awg2.flush_waveform()
        awg1.flush_waveform()
        dig_ch_tx.stop()
