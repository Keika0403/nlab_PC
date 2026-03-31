from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]
tags.append("ge_pi")

drive_amplitude = np.linspace(0., 1., 51)
num_of_cycles = 3000
num_of_pulses = 5

var = Variables()
v1 = Variable('drive_amplitude', value_array=drive_amplitude, unit='V')
var.add(v1)
var.compile()

if measure_which == "tx":
    ge_pi_pulse_tx.params['amplitude']=v1
    seq = Sequence(port_list=ports_tx)
    for _ in range(num_of_pulses):
        seq.call(ge_pi_seq_tx)
    seq.trigger(ports_tx)
    seq.call(readout_seq_tx)
elif measure_which == "txQrx":
    ge_pi_pulse_rx.params['amplitude']=v1
    seq = Sequence(port_list=ports_txQrx)
    for _ in range(num_of_pulses):
        seq.call(ge_pi_seq_rx)
    seq.trigger(ports_txQrx)
    seq.call(readout_seq_tx)
elif measure_which == "rx":
    ge_pi_pulse_rx.params['amplitude']=v1
    seq = Sequence(port_list=ports_rx)
    for _ in range(num_of_pulses):
        seq.call(ge_pi_seq_rx)
    seq.trigger(ports_rx)
    seq.call(readout_seq_rx)
# check_sequence_formulti(seq, var, [40])

data = DataDict(
    amplitude=dict(unit="V"),
    s11=dict(axes=["amplitude"]),
)
data.validate()
# try:
with DDH5Writer(data, data_path, name=measurement_name) as writer:
    writer.add_tag(tags)
    writer.backup_file([__file__, setup_file, setup_parameters_file])
    writer.save_text("wiring.md", wiring)
    writer.save_dict("station_snapshot.json", station.snapshot())
    for update_command in tqdm(var.update_command_list):
        seq.update_variables(update_command)
        # seq.draw()
        # raise SystemError
        load_sequence(seq, cycles=num_of_cycles)
        data = run(seq, which=measure_which[:2]).mean(axis=0) * voltage_step_tx
        spara=demodulate(data)
        writer.add_data(
            amplitude=seq.variable_dict['drive_amplitude'][0].value, 
            s11=spara,
        )
        awg2.flush_waveform()
        awg1.flush_waveform()
        dig_ch_tx.stop()
#     except:print('An error occurred')
# finally:
#     off()
#     print('finished')