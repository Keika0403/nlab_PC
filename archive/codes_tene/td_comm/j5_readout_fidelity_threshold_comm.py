from setup_td import *
from setup_td_tomography import *

measurement_name = os.path.basename(__file__)[:-3]
tags.append("single_shot")
tags.append("readout_threshold")

shot_count = 50000
repetition = 2
hvi_trigger.trigger_period(400000)

sequence_g = Sequence(port_list=[qubit_drive_port_rx, fogi_port_rx, readout_port, JPA_port, dig_port])
sequence_g.call(readout_single_shot_seq)
# sequence_g.draw()

sequence_e = Sequence(port_list=[qubit_drive_port_rx, fogi_port_rx, readout_port, JPA_port, dig_port])
sequence_e.call(ge_pi_seq_rx)
sequence_e.trigger([qubit_drive_port_rx, readout_port, JPA_port, dig_port])
sequence_e.call(readout_single_shot_seq)
# sequence_e.draw()
data = DataDict(
    time=dict(unit="ns"),
    pulse_g=dict(axes=["time"]),
    pulse_e=dict(axes=["time"]),
    pulse_g_org=dict(axes=["time"]),
    pulse_e_org=dict(axes=["time"]),
    signal_g=dict(),
    signal_e=dict(),
)
data.validate()

try:
    with DDH5Writer(data, data_path, name=measurement_name) as writer:
        writer.add_tag(tags)
        writer.backup_file([__file__, setup_file, setup_parameters_file])
        writer.save_text("wiring.md", wiring)
        writer.save_dict("station_snapshot.json", station.snapshot())
        result_g = []
        result_e = []
        for _ in tqdm(range(repetition)):
            load_sequence_comm(sequence_g, cycles=shot_count)
            pulse_g = run_comm(sequence_g, plot=0) * voltage_step_comm
            result_g.append(pulse_g)
            
            load_sequence_comm(sequence_e, cycles=shot_count)
            pulse_e = run_comm(sequence_e, plot=0) * voltage_step_comm
            result_e.append(pulse_e)
        result_g = np.array(result_g).reshape(int(repetition*shot_count), len(pulse_g[0]))
        result_e = np.array(result_e).reshape(int(repetition*shot_count), len(pulse_e[0]))
        s11_g = demodulate_comm(result_g)
        s11_e = demodulate_comm(result_e)
        writer.add_data(
            time=dig_ch_comm.sampling_interval() * np.arange(len(pulse_g[0])),
            pulse_g=result_g.mean(axis=0),
            pulse_e=result_e.mean(axis=0),
            pulse_g_org=result_g,
            pulse_e_org=result_e,
            signal_g=s11_g,
            signal_e=s11_e,
        )

finally:
    off_comm()
    print("finished")