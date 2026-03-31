from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]
tags.append("waveform")
hvi_trigger.trigger_period(5000)
lo1.frequency(9.5e9)
JPA_port_tx.if_freq = (9.38 - 9.5) * 2
tags = ["TD", "MPC2-BW9500-1-601", "tx", "photon"]
JPA_amp =0.7

def photon_generation_freq_sweep_JPA(
        fogi_amplitude=1.,
        fogi_drive_freqs=np.linspace(5.45, 5.75, 21),
        fogi_drive_duration=1000,
        cycles=30000, 
        repetition=20, 
        acquisition_period=1000,
        JPA_amp=JPA_amp,
        degenerate=True,
):
    if measure_which == "tx":
        def fogi_sequence(half_pi:bool, JPA_direction=1) -> Sequence:
            assert JPA_direction == -1 or JPA_direction == 1
            seq = Sequence(port_list=ports_tx)
            seq.call(reset_sequence_tx)
            # seq.call(reset_sequence_rx)
            seq.trigger(ports_tx)
            seq.add(Square(amplitude=JPA_direction*JPA_amp, duration=fogi_drive_duration+100), JPA_port_tx)
            if half_pi:
                seq.call(ge_half_pi_seq_tx)
            else:
                seq.add(VirtualZ(np.pi), qubit_drive_port_tx)
                seq.call(ge_half_pi_seq_tx)
                seq.add(VirtualZ(-np.pi), qubit_drive_port_tx)
            seq.call(ef_pi_seq_tx)
            seq.trigger(ports_tx_wo_JPA)
            seq.add(Square(amplitude=fogi_amplitude, duration=fogi_drive_duration), fogi_port_tx)
            seq.add(Acquire(acquisition_period), dig_port_tx)
            return seq
    elif measure_which == "txQrx":
        def fogi_sequence(half_pi:bool, JPA_direction=1) -> Sequence:
            assert JPA_direction == -1 or JPA_direction == 1
            seq = Sequence(port_list=ports_txQrx)
            seq.call(reset_sequence_rx)
            seq.trigger(ports_txQrx)
            seq.add(Square(amplitude=JPA_direction*JPA_amp, duration=fogi_drive_duration+100), JPA_port_tx)
            if half_pi:
                seq.call(ge_half_pi_seq_rx)
            else:
                seq.add(VirtualZ(np.pi), qubit_drive_port_rx)
                seq.call(ge_half_pi_seq_rx)
                seq.add(VirtualZ(-np.pi), qubit_drive_port_rx)
            seq.call(ef_pi_seq_rx)
            seq.trigger(ports_txQrx_wo_JPA)
            seq.add(Square(amplitude=fogi_amplitude, duration=fogi_drive_duration), fogi_port_rx)
            seq.add(Acquire(acquisition_period), dig_port_tx)
            return seq
    elif measure_which == "rx":
        def fogi_sequence(half_pi:bool, JPA_direction=1) -> Sequence:
            assert JPA_direction == -1 or JPA_direction == 1
            seq = Sequence(port_list=ports_rx)
            seq.call(reset_sequence_rx)
            # seq.call(reset_sequence_rx)
            seq.trigger(ports_rx)
            seq.add(Square(amplitude=JPA_direction*JPA_amp, duration=fogi_drive_duration+100), JPA_port_tx)
            if half_pi:
                seq.call(ge_half_pi_seq_rx)
            else:
                seq.add(VirtualZ(np.pi), qubit_drive_port_rx)
                seq.call(ge_half_pi_seq_rx)
                seq.add(VirtualZ(-np.pi), qubit_drive_port_rx)
            seq.call(ef_pi_seq_rx)
            seq.trigger(ports_rx_wo_JPA)
            seq.add(Square(amplitude=fogi_amplitude, duration=fogi_drive_duration), fogi_port_rx)
            seq.add(Acquire(acquisition_period), dig_port_rx)
            return seq

    data = DataDict(
        fogi_frequency=dict(unit="GHz"),
        time=dict(unit="ns"),
        waveform=dict(axes=["fogi_frequency", "time"], unit="V"),
        g_plus_e_I=dict(axes=["fogi_frequency", "time"], unit="V"),
        g_minus_e_I=dict(axes=["fogi_frequency", "time"], unit="V"),
        g_plus_e_Q=dict(axes=["fogi_frequency", "time"], unit="V"),
        g_minus_e_Q=dict(axes=["fogi_frequency", "time"], unit="V"),
    )
    data.validate()
    try:
        with DDH5Writer(data, data_path, name=measurement_name) as writer:
            writer.add_tag(tags)
            writer.save_text("note.md", f"{fogi_amplitude} V")
            writer.backup_file([__file__, setup_file, setup_parameters_file])
            writer.save_text("wiring.md", wiring)
            writer.save_dict("station_snapshot.json", station.snapshot())
            for fogi_drive_freq in tqdm(fogi_drive_freqs):
                fogi_port_tx.if_freq = fogi_drive_freq - fogi_lo_freq_tx
                fogi_port_rx.if_freq = fogi_drive_freq - fogi_lo_freq_rx
                g_plus_e_I=[]
                g_plus_e_Q=[]
                g_minus_e_I=[]
                g_minus_e_Q=[]
                for _ in  range(repetition):
                    for state in ["0+1_i", "0+1_q", "0-1_i", "0-1_q"]:
                        awg1.flush_waveform()
                        awg2.flush_waveform()
                        awg3.flush_waveform()
                        if state=="0+1_i":
                            seq = fogi_sequence(half_pi=True, JPA_direction=1)
                            # seq.draw()
                            # raise SystemError
                        if state=="0+1_q":
                            seq = fogi_sequence(half_pi=True, JPA_direction=-2*degenerate+1)
                        if state=="0-1_i":
                            seq = fogi_sequence(half_pi=False, JPA_direction=1)
                        if state=="0-1_q":
                            seq = fogi_sequence(half_pi=False, JPA_direction=-2*degenerate+1)
                        load_sequence(seq, cycles=cycles)
                        data = run(seq, which=measure_which[:2]).mean(axis=0) * voltage_step_tx
                        if state=="0+1_i": g_plus_e_I = np.append(g_plus_e_I, data)
                        if state=="0+1_q": g_plus_e_Q = np.append(g_plus_e_Q, data)
                        if state=="0-1_i": g_minus_e_I = np.append(g_minus_e_I, data)
                        if state=="0-1_q": g_minus_e_Q = np.append(g_minus_e_Q, data)
                g_plus_e_I = g_plus_e_I.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
                g_plus_e_Q = g_plus_e_Q.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
                g_minus_e_I = g_minus_e_I.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
                g_minus_e_Q = g_minus_e_Q.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
                waveform_I = (g_plus_e_I - g_minus_e_I) / 2
                waveform_Q = (g_plus_e_Q - g_minus_e_Q) / 2
                waveform = (waveform_I + waveform_Q) / 2
                writer.add_data(
                    fogi_frequency=fogi_drive_freq,
                    time=np.arange(len(waveform)) * dig_ch_tx.sampling_interval(),
                    waveform=waveform,
                    g_plus_e_I=g_plus_e_I,
                    g_minus_e_I=g_minus_e_I,
                    g_plus_e_Q=g_plus_e_Q,
                    g_minus_e_Q=g_minus_e_Q,
                )
    finally:
        off()
        print('finished')
# ### tx
# photon_generation_freq_sweep_JPA(fogi_amplitude=0.1, 
#                                  fogi_drive_freqs=np.linspace(6.63, 6.73, 11), #6.73
#                                 fogi_drive_duration=5000,
#                                 acquisition_period=5000, repetition=25,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.2, 
#                                  fogi_drive_freqs=np.linspace(6.63, 6.73, 11), #6.73
#                                  fogi_drive_duration=4000, 
#                                  acquisition_period=4000, repetition=20,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.25, 
#                                  fogi_drive_freqs=np.linspace(6.63, 6.73, 11), #6.73
#                                  fogi_drive_duration=3000, 
#                                  acquisition_period=2000, repetition=20,)

photon_generation_freq_sweep_JPA(fogi_amplitude=0.275, 
                                 fogi_drive_freqs=np.linspace(6.635, 6.71, 16), #6.73
                                 fogi_drive_duration=3000, 
                                 acquisition_period=2000, repetition=15,)

photon_generation_freq_sweep_JPA(fogi_amplitude=0.3,
                                 fogi_drive_freqs=np.linspace(6.6, 6.7, 21),#6.65
                                 fogi_drive_duration=3000,
                                 acquisition_period=2000, repetition=20,)

photon_generation_freq_sweep_JPA(fogi_amplitude=0.325,
                                 fogi_drive_freqs=np.linspace(6.58, 6.68, 21),#5.36
                                 fogi_drive_duration=1500,
                                 acquisition_period=1000, repetition=15,)

photon_generation_freq_sweep_JPA(fogi_amplitude=0.34,
                                 fogi_drive_freqs=np.linspace(6.58, 6.68, 21),#5.36
                                 fogi_drive_duration=1500,
                                 acquisition_period=1000, repetition=15,)

photon_generation_freq_sweep_JPA(fogi_amplitude=0.35,
                                 fogi_drive_freqs=np.linspace(6.58, 6.68, 21),#5.36
                                 fogi_drive_duration=1500,
                                 acquisition_period=1000, repetition=15,)

photon_generation_freq_sweep_JPA(fogi_amplitude=0.36,
                                 fogi_drive_freqs=np.linspace(6.58, 6.68, 31),#5.36
                                 fogi_drive_duration=1500,
                                 acquisition_period=1000, repetition=15,)

photon_generation_freq_sweep_JPA(fogi_amplitude=0.365,
                                 fogi_drive_freqs=np.linspace(6.59, 6.68, 31),#5.36
                                 fogi_drive_duration=1500,
                                 acquisition_period=1000, repetition=15,)

photon_generation_freq_sweep_JPA(fogi_amplitude=0.37,
                                 fogi_drive_freqs=np.linspace(6.595, 6.67, 31),#5.36
                                 fogi_drive_duration=1500,
                                 acquisition_period=1000, repetition=15,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.375,
#                                  fogi_drive_freqs=np.linspace(6.56, 6.66, 21),#5.36
#                                  fogi_drive_duration=1500,
#                                  acquisition_period=1000, repetition=15,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.4, 
#                                  fogi_drive_freqs=np.linspace(6.55, 6.65, 21),  #6.59
#                                  fogi_drive_duration=3000,
#                                  acquisition_period=2000, repetition=15,)


### rx
# photon_generation_freq_sweep_JPA(fogi_amplitude=0.1, 
#                                  fogi_drive_freqs=np.linspace(6.18, 6.28, 11), #6.2
#                                 fogi_drive_duration=5000,
#                                 acquisition_period=5000, repetition=25,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.2, 
#                                  fogi_drive_freqs=np.linspace(6.17, 6.27, 11), #6.217
#                                  fogi_drive_duration=4000, 
#                                  acquisition_period=4000, repetition=20,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.25, 
#                                  fogi_drive_freqs=np.linspace(6.16, 6.26, 11), #6.217
#                                  fogi_drive_duration=3000, 
#                                  acquisition_period=2000, repetition=20,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.3,
#                                  fogi_drive_freqs=np.linspace(6.15, 6.25, 21),#6.2
#                                  fogi_drive_duration=3000,
#                                  acquisition_period=2000, repetition=20,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.325,
#                                  fogi_drive_freqs=np.linspace(6.15, 6.25, 21),#6.2
#                                  fogi_drive_duration=2000,
#                                  acquisition_period=1500, repetition=15,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.35,
#                                  fogi_drive_freqs=np.linspace(6.15, 6.25, 21),#
#                                  fogi_drive_duration=1500,
#                                  acquisition_period=1000, repetition=15,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.375,
#                                  fogi_drive_freqs=np.linspace(6.15, 6.25, 21),#
#                                  fogi_drive_duration=1500,
#                                  acquisition_period=1000, repetition=15,)


# photon_generation_freq_sweep_JPA(fogi_amplitude=0.4, 
#                                  fogi_drive_freqs=np.linspace(6.12, 6.22, 21),  #6.16
#                                  fogi_drive_duration=3000,
#                                  acquisition_period=2000, repetition=15,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.425, 
#                                  fogi_drive_freqs=np.linspace(6.12, 6.22, 21),  #6.16
#                                  fogi_drive_duration=3000,
#                                  acquisition_period=2000, repetition=15,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.45, 
#                                  fogi_drive_freqs=np.linspace(6.12, 6.22, 21),  #
#                                  fogi_drive_duration=2000,
#                                  acquisition_period=1000, repetition=15,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.475, 
#                                  fogi_drive_freqs=np.linspace(6.12, 6.22, 21),  #
#                                  fogi_drive_duration=2000,
#                                  acquisition_period=1000, repetition=15,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.5, 
#                                  fogi_drive_freqs=np.linspace(6.11, 6.21, 21), #
#                                  fogi_drive_duration=2000,
#                                  acquisition_period=1000, repetition=15)
