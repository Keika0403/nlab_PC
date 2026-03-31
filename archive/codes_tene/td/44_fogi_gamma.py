from setup_td import *

measurement_name = os.path.basename(__file__)[:-3]
tags.append("photon")
tags.append("waveform")
hvi_trigger.trigger_period(30000)

def photon_generation_freq_sweep_JPA(
        fogi_amplitude=0.5,
        fogi_drive_freqs=np.linspace(5.45, 5.75, 26),
        fogi_drive_duration=1000,
        cycles=50000, 
        repetition=10, 
        acquisition_period=1000,
):
    def fogi_sequence(half_pi:bool, JPA_direction=1) -> Sequence:
        assert JPA_direction == -1 or 1
        seq = Sequence(port_list=ports)
        seq.add(Square(amplitude=JPA_direction*1.0, duration=fogi_drive_duration+100), JPA_port)
        if half_pi:
            seq.call(ge_half_pi_seq)
        else:
            seq.add(VirtualZ(np.pi), qubit_drive_port)
            seq.call(ge_half_pi_seq)
            seq.add(VirtualZ(-np.pi), qubit_drive_port)
        seq.call(ef_pi_seq)
        seq.trigger(ports_wo_JPA)
        seq.add(Square(amplitude=fogi_amplitude, duration=fogi_drive_duration), fogi_port)
        seq.add(Acquire(acquisition_period), dig_port)
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
                fogi_port.if_freq = fogi_drive_freq - fogi_lo_freq
                g_plus_e_I=[]
                g_plus_e_Q=[]
                g_minus_e_I=[]
                g_minus_e_Q=[]
                for _ in  range(repetition):
                    for state in ["0+1_i", "0+1_q", "0-1_i", "0-1_q"]:
                        awg1.flush_waveform()
                        awg2.flush_waveform()
                        if state=="0+1_i":
                            seq = fogi_sequence(half_pi=True, JPA_direction=1)
                            # seq.draw()
                            # raise SystemError
                        if state=="0+1_q":
                            seq = fogi_sequence(half_pi=True, JPA_direction=-1)
                        if state=="0-1_i":
                            seq = fogi_sequence(half_pi=False, JPA_direction=1)
                        if state=="0-1_q":
                            seq = fogi_sequence(half_pi=False, JPA_direction=-1)
                        load_sequence(seq, cycles=cycles)
                        data = run(seq).mean(axis=0) * voltage_step
                        if state=="0+1_i": g_plus_e_I = np.append(g_plus_e_I, data)
                        if state=="0+1_q": g_plus_e_Q = np.append(g_plus_e_Q, data)
                        if state=="0-1_i": g_minus_e_I = np.append(g_minus_e_I, data)
                        if state=="0-1_q": g_minus_e_Q = np.append(g_minus_e_Q, data)
                g_plus_e_I = g_plus_e_I.reshape(int(repetition), int(dig_ch.points_per_cycle())).mean(axis=0)
                g_plus_e_Q = g_plus_e_Q.reshape(int(repetition), int(dig_ch.points_per_cycle())).mean(axis=0)
                g_minus_e_I = g_minus_e_I.reshape(int(repetition), int(dig_ch.points_per_cycle())).mean(axis=0)
                g_minus_e_Q = g_minus_e_Q.reshape(int(repetition), int(dig_ch.points_per_cycle())).mean(axis=0)
                waveform_I = (g_plus_e_I - g_minus_e_I) / 2
                waveform_Q = (g_plus_e_Q - g_minus_e_Q) / 2
                waveform = (waveform_I + waveform_Q) / 2
                writer.add_data(
                    fogi_frequency=fogi_drive_freq,
                    time=np.arange(len(waveform)) * dig_ch.sampling_interval(),
                    waveform=waveform,
                    g_plus_e_I=g_plus_e_I,
                    g_minus_e_I=g_minus_e_I,
                    g_plus_e_Q=g_plus_e_Q,
                    g_minus_e_Q=g_minus_e_Q,
                )
    finally:
        off()
        print('finished')

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.05, 
#                                  fogi_drive_freqs=np.linspace(4.57, 4.67, 26), #4.62
#                                  fogi_drive_duration=20000,
#                                  acquisition_period=6000, repetition=20,)

photon_generation_freq_sweep_JPA(fogi_amplitude=0.1, 
                                 fogi_drive_freqs=np.linspace(5.6, 5.7, 26), #5.63
                                 fogi_drive_duration=20000,
                                 acquisition_period=6000, repetition=20,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.15, 
#                                  fogi_drive_freqs=np.linspace(5.595, 5.695, 26),
#                                  fogi_drive_duration=20000,
#                                  acquisition_period=6000, repetition=20,)

photon_generation_freq_sweep_JPA(fogi_amplitude=0.2, 
                                 fogi_drive_freqs=np.linspace(5.59, 5.69, 26),#5.64
                                 fogi_drive_duration=20000, 
                                 acquisition_period=6000, repetition=20,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.25,
#                                  fogi_drive_freqs=np.linspace(5.59, 5.69, 26),
#                                  fogi_drive_duration=20000,
#                                  acquisition_period=5000, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplit_freqs=np.linspace(5.57, 5.67, 26), #5.62
#                                  fogi_drive_duration=20000,
#                                  acquisitiude=0.3,
#                                  fogi_driveon_period=5000, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.35,
#                                  fogi_drive_freqs=np.linspace(5.585, 5.685, 26),
#                                  fogi_drive_duration=20000,
#                                  acquisition_period=5000, repetition=10,)

photon_generation_freq_sweep_JPA(fogi_amplitude=0.4, 
                                 fogi_drive_freqs=np.linspace(5.58, 5.68, 26),#5.63   4.66, 26),
                                 fogi_drive_duration=10000,
                                 acquisition_period=5000, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.45, 
#                                  fogi_drive_freqs=np.linspace(5.58, 5.68, 26),
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=4000, repetition=10)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.5, 
#                                  fogi_drive_freqs=np.linspace(5.57, 5.67, 26), #5.6
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=4000, repetition=10)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.55, 
#                                  fogi_drive_freqs=np.linspace(5.575, 5.675, 31),
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=4000, repetition=10)

photon_generation_freq_sweep_JPA(fogi_amplitude=0.6, 
                                 fogi_drive_freqs=np.linspace(5.565, 5.665, 26), #5.615
                                 fogi_drive_duration=10000,
                                 acquisition_period=4000, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.65, 
#                                  fogi_drive_freqs=np.linspace(4.54, 4.64, 26),
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=4000, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.7, 
#                                  fogi_drive_freqs=np.linspace(5.555, 5.655, 26),
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=3000, repetition=10,)

photon_generation_freq_sweep_JPA(fogi_amplitude=0.8, 
                                 fogi_drive_freqs=np.linspace(5.55, 5.65, 26),
                                 fogi_drive_duration=10000,
                                 acquisition_period=3000, repetition=10,)

photon_generation_freq_sweep_JPA(fogi_amplitude=0.9, 
                                 fogi_drive_freqs=np.linspace(5.55, 5.65, 26),
                                 fogi_drive_duration=10000,
                                 acquisition_period=2000, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=0.95, 
#                                  fogi_drive_freqs=np.linspace(5.52, 5.66, 26),
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=1000, repetition=10,)

photon_generation_freq_sweep_JPA(fogi_amplitude=1.0, 
                                 fogi_drive_freqs=np.linspace(5.54, 5.64, 26), #5.59
                                 fogi_drive_duration=10000,
                                 acquisition_period=2000, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=1.05, 
#                                  fogi_drive_freqs=np.linspace(5.22, 5.68, 26),
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=1000, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=1.075, 
#                                  fogi_drive_freqs=np.linspace(5.48, 5.68, 26),
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=1000, repetition=10,)

photon_generation_freq_sweep_JPA(fogi_amplitude=1.1, 
                                 fogi_drive_freqs=np.linspace(5.52, 5.62, 26),
                                 fogi_drive_duration=10000,
                                 acquisition_period=1500, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=1.125, 
#                                  fogi_drive_freqs=np.linspace(5.48, 5.68, 26),
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=1000, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=1.15, 
#                                  fogi_drive_freqs=np.linspace(5.45, 5.7, 26),
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=1000, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=1.175, 
#                                  fogi_drive_freqs=np.linspace(5.45, 5.7, 26),
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=1000, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=1.2, 
#                                  fogi_drive_freqs=np.linspace(5.22, 5.32, 26),
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=1000, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=1.25, 
#                                  fogi_drive_freqs=np.linspace(4.92, 5.07, 26),
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=1000, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=1.3, 
#                                  fogi_drive_freqs=np.linspace(5.18, 5.28, 26),
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=1000, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=1.35, 
#                                  fogi_drive_freqs=np.linspace(5.45, 5.65, 26),
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=1500, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=1.4, 
#                                  fogi_drive_freqs=np.linspace(4.91, 5.05, 26),
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=800, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=1.45, 
#                                  fogi_drive_freqs=np.linspace(4.91, 5.05, 26),
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=800, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=1.48, 
#                                  fogi_drive_freqs=np.linspace(4.91, 5.05, 26),
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=1500, repetition=10,)

# photon_generation_freq_sweep_JPA(fogi_amplitude=1.49, 
#                                  fogi_drive_freqs=np.linspace(4.9, 5.05, 26),
#                                  fogi_drive_duration=10000,
#                                  acquisition_period=1500, repetition=10,)