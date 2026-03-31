from setup_td import *
from setup_td_tomography import *
from time_reverse import *

measurement_name = os.path.basename(__file__)[:-3]
tags.append("single_shot")

shot_count = 50000
repetition = 5
hvi_trigger.trigger_period(100000)

fogi_delay = 30
fogi_amp = 0.8
fogi_duration = 800
fogi_freq = 5.3405

ph_phase = [0, np.pi/2, np.pi, -np.pi/2]

# ph_amp =  [0.195]#[0.053, 0.098, 0.138, 0.169, 0.195] # id=0
# ph_amp = [0.147]#[0.041, 0.074, 0.105, 0.128, 0.147] # id=1
# ph_amp =  [0.189]#[0.052, 0.095, 0.134, 0.164, 0.189] # id=2
ph_amp =  [0.184]#[0.051, 0.093, 0.13, 0.159, 0.184] # id=3
# ph_amp =  [0.213]#[0.058, 0.107, 0.15, 0.184, 0.213] # id=4


## photon shape ### 
id = 3
duration = 520
header = "D:/K_sunada/result/CDY155/"
data = "/2024-02-18/2024-02-18T171640_0ad3b17a-70_JPA_photon_generation"
dd = datadict_from_hdf5(header+data+"/data")
x = dd['time']['values'][id][0:duration]
y = dd['waveform']['values'][id][0:duration]
ph_if = [62.5e6, 72.5e6, 82.5e6, 90e6, 95e6]
y_shift =  y[13:duration-7]* np.exp(-1j*2 *np.pi* ph_if[id]*(x[13:duration-7]*1e-9))
y_LPF = np.array(np.abs(LPF(y_shift, 500e6, 15e6, 40e6, 5, 40))*2)

x = x[0+13:250+13]
y = y[0+13:250+13]*1e2
amp = [0.00571881, 0.01057035, 0.00660134, 0.00632839,0.00444275]
gamma = [0.01426486, 0.02273654, 0.01688674, 0.01522837, 0.01104741]
x = np.linspace(0, 799, 800)
y_env = amp[id]*np.exp(-(gamma[id]/2)*(x+26))*1e2
time_reversed_waveform = (y_env*np.cos(2*np.pi* ph_if[id]*(x*1e-9)))[::-1]
## photon shape ### 



control_pulse = time_reversed_waveform  * ph_amp 

##
sequence_x = Sequence(port_list=ports)
sequence_x.call(readout_single_shot_seq)
sequence_x.trigger(ports)

sequence_x.add(Delay(fogi_delay), fogi_port)
sequence_x.add(Square(amplitude=fogi_amp, duration=fogi_duration), fogi_port)
sequence_x.add(Acquire(len(control_pulse)), readout_port)

sequence_x.trigger(ports)
sequence_x.call(ef_pi_seq)
sequence_x.add(Delay(4), qubit_drive_port)
sequence_x.trigger(ports)
sequence_x.add(ResetPhase(0), qubit_drive_port, copy=False)
sequence_x.call(ge_half_pi_seq)
sequence_x.call(readout_single_shot_seq)

# sequence_x.draw()
# raise SystemError

##
sequence_y = Sequence(port_list=ports)
sequence_y.call(readout_single_shot_seq)
sequence_y.trigger(ports)

sequence_y.add(Delay(fogi_delay), fogi_port)
sequence_y.add(Square(amplitude=fogi_amp, duration=fogi_duration), fogi_port)
sequence_y.add(Acquire(len(control_pulse)), readout_port)

sequence_y.trigger(ports)
sequence_y.call(ef_pi_seq)
sequence_y.add(Delay(4), qubit_drive_port)
sequence_y.trigger(ports)
sequence_y.add(ResetPhase(np.pi/2), qubit_drive_port, copy=False)
sequence_y.call(ge_half_pi_seq)
sequence_y.call(readout_single_shot_seq)
# sequence_y.draw()
# raise SystemError

##
sequence_z = Sequence(port_list=ports)
sequence_z.call(readout_single_shot_seq)
sequence_z.trigger(ports)

sequence_z.add(Delay(fogi_delay), fogi_port)
sequence_z.add(Square(amplitude=fogi_amp, duration=fogi_duration), fogi_port)
sequence_z.add(Acquire(len(control_pulse)), readout_port)

sequence_z.trigger(ports)
sequence_z.call(ef_pi_seq)
sequence_z.add(Delay(6), qubit_drive_port)
sequence_z.trigger(ports)
sequence_z.call(readout_single_shot_seq)

# sequence_z.draw()
# raise SystemError
data = DataDict(
    signal_x_0=dict(),
    signal_y_0=dict(),
    signal_z_0=dict(),
    signal_x_pi_2=dict(),
    signal_y_pi_2=dict(),
    signal_z_pi_2=dict(),
    signal_x_pi=dict(),
    signal_y_pi=dict(),
    signal_z_pi=dict(),
    signal_x_pi_2_mi=dict(),
    signal_y_pi_2_mi=dict(),
    signal_z_pi_2_mi=dict(),
)
data.validate()

try:
    with DDH5Writer(data, data_path, name=measurement_name) as writer:
        writer.add_tag(tags)
        writer.backup_file([__file__, setup_file])
        writer.save_text("wiring.md", wiring)
        writer.save_dict("station_snapshot.json", station.snapshot())
        signal_xs = []
        signal_ys = []
        signal_zs = []
        fogi_port.if_freq = fogi_freq - fogi_lo_freq
        for theta in tqdm(ph_phase):
            x = np.linspace(0, 799, 800)
            y_env = amp[id]*np.exp(-(gamma[id]/2)*(x+26))*1e2
            time_reversed_waveform = (y_env*np.cos(2*np.pi* ph_if[id]*(x*1e-9)+theta))[::-1]
            control_pulse = time_reversed_waveform  * ph_amp 
              
            result_x = []
            result_y = []
            result_z = []
            for _ in tqdm(range(repetition)):
                load_sequence_w_append(sequence_x, append_port=readout_port, waveform_appended=control_pulse, cycles=shot_count)
                pulse_x = run(sequence_x, plot=0) * voltage_step
                result_x.append(pulse_x)
                post_selection_start_x = int(dig_port.measurement_windows[1][0] // 2 )
                
                load_sequence_w_append(sequence_y, append_port=readout_port, waveform_appended=control_pulse, cycles=shot_count)
                pulse_y = run(sequence_y, plot=0) * voltage_step
                result_y.append(pulse_y)
                post_selection_start_y = int(dig_port.measurement_windows[1][0] // 2 )

                load_sequence_w_append(sequence_z, append_port=readout_port, waveform_appended=control_pulse, cycles=shot_count)
                pulse_z = run(sequence_z, plot=0) * voltage_step
                result_z.append(pulse_z)
                post_selection_start_z = int(dig_port.measurement_windows[1][0] // 2 )
            result_x = np.array(result_x).reshape(int(repetition*shot_count), len(pulse_x[0]))
            result_y = np.array(result_y).reshape(int(repetition*shot_count), len(pulse_y[0]))
            result_z = np.array(result_z).reshape(int(repetition*shot_count), len(pulse_z[0]))
            # plt.plot(result_z.mean(axis=0))

            ##### anlyzation
            delete_idx_x = []
            delete_idx_y = []
            delete_idx_z = []
            for i in range(len(result_z)):
                single_shot_x = result_x[i][0:len(mean_g)]
                single_shot_y = result_y[i][0:len(mean_g)]
                single_shot_z = result_z[i][0:len(mean_g)]
                if np.dot(single_shot_x - mean_g, ein_vec) > 1/2: # eliminate e state
                    delete_idx_x.append(i)
                if np.dot(single_shot_y - mean_g, ein_vec) > 1/2: # eliminate e state
                    delete_idx_y.append(i)
                if np.dot(single_shot_z - mean_g, ein_vec) > 1/2: # eliminate e state
                    delete_idx_z.append(i)
            result_x = np.delete(result_x, delete_idx_x, axis=0)[:, post_selection_start_x:post_selection_start_x+len(mean_g)]
            result_y = np.delete(result_y, delete_idx_y, axis=0)[:, post_selection_start_y:post_selection_start_y+len(mean_g)]
            result_z = np.delete(result_z, delete_idx_z, axis=0)[:, post_selection_start_z:post_selection_start_z+len(mean_g)]
            # print(result_x.shape, result_y.shape)
            # print(result_z.shape)
            # plt.plot(result_z.mean(axis=0))
            # plt.show()
            signal_x = np.dot(result_x - mean_g, ein_vec)
            signal_y = np.dot(result_y - mean_g, ein_vec)
            signal_z = np.dot(result_z - mean_g, ein_vec)
            print(signal_x.shape, signal_y.shape, signal_z.shape)
            signal_xs.append(signal_x)
            signal_ys.append(signal_y)
            signal_zs.append(signal_z)
            
        writer.add_data(
            signal_x_0=signal_xs[0],
            signal_y_0=signal_ys[0],
            signal_z_0=signal_zs[0],
            signal_x_pi_2=signal_xs[1],
            signal_y_pi_2=signal_ys[1],
            signal_z_pi_2=signal_zs[1],
            signal_x_pi=signal_xs[2],
            signal_y_pi=signal_ys[2],
            signal_z_pi=signal_zs[2],
            signal_x_pi_2_mi=signal_xs[3],
            signal_y_pi_2_mi=signal_ys[3],
            signal_z_pi_2_mi=signal_zs[3],
        )
finally:
    off()
    print("finished")
    