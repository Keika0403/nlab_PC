from calendar import timegm
from qcodes import initialise_or_create_database_at, load_or_create_experiment, Measurement
from qcodes.instrument.parameter import Parameter
from qcodes.utils.validators import ComplexNumbers
from tqdm import tqdm
from TD_setups_for_tomography import *

database_name="tomography"
initialise_or_create_database_at(db_dir+f"\\{database_name}.db")
with open(__file__) as file:
    script = file.read()

cycles = 60000
# repetition = 10
hvi_trigger.digitizer_delay(400)
hvi_trigger.trigger_period(150000) # ns

freq, F = fourier_tr(time, target)
r_omega = s11_g_(freq)
waveform_AWG_coherent_state_g = np.fft.ifft(F / r_omega).real * 0.9
r_omega = s11_e_(freq)
waveform_AWG_coherent_state_e = np.fft.ifft(F / r_omega).real * 0.03
drive_length = len(time)

var = Variables()
amplitude = Variable("amplitude", np.append([0], np.linspace(0.8, 0.95, 31)), "V")
var.add(amplitude)
var.compile()

seq_JPA = Sequence(port_list=[JPA_port])
seq_JPA.add(SetDetuning(JPA_if_freq_for_photon-JPA_port.if_freq), JPA_port)
seq_JPA.add(Square(amplitude=amplitude, duration=drive_length+160), JPA_port, copy=False)

seq_g = Sequence(port_list=ports)
seq_g.call(readout_JPA_seq)
seq_g.compile()
jpa_g = JPA_port.waveform
seq_g.add(Delay(100), readout_port)
seq_g.trigger(ports_wo_JPA)
seq_g.compile()
readout_g = readout_port.waveform.real
# print(readout_g.shape)
# seq_g.trigger(ports)
seq_g.add(Square(amplitude=amplitude, duration=drive_length+160), JPA_port, copy=False)
seq_g.add(Acquire(drive_length), dig_port)

# seq_JPA.update_variables(var.update_command_list[2])
# seq_JPA.compile()
# jpa_pulse = JPA_port.waveform

# seq_g.update_variables(var.update_command_list[2])
# seq_g.compile()
# i_qubit_g, q_qubit_g = iq_corrector_q.correct(qubit_drive_port.waveform.conj())
# i_jpa_g, q_jpa_g = iq_corrector_JPA.correct(np.append(jpa_g, jpa_pulse).conj())
# plt.plot(np.append(readout_g, waveform_AWG_coherent_state_g))
# plt.plot(i_jpa_g)
# plt.plot(i_qubit_g)
# plt.show()
# seq_g.draw()
# raise SystemError

seq_e = Sequence(port_list=ports)
seq_e.call(ge_pi_seq)
seq_e.trigger(ports)
seq_e.call(readout_JPA_seq)
seq_e.compile()
readout_e = readout_port.waveform.real
jpa_e = JPA_port.waveform
seq_e.trigger(ports)
seq_e.add(Square(amplitude=amplitude, duration=drive_length), JPA_port, copy=False)
seq_e.add(Acquire(drive_length), dig_port)

measurement_name = f"trigger delay {hvi_trigger.digitizer_delay()}"
experiment_name = "JPA gain vs qubit state"
exp = load_or_create_experiment(experiment_name, sample_name)
meas = Measurement(exp, station, measurement_name)
waveform_g_param = Parameter(name='waveform_g', label='waveform_g', unit='V', set_cmd=None, get_cmd=None)
waveform_e_param = Parameter(name='waveform_e', label='waveform_e', unit='V', set_cmd=None, get_cmd=None)
time_param = Parameter(name='time', label='Time', unit='ns', set_cmd=None, get_cmd=None)
amplitude_param = Parameter(name='amplitude', label='amplitude', unit='V', set_cmd=None, get_cmd=None)
meas.register_parameter(time_param, paramtype="array")
meas.register_parameter(amplitude_param, paramtype="array")
meas.register_parameter(waveform_g_param, setpoints=[amplitude_param, time_param], paramtype="array")
meas.register_parameter(waveform_e_param, setpoints=[amplitude_param, time_param], paramtype="array")
dig_ch.cycles(cycles)
# lo1.output(True)
def queue_waveform_all(cycles=cycles):
    awg_readout_I.queue_waveform(0, trigger="software/hvi", cycles=cycles)
    awg_JPA_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
    awg_JPA_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
    awg_Qdrive_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
    awg_Qdrive_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)

try:
    with meas.run() as datasaver:
        datasaver.dataset.add_metadata("wiring", wiring)
        datasaver.dataset.add_metadata("setup_script", setup_script)
        datasaver.dataset.add_metadata("script", script)
        for update_command in tqdm(var.update_command_list):
            seq_JPA.update_variables(update_command)
            seq_JPA.compile()
            jpa_pulse = JPA_port.waveform

            seq_g.update_variables(update_command)
            seq_g.compile()
            i_qubit_g, q_qubit_g = iq_corrector_q.correct(qubit_drive_port.waveform.conj())
            measurement_windows_g = seq_g.get_waveform_information()[dig_port.name]["measurement_windows"]
            
            seq_e.update_variables(update_command)
            seq_e.compile()
            i_qubit_e, q_qubit_e = iq_corrector_q.correct(qubit_drive_port.waveform.conj())
            measurement_windows_e = seq_e.get_waveform_information()[dig_port.name]["measurement_windows"]

            for mode in ["plus_I", "plus_Q", "minus_I", "minus_Q"]:
                awg1.flush_waveform()
                awg2.flush_waveform()
                dig_ch.delay(int(measurement_windows_g[0][0] // dig_ch.sampling_interval()))
                dig_ch.points_per_cycle(int(measurement_windows_g[-1][-1] - measurement_windows_g[0][0]) 
                                    // dig_ch.sampling_interval())
                if mode == "plus_I":
                    awg1_pulse = np.append(readout_g, waveform_AWG_coherent_state_g)
                    i_jpa_g, q_jpa_g = iq_corrector_JPA.correct(np.append(jpa_g, jpa_pulse).conj())
                    # plt.plot(awg1_pulse)
                    # plt.plot(i_jpa_g)
                    # plt.plot(i_qubit_g)
                    # plt.show()
                    awg1.load_waveform(awg1_pulse, 0, append_zeros=True)
                    awg1.load_waveform(i_jpa_g, 1, append_zeros=True)
                    awg1.load_waveform(q_jpa_g, 2, append_zeros=True)
                    awg2.load_waveform(i_qubit_g, 1, append_zeros=True)
                    awg2.load_waveform(q_qubit_g, 2, append_zeros=True)
                    queue_waveform_all()
                elif mode == "plus_Q":
                    awg1_pulse = np.append(readout_g, waveform_AWG_coherent_state_g)
                    i_jpa_g, q_jpa_g = iq_corrector_JPA.correct(np.append(jpa_g, -jpa_pulse).conj())
                    awg1.load_waveform(awg1_pulse, 0, append_zeros=True)
                    awg1.load_waveform(i_jpa_g, 1, append_zeros=True)
                    awg1.load_waveform(q_jpa_g, 2, append_zeros=True)
                    awg2.load_waveform(i_qubit_g, 1, append_zeros=True)
                    awg2.load_waveform(q_qubit_g, 2, append_zeros=True)
                    queue_waveform_all()
                elif mode == "minus_I":
                    awg1_pulse = np.append(readout_g, -waveform_AWG_coherent_state_g)
                    i_jpa_g, q_jpa_g = iq_corrector_JPA.correct(np.append(jpa_g, jpa_pulse).conj())
                    awg1.load_waveform(awg1_pulse, 0, append_zeros=True)
                    awg1.load_waveform(i_jpa_g, 1, append_zeros=True)
                    awg1.load_waveform(q_jpa_g, 2, append_zeros=True)
                    awg2.load_waveform(i_qubit_g, 1, append_zeros=True)
                    awg2.load_waveform(q_qubit_g, 2, append_zeros=True)
                    queue_waveform_all()
                elif mode == "minus_Q":
                    awg1_pulse = np.append(readout_g, -waveform_AWG_coherent_state_g)
                    i_jpa_g, q_jpa_g = iq_corrector_JPA.correct(np.append(jpa_g, -jpa_pulse).conj())
                    awg1.load_waveform(awg1_pulse, 0, append_zeros=True)
                    awg1.load_waveform(i_jpa_g, 1, append_zeros=True)
                    awg1.load_waveform(q_jpa_g, 2, append_zeros=True)
                    awg2.load_waveform(i_qubit_g, 1, append_zeros=True)
                    awg2.load_waveform(q_qubit_g, 2, append_zeros=True)
                    queue_waveform_all()
                awg1.start_all()
                awg2.start_all()
                dig_ch.start()
                hvi_trigger.output(True)
                data = dig_ch.read()* voltage_step
                hvi_trigger.output(False)
                dig_ch.stop()
                # plt.plot(data.mean(axis=0))
                # plt.plot(mean_g)
                # plt.plot(mean_e)
                # plt.show()

                data = postselection(data, 0, 40, state="g")[:, int((measurement_windows_g[1][0]-measurement_windows_g[0][0]) // dig_ch.sampling_interval()):
                                                             int((measurement_windows_g[1][1]-measurement_windows_g[0][0]) // dig_ch.sampling_interval())]
                print(data.shape, mode, "g")

                if mode=="plus_I": plus_I_g = data.mean(axis=0)
                if mode=="plus_Q": plus_Q_g = data.mean(axis=0)
                if mode=="minus_I": minus_I_g = data.mean(axis=0)
                if mode=="minus_Q": minus_Q_g = data.mean(axis=0)

                awg1.flush_waveform()
                dig_ch.delay(int(measurement_windows_e[0][0] // dig_ch.sampling_interval()))
                dig_ch.points_per_cycle(int(measurement_windows_e[-1][-1] - measurement_windows_e[0][0]) 
                                    // dig_ch.sampling_interval())
                if mode == "plus_I":
                    awg1_pulse = np.append(readout_e, waveform_AWG_coherent_state_e)
                    i_jpa_e, q_jpa_e = iq_corrector_JPA.correct(np.append(jpa_e, jpa_pulse).conj())
                    awg1.load_waveform(awg1_pulse, 0, append_zeros=True)
                    awg1.load_waveform(i_jpa_e, 1, append_zeros=True)
                    awg1.load_waveform(q_jpa_e, 2, append_zeros=True)
                    awg2.load_waveform(i_qubit_e, 1, append_zeros=True)
                    awg2.load_waveform(q_qubit_e, 2, append_zeros=True)
                    queue_waveform_all()
                elif mode == "plus_Q":
                    awg1_pulse = np.append(readout_e, waveform_AWG_coherent_state_e)
                    i_jpa_e, q_jpa_e = iq_corrector_JPA.correct(np.append(jpa_e, -jpa_pulse).conj())
                    awg1.load_waveform(awg1_pulse, 0, append_zeros=True)
                    awg1.load_waveform(i_jpa_e, 1, append_zeros=True)
                    awg1.load_waveform(q_jpa_e, 2, append_zeros=True)
                    awg2.load_waveform(i_qubit_e, 1, append_zeros=True)
                    awg2.load_waveform(q_qubit_e, 2, append_zeros=True)
                    queue_waveform_all()
                elif mode == "minus_I":
                    awg1_pulse = np.append(readout_e, -waveform_AWG_coherent_state_e)
                    i_jpa_e, q_jpa_e = iq_corrector_JPA.correct(np.append(jpa_e, jpa_pulse).conj())
                    awg1.load_waveform(awg1_pulse, 0, append_zeros=True)
                    awg1.load_waveform(i_jpa_e, 1, append_zeros=True)
                    awg1.load_waveform(q_jpa_e, 2, append_zeros=True)
                    awg2.load_waveform(i_qubit_e, 1, append_zeros=True)
                    awg2.load_waveform(q_qubit_e, 2, append_zeros=True)
                    queue_waveform_all()
                elif mode == "minus_Q":
                    awg1_pulse = np.append(readout_e, -waveform_AWG_coherent_state_e)
                    i_jpa_e, q_jpa_e = iq_corrector_JPA.correct(np.append(jpa_e, -jpa_pulse).conj())
                    awg1.load_waveform(awg1_pulse, 0, append_zeros=True)
                    awg1.load_waveform(i_jpa_e, 1, append_zeros=True)
                    awg1.load_waveform(q_jpa_e, 2, append_zeros=True)
                    awg2.load_waveform(i_qubit_e, 1, append_zeros=True)
                    awg2.load_waveform(q_qubit_e, 2, append_zeros=True)
                    queue_waveform_all()
                awg1.start_all()
                awg2.start_all()
                dig_ch.start()
                hvi_trigger.output(True)
                data = dig_ch.read()* voltage_step
                hvi_trigger.output(False)
                dig_ch.stop()
                # plt.plot(data.mean(axis=0))
                # plt.plot(mean_g)
                # plt.show()

                data = postselection(data, 0, 40, state="e")[:, int(measurement_windows_e[1][0]-measurement_windows_e[0][0] // dig_ch.sampling_interval()):
                                                             int(measurement_windows_e[1][1]-measurement_windows_e[0][0] // dig_ch.sampling_interval())]
                print(data.shape, mode, "e")

                if mode=="plus_I": plus_I_e = data.mean(axis=0)
                if mode=="plus_Q": plus_Q_e = data.mean(axis=0)
                if mode=="minus_I": minus_I_e = data.mean(axis=0)
                if mode=="minus_Q": minus_Q_e = data.mean(axis=0)

            waveform_g = ((plus_I_g - minus_I_g) / 2 + (plus_Q_g - minus_Q_g) / 2) / 2
            waveform_e = ((plus_I_e - minus_I_e) / 2 + (plus_Q_e - minus_Q_e) / 2) / 2
            datasaver.add_result(
                (amplitude_param, seq_g.variable_dict["amplitude"][0].value),
                (time_param, np.arange(len(waveform_g))*dig_ch.sampling_interval()), 
                (waveform_g_param, waveform_g),
                (waveform_e_param, waveform_e),
            )
            dig_ch.stop()
except:print('An error occurred')
finally:
    off()
    print('finished')