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
repetition = 10
hvi_trigger.digitizer_delay(400)
hvi_trigger.trigger_period(100000) # ns

freq, F = fourier_tr(time, target)
r_omega = s11_g_(freq)
waveform_AWG_coherent_state_g = np.fft.ifft(F / r_omega).real * 0.03
drive_length = len(time)

var = Variables()
amplitude = Variable("amplitude", np.append([0], np.linspace(0.8, 0.95, 4)), "V")
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
# seq_g.trigger(ports)
seq_g.add(Square(amplitude=0.5, duration=drive_length+160), JPA_port, copy=False)
seq_g.add(Acquire(drive_length), dig_port)
seq_g.compile()
i_qubit_g, q_qubit_g = iq_corrector_q.correct(qubit_drive_port.waveform.conj())
measurement_windows_g = seq_g.get_waveform_information()[dig_port.name]["measurement_windows"]
# seq_g.draw()
# measurement_windows_g = seq_g.get_waveform_information()[dig_port.name]["measurement_windows"]
# print(measurement_windows_g)
# raise SystemError

# seq_JPA.update_variables(var.update_command_list[2])
# seq_JPA.compile()
# jpa_pulse = JPA_port.waveform

# # seq_g.update_variables(var.update_command_list[2])
# seq_g.compile()
# i_qubit_g, q_qubit_g = iq_corrector_q.correct(qubit_drive_port.waveform.conj())
# i_jpa_g, q_jpa_g = iq_corrector_JPA.correct(np.append(jpa_g, jpa_pulse).conj())
# plt.plot(np.append(readout_g, waveform_AWG_coherent_state_g))
# plt.plot(i_jpa_g)
# plt.plot(i_qubit_g)
# plt.show()
# seq_g.draw()
# raise SystemError

measurement_name = f"target_freq = {target_freq}"
experiment_name = "JPA gain when qubit state = g"
exp = load_or_create_experiment(experiment_name, sample_name)
meas = Measurement(exp, station, measurement_name)
waveform_g_param = Parameter(name='waveform_g', label='waveform_g', unit='V', set_cmd=None, get_cmd=None)
time_param = Parameter(name='time', label='Time', unit='ns', set_cmd=None, get_cmd=None)
amplitude_param = Parameter(name='amplitude', label='amplitude', unit='V', set_cmd=None, get_cmd=None)
meas.register_parameter(time_param, paramtype="array")
meas.register_parameter(amplitude_param, paramtype="array")
meas.register_parameter(waveform_g_param, setpoints=[amplitude_param, time_param], paramtype="array")
dig_ch.cycles(cycles)

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

            plus_I_g = []
            plus_Q_g = []
            minus_I_g = []
            minus_Q_g = []
            for mode in ["plus_I", "plus_Q", "minus_I", "minus_Q"]:
                box = []
                for _ in range(repetition):
                    awg1.flush_waveform()
                    awg2.flush_waveform()
                    dig_ch.delay(int(measurement_windows_g[0][0] // dig_ch.sampling_interval()))
                    dig_ch.points_per_cycle(int(measurement_windows_g[-1][-1] - measurement_windows_g[0][0]) 
                                        // dig_ch.sampling_interval())
                    if mode == "plus_I":
                        awg1_pulse = np.append(readout_g, waveform_AWG_coherent_state_g)
                        i_jpa_g, q_jpa_g = iq_corrector_JPA.correct(np.append(jpa_g, jpa_pulse).conj())
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
                    # plt.plot(np.append(data.mean(axis=0)))
                    # plt.plot(mean_g)
                    # plt.plot(mean_e)
                    # plt.show()
                    data = postselection(data, 0, 40, state="g")
                    data = data[:, int((measurement_windows_g[1][0]-measurement_windows_g[0][0]) // dig_ch.sampling_interval()):
                                                                int((measurement_windows_g[1][1]-measurement_windows_g[0][0]) // dig_ch.sampling_interval())]
                    
                    box = np.append(box, data)
                box = box.reshape(int(len(box) / data.shape[-1]), data.shape[-1])
                print(box.shape, mode, "g")
                if mode=="plus_I": plus_I_g = box.mean(axis=0)
                if mode=="plus_Q": plus_Q_g = box.mean(axis=0)
                if mode=="minus_I": minus_I_g =  box.mean(axis=0)
                if mode=="minus_Q": minus_Q_g =  box.mean(axis=0)
            waveform_g = ((plus_I_g - minus_I_g) / 2 + (plus_Q_g - minus_Q_g) / 2) / 2
            datasaver.add_result(
                (amplitude_param, seq_JPA.variable_dict["amplitude"][0].value),
                (time_param, np.arange(len(waveform_g))*dig_ch.sampling_interval()), 
                (waveform_g_param, waveform_g),
            )
            dig_ch.stop()
except:print('An error occurred')
finally:
    off()
    print('finished')