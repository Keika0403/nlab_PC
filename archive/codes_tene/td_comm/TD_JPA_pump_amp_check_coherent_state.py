from calendar import timegm
from qcodes import initialise_or_create_database_at, load_or_create_experiment, Measurement, load_by_run_spec
from qcodes.instrument.parameter import Parameter
from qcodes.utils.validators import ComplexNumbers
from TD_setups_JPA import *
from tqdm import tqdm

database_name="photon_generation"
with open(__file__) as file:
    script = file.read()

cycles = 50000
repetition = 3
hvi_trigger.digitizer_delay(400)
hvi_trigger.trigger_period(10000)

freq, F = fourier_tr(time, target)
r_omega = s11_g_(freq)
waveform_AWG_coherent_state = np.fft.ifft(F / r_omega).real
drive_length = len(time)+100

seq = Sequence([qubit_drive_port, fogi_port, dig_port])
seq.add(Acquire(drive_length+100), dig_port)
seq.compile()
measurement_windows = seq.get_waveform_information()[dig_port.name]["measurement_windows"]
acquisition_period = int(measurement_windows[-1][-1] - measurement_windows[0][0])
# seq.draw()

# raise SystemError
var = Variables()
pump_amplitude = Variable("pump_amplitude", np.linspace(0.9, 1.2, 31), "V")
var.add(pump_amplitude)
var.compile()
sequence_JPA = Sequence(port_list=[JPA_port])
sequence_JPA.add(Square(amplitude=pump_amplitude, duration=drive_length), JPA_port)

dig_ch.cycles(cycles)
dig_ch.delay(int(measurement_windows[0][0]//dig_ch.sampling_interval()))
dig_ch.points_per_cycle(acquisition_period//dig_ch.sampling_interval())

initialise_or_create_database_at(db_dir+f"\\{database_name}.db")
experiment_name = "pump amplitude check with coherent state"
measurement_name = f"target freq: {target_freq}, target_form:#{run_id}"
exp = load_or_create_experiment(experiment_name, sample_name)

meas = Measurement(exp, station, measurement_name)
amplitude_param = Parameter(name='amplitude', label='Pump Amplitdue', unit='V', set_cmd=None, get_cmd=None)
time_param = Parameter(name='time', label='Time', unit='ns', set_cmd=None, get_cmd=None)
waveform_param = Parameter(name='waveform', label='waveform', unit='V', set_cmd=None, get_cmd=None, vals=ComplexNumbers())
g_plus_e_I_param = Parameter(name='g_plus_e_I', label='g+e amp_dir=I', unit='V', set_cmd=None, get_cmd=None)
g_minus_e_I_param = Parameter(name='g_minus_e_I', label='g-e amp_dir=I', unit='V', set_cmd=None, get_cmd=None)
g_plus_e_Q_param = Parameter(name='g_plus_e_Q', label='g+e amp_dir=Q', unit='V', set_cmd=None, get_cmd=None)
g_minus_e_Q_param = Parameter(name='g_minus_e_Q', label='g-e amp_dir=Q', unit='V', set_cmd=None, get_cmd=None)
meas.register_parameter(time_param, paramtype="array")
meas.register_parameter(amplitude_param, paramtype="array")
meas.register_parameter(waveform_param, setpoints=[amplitude_param, time_param], paramtype="array")
meas.register_parameter(g_plus_e_I_param, setpoints=[amplitude_param, time_param], paramtype="array")
meas.register_parameter(g_minus_e_I_param, setpoints=[amplitude_param, time_param], paramtype="array")
meas.register_parameter(g_plus_e_Q_param, setpoints=[amplitude_param, time_param], paramtype="array")
meas.register_parameter(g_minus_e_Q_param, setpoints=[amplitude_param, time_param], paramtype="array")

try:
    with meas.run() as datasaver:
        datasaver.dataset.add_metadata("wiring", wiring)
        datasaver.dataset.add_metadata("setup_script", setup_script)
        datasaver.dataset.add_metadata("setup_script_JPA", setup_script_JPA)
        datasaver.dataset.add_metadata("script", script)
        for update_command in tqdm(var.update_command_list):
            sequence_JPA.update_variables(update_command)
            sequence_JPA.compile()
            i_JPA, q_JPA = iq_corrector_JPA.correct(JPA_port.waveform.conj())
            g_plus_e_I=[]
            g_plus_e_Q=[]
            g_minus_e_I=[]
            g_minus_e_Q=[]
            for _ in  tqdm(range(repetition)):
                for state in ["0+1_i", "0+1_q", "0-1_i", "0-1_q"]:
                    awg1.flush_waveform()
                    if state=="0+1_i":
                        awg1.load_waveform(waveform_AWG_coherent_state, 0, append_zeros=True)
                        awg1.load_waveform(i_JPA, 1, append_zeros=True)
                        awg1.load_waveform(q_JPA, 2, append_zeros=True)
                        awg_readout_I.queue_waveform(0, trigger="software/hvi", cycles=cycles)
                        awg_JPA_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
                        awg_JPA_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
                    if state=="0+1_q":
                        awg1.load_waveform(waveform_AWG_coherent_state, 0, append_zeros=True)
                        awg1.load_waveform(-i_JPA, 1, append_zeros=True)
                        awg1.load_waveform(-q_JPA, 2, append_zeros=True)
                        awg_readout_I.queue_waveform(0, trigger="software/hvi", cycles=cycles)
                        awg_JPA_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
                        awg_JPA_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
                    if state=="0-1_i":
                        awg1.load_waveform(-waveform_AWG_coherent_state, 0, append_zeros=True)
                        awg1.load_waveform(i_JPA, 1, append_zeros=True)
                        awg1.load_waveform(q_JPA, 2, append_zeros=True)
                        awg_readout_I.queue_waveform(0, trigger="software/hvi", cycles=cycles)
                        awg_JPA_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
                        awg_JPA_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
                    if state=="0-1_q":
                        awg1.load_waveform(-waveform_AWG_coherent_state, 0, append_zeros=True)
                        awg1.load_waveform(-i_JPA, 1, append_zeros=True)
                        awg1.load_waveform(-q_JPA, 2, append_zeros=True)
                        awg_readout_I.queue_waveform(0, trigger="software/hvi", cycles=cycles)
                        awg_JPA_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
                        awg_JPA_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
                    awg1.start_all()
                    dig_ch.start()
                    hvi_trigger.output(True)
                    data = dig_ch.read().mean(axis=0)* voltage_step
                    hvi_trigger.output(False)
                    dig_ch.stop()
                    
                    if state=="0+1_i": g_plus_e_I = np.append(g_plus_e_I, data)
                    if state=="0+1_q": g_plus_e_Q = np.append(g_plus_e_Q, data)
                    if state=="0-1_i": g_minus_e_I = np.append(g_minus_e_I, data)
                    if state=="0-1_q": g_minus_e_Q = np.append(g_minus_e_Q, data)
            g_plus_e_I = g_plus_e_I.reshape(int(repetition), int(dig_ch.points_per_cycle())).mean(axis=0)
            g_plus_e_Q = g_plus_e_Q.reshape(int(repetition), int(dig_ch.points_per_cycle())).mean(axis=0)
            g_minus_e_I = g_minus_e_I.reshape(int(repetition), int(dig_ch.points_per_cycle())).mean(axis=0)
            g_minus_e_Q = g_minus_e_Q.reshape(int(repetition), int(dig_ch.points_per_cycle())).mean(axis=0)
            waveform_I = (g_plus_e_I - g_minus_e_I)/2
            waveform_Q = (g_plus_e_Q - g_minus_e_Q)/2
            waveform = (waveform_I + waveform_Q) / 2
            datasaver.add_result(
                (amplitude_param, sequence_JPA.variable_dict['pump_amplitude'][0].value),
                (time_param, np.arange(len(waveform))*dig_ch.sampling_interval()), 
                (waveform_param,waveform),
                (g_plus_e_I_param, g_plus_e_I),(g_plus_e_Q_param, g_plus_e_Q),(g_minus_e_I_param, g_minus_e_I),(g_minus_e_Q_param, g_minus_e_Q),
            )
finally:
    off()
    print('finished')
