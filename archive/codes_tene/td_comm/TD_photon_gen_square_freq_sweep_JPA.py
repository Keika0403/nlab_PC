from calendar import timegm
import time
from qcodes import initialise_or_create_database_at, load_or_create_experiment, Measurement, load_by_run_spec
from qcodes.instrument.parameter import Parameter
from qcodes.utils.validators import ComplexNumbers
from TD_setups_JPA import *
from tqdm import tqdm

database_name="photon_generation"
with open(__file__) as file:
    script = file.read()

cycles = 50000
repetition = 10
amplitude = 0.6
fogi_drive_freqs = np.linspace(5.45, 5.75, 41)
measurement_length = 3000
hvi_trigger.trigger_period(50000)
JPA_pump_amplitude = 0.8
drive_duration = 20000

seq1 = Sequence([qubit_drive_port, fogi_port, dig_port])
seq1.add(Delay(300), qubit_drive_port)
seq1.add(ResetPhase(0), qubit_drive_port)
seq1.add(ResetPhase(0), fogi_port)
# seq1.call(reset_sequence)
seq1.call(ge_half_pi_seq)
seq1.call(ef_pi_seq)
seq1.trigger([qubit_drive_port, fogi_port, dig_port])
seq1.add(Square(amplitude=amplitude, duration=drive_duration), fogi_port)
seq1.add(Acquire(measurement_length+100), dig_port)


seq2 = Sequence([qubit_drive_port, fogi_port, dig_port])
seq2.add(Delay(300), qubit_drive_port)
seq2.add(ResetPhase(0), qubit_drive_port)
seq2.add(ResetPhase(0), fogi_port)
# seq2.call(reset_sequence)
seq2.add(VirtualZ(np.pi), qubit_drive_port)
seq2.call(ge_half_pi_seq)
seq2.add(VirtualZ(np.pi), qubit_drive_port)
seq2.call(ef_pi_seq)
seq2.trigger([qubit_drive_port, fogi_port, dig_port])
seq2.add(Square(amplitude=amplitude, duration=drive_duration), fogi_port)
seq2.add(Acquire(measurement_length+100), dig_port)

# plt.plot(i_qubit1)
# plt.plot(i_fogi)
# plt.plot(i_JPA, lw=1)
# plt.show()
# seq1.draw()
# seq2.draw()
# raise SystemError


initialise_or_create_database_at(db_dir+f"\\{database_name}.db")
experiment_name = "photon gen square freq sweep JPA"
measurement_name = f"amplitude {amplitude}, drive:{fogi_drive_freqs[0]}~{fogi_drive_freqs[-1]}"
exp = load_or_create_experiment(experiment_name, sample_name)

meas = Measurement(exp, station, measurement_name)
fogi_freq_param = Parameter(name='fogi_freq', label='fogi frequency', unit='GHz', set_cmd=None, get_cmd=None)
time_param = Parameter(name='time', label='Time', unit='ns', set_cmd=None, get_cmd=None)
waveform_param = Parameter(name='waveform', label='waveform', unit='V', set_cmd=None, get_cmd=None, vals=ComplexNumbers())
g_plus_e_I_param = Parameter(name='g_plus_e_I', label='g+e amp_dir=I', unit='V', set_cmd=None, get_cmd=None)
g_minus_e_I_param = Parameter(name='g_minus_e_I', label='g-e amp_dir=I', unit='V', set_cmd=None, get_cmd=None)
g_plus_e_Q_param = Parameter(name='g_plus_e_Q', label='g+e amp_dir=Q', unit='V', set_cmd=None, get_cmd=None)
g_minus_e_Q_param = Parameter(name='g_minus_e_Q', label='g-e amp_dir=Q', unit='V', set_cmd=None, get_cmd=None)
meas.register_parameter(time_param, paramtype="array")
meas.register_parameter(fogi_freq_param, paramtype="array")
meas.register_parameter(waveform_param, setpoints=[fogi_freq_param, time_param,], paramtype="array")
meas.register_parameter(g_plus_e_I_param, setpoints=[fogi_freq_param, time_param,], paramtype="array")
meas.register_parameter(g_minus_e_I_param, setpoints=[fogi_freq_param, time_param,], paramtype="array")
meas.register_parameter(g_plus_e_Q_param, setpoints=[fogi_freq_param, time_param,], paramtype="array")
meas.register_parameter(g_minus_e_Q_param, setpoints=[fogi_freq_param, time_param,], paramtype="array")

# lo1.output(True)
# lo2.output(True)
# lo3.output(True)
try:
    with meas.run() as datasaver:
        datasaver.dataset.add_metadata("wiring", wiring)
        datasaver.dataset.add_metadata("setup_script", setup_script)
        datasaver.dataset.add_metadata("setup_script_JPA", setup_script_JPA)
        datasaver.dataset.add_metadata("parameter_script", parameter_script)
        datasaver.dataset.add_metadata("script", script)
        for fogi_drive_freq in tqdm(fogi_drive_freqs):
            fogi_port.if_freq = fogi_drive_freq - fogi_lo_freq
            seq1.compile()
            i_qubit1, q_qubit1 = iq_corrector_q.correct(qubit_drive_port.waveform.conj()) # 0+1
            i_fogi, q_fogi = iq_corrector_fogi.correct(fogi_port.waveform.conj())
            measurement_windows_1 = seq1.get_waveform_information()[dig_port.name]["measurement_windows"]
            seq2.compile()
            i_qubit2, q_qubit2 = iq_corrector_q.correct(qubit_drive_port.waveform.conj()) # 0-1
            measurement_windows_2 = seq2.get_waveform_information()[dig_port.name]["measurement_windows"]
            acquisition_period = int(measurement_windows_2[-1][-1] - measurement_windows_2[0][0])

            sequence_JPA = Sequence(port_list=[JPA_port])
            sequence_JPA.add(Square(amplitude=JPA_pump_amplitude, duration=len(i_fogi)), JPA_port)
            sequence_JPA.compile()
            i_JPA, q_JPA = iq_corrector_JPA.correct(JPA_port.waveform.conj())

            dig_ch.cycles(cycles)
            dig_ch.delay(int(measurement_windows_2[0][0]//dig_ch.sampling_interval()))
            dig_ch.points_per_cycle(acquisition_period//dig_ch.sampling_interval())

            g_plus_e_I=[]
            g_plus_e_Q=[]
            g_minus_e_I=[]
            g_minus_e_Q=[]

            for _ in  tqdm(range(repetition)):
                for state in ["0+1_i", "0+1_q", "0-1_i", "0-1_q"]:
                    awg1.flush_waveform()
                    awg2.flush_waveform()
                    awg_Qdrive_I.dc_offset(iq_corrector_q.i_offset)
                    awg_Qdrive_Q.dc_offset(iq_corrector_q.q_offset)
                    awg_fogi_I.dc_offset(iq_corrector_fogi.i_offset)
                    awg_fogi_Q.dc_offset(iq_corrector_fogi.q_offset)
                    awg_JPA_I.dc_offset(iq_corrector_JPA.i_offset)
                    awg_JPA_Q.dc_offset(iq_corrector_JPA.q_offset)
                    if state=="0+1_i":
                        awg1.load_waveform(i_JPA, 1, append_zeros=True)
                        awg1.load_waveform(q_JPA, 2, append_zeros=True)
                        awg2.load_waveform(i_qubit1, 1, append_zeros=True)
                        awg2.load_waveform(q_qubit1, 2, append_zeros=True)
                        awg2.load_waveform(i_fogi, 3, append_zeros=True)
                        awg2.load_waveform(q_fogi, 4, append_zeros=True)
                        awg_Qdrive_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
                        awg_Qdrive_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
                        awg_fogi_I.queue_waveform(3, trigger="software/hvi", cycles=cycles)
                        awg_fogi_Q.queue_waveform(4, trigger="software/hvi", cycles=cycles)
                        awg_JPA_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
                        awg_JPA_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
                    if state=="0+1_q":
                        awg1.load_waveform(-i_JPA, 1, append_zeros=True)
                        awg1.load_waveform(-q_JPA, 2, append_zeros=True)
                        awg2.load_waveform(i_qubit1, 1, append_zeros=True)
                        awg2.load_waveform(q_qubit1, 2, append_zeros=True)
                        awg2.load_waveform(i_fogi, 3, append_zeros=True)
                        awg2.load_waveform(q_fogi, 4, append_zeros=True)
                        awg_Qdrive_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
                        awg_Qdrive_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
                        awg_fogi_I.queue_waveform(3, trigger="software/hvi", cycles=cycles)
                        awg_fogi_Q.queue_waveform(4, trigger="software/hvi", cycles=cycles)
                        awg_JPA_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
                        awg_JPA_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
                    if state=="0-1_i":
                        awg1.load_waveform(i_JPA, 1, append_zeros=True)
                        awg1.load_waveform(q_JPA, 2, append_zeros=True)
                        awg2.load_waveform(i_qubit2, 1, append_zeros=True)
                        awg2.load_waveform(q_qubit2, 2, append_zeros=True)
                        awg2.load_waveform(i_fogi, 3, append_zeros=True)
                        awg2.load_waveform(q_fogi, 4, append_zeros=True)
                        awg_Qdrive_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
                        awg_Qdrive_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
                        awg_fogi_I.queue_waveform(3, trigger="software/hvi", cycles=cycles)
                        awg_fogi_Q.queue_waveform(4, trigger="software/hvi", cycles=cycles)
                        awg_JPA_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
                        awg_JPA_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
                    if state=="0-1_q":
                        awg1.load_waveform(-i_JPA, 1, append_zeros=True)
                        awg1.load_waveform(-q_JPA, 2, append_zeros=True)
                        awg2.load_waveform(i_qubit2, 1, append_zeros=True)
                        awg2.load_waveform(q_qubit2, 2, append_zeros=True)
                        awg2.load_waveform(i_fogi, 3, append_zeros=True)
                        awg2.load_waveform(q_fogi, 4, append_zeros=True)
                        awg_Qdrive_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
                        awg_Qdrive_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
                        awg_fogi_I.queue_waveform(3, trigger="software/hvi", cycles=cycles)
                        awg_fogi_Q.queue_waveform(4, trigger="software/hvi", cycles=cycles)
                        awg_JPA_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
                        awg_JPA_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
                    awg1.start_all()
                    awg2.start_all()
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
                (fogi_freq_param, fogi_drive_freq),
                (time_param, np.arange(len(waveform))*dig_ch.sampling_interval()), 
                (waveform_param,waveform),
                (g_plus_e_I_param, g_plus_e_I),(g_plus_e_Q_param, g_plus_e_Q),(g_minus_e_I_param, g_minus_e_I),(g_minus_e_Q_param, g_minus_e_Q),
            )
finally:
    off()
    print('finished')
