from calendar import timegm
import time
from qcodes import initialise_or_create_database_at, load_or_create_experiment, Measurement, load_by_run_spec
from qcodes.instrument.parameter import Parameter
from qcodes.utils.validators import ComplexNumbers
from TD_setups import *
from tqdm import tqdm

database_name="photon_generation"
starttime=time.time()
with open(__file__) as file:
    script = file.read()

repetition = int(60000)
acquisition_period = 600
amplitude = 0.1
fogi_drive_freq = 5.65
drive_duration = 10000

seq1 = Sequence([qubit_drive_port, fogi_port])
seq1.call(ge_half_pi_seq)
seq1.call(ef_pi_seq)
seq1.trigger([qubit_drive_port, fogi_port])
seq1.add(Square(amplitude=amplitude, duration=drive_duration), fogi_port)
fogi_port.if_freq = fogi_drive_freq - fogi_lo_freq
seq1.compile()
i1, q1 = iq_corrector_q.correct(qubit_drive_port.waveform.conj())
fogi_i1, fogi_q1 = iq_corrector_fogi.correct(fogi_port.waveform.conj())

seq2 = Sequence([qubit_drive_port, fogi_port])
seq2.add(VirtualZ(np.pi), qubit_drive_port)
seq2.call(ge_half_pi_seq)
seq2.add(VirtualZ(np.pi), qubit_drive_port)
seq2.call(ef_pi_seq)
seq2.trigger([qubit_drive_port, fogi_port])
seq2.add(Square(amplitude=amplitude, duration=drive_duration), fogi_port)
fogi_port.if_freq = fogi_drive_freq - fogi_lo_freq
seq2.compile()
i2, q2 = iq_corrector_q.correct(qubit_drive_port.waveform.conj())
fogi_i2, fogi_q2 = iq_corrector_fogi.correct(fogi_port.waveform.conj())

seq_JPA = Sequence(port_list=[JPA_port])
seq_JPA.add(Square(amplitude=0.65, duration=len(fogi_i2)), JPA_port)
seq_JPA.compile()
i_JPA, q_JPA = iq_corrector_JPA.correct(JPA_port.waveform.conj())

dig_ch.cycles(4)
dig_ch.delay(len(qubit_drive_port.waveform)//dig_ch.sampling_interval())
dig_ch.points_per_cycle(acquisition_period//dig_ch.sampling_interval())

initialise_or_create_database_at(db_dir+f"\\{database_name}.db")
experiment_name = "photon gen square JPA"
measurement_name = f"amplitude {amplitude}, drive:{fogi_drive_freq}"
exp = load_or_create_experiment(experiment_name, sample_name)

meas = Measurement(exp, station, measurement_name)
time_param = Parameter(name='time', label='Time', unit='ns', set_cmd=None, get_cmd=None)
waveform_param = Parameter(name='waveform', label='waveform', unit='V', set_cmd=None, get_cmd=None, vals=ComplexNumbers())
g_plus_e_I_param = Parameter(name='g_plus_e_I', label='g+e amp_dir=I', unit='V', set_cmd=None, get_cmd=None)
g_minus_e_I_param = Parameter(name='g_minus_e_I', label='g-e amp_dir=I', unit='V', set_cmd=None, get_cmd=None)
g_plus_e_Q_param = Parameter(name='g_plus_e_Q', label='g+e amp_dir=Q', unit='V', set_cmd=None, get_cmd=None)
g_minus_e_Q_param = Parameter(name='g_minus_e_Q', label='g-e amp_dir=Q', unit='V', set_cmd=None, get_cmd=None)
meas.register_parameter(time_param, paramtype="array")
meas.register_parameter(waveform_param, setpoints=[time_param], paramtype="array")
meas.register_parameter(g_plus_e_I_param, setpoints=[time_param], paramtype="array")
meas.register_parameter(g_minus_e_I_param, setpoints=[time_param], paramtype="array")
meas.register_parameter(g_plus_e_Q_param, setpoints=[time_param], paramtype="array")
meas.register_parameter(g_minus_e_Q_param, setpoints=[time_param], paramtype="array")

lo1.output(True)
lo2.output(True)
lo3.output(True)
try:
    with meas.run() as datasaver:
        datasaver.dataset.add_metadata("wiring", wiring)
        datasaver.dataset.add_metadata("setup_script", setup_script)
        datasaver.dataset.add_metadata("script", script)
        res=[]
        for _ in  tqdm(range(repetition)):
            awg1.flush_waveform()
            awg2.flush_waveform()
            awg2.load_waveform(i1, 1, append_zeros=True)
            awg2.load_waveform(q1, 2, append_zeros=True)
            awg2.load_waveform(i2, 3, append_zeros=True)
            awg2.load_waveform(q2, 4, append_zeros=True)
            awg2.load_waveform(fogi_i1, 5, append_zeros=True)
            awg2.load_waveform(fogi_q1, 6, append_zeros=True)
            awg1.load_waveform(i_JPA, 1, append_zeros=True)
            awg1.load_waveform(q_JPA, 2, append_zeros=True)
            awg1.load_waveform(-i_JPA, 3, append_zeros=True)
            awg1.load_waveform(-q_JPA, 4, append_zeros=True)
            awg_Qdrive_I.queue_waveform(1, trigger="software/hvi", cycles=2)
            awg_Qdrive_I.queue_waveform(2, trigger="software/hvi", cycles=2)
            awg_Qdrive_Q.queue_waveform(3, trigger="software/hvi", cycles=2)
            awg_Qdrive_Q.queue_waveform(4, trigger="software/hvi", cycles=2)
            awg_fogi_I.queue_waveform(5, trigger="software/hvi", cycles=4)
            awg_fogi_Q.queue_waveform(6, trigger="software/hvi", cycles=4)
            awg_JPA_I.queue_waveform(1, trigger="software/hvi", cycles=1)
            awg_JPA_I.queue_waveform(3, trigger="software/hvi", cycles=1)
            awg_JPA_I.queue_waveform(1, trigger="software/hvi", cycles=1)
            awg_JPA_I.queue_waveform(3, trigger="software/hvi", cycles=1)
            awg_JPA_Q.queue_waveform(2, trigger="software/hvi", cycles=1)
            awg_JPA_Q.queue_waveform(4, trigger="software/hvi", cycles=1)
            awg_JPA_Q.queue_waveform(2, trigger="software/hvi", cycles=1)
            awg_JPA_Q.queue_waveform(4, trigger="software/hvi", cycles=1)
            awg_Qdrive_I.dc_offset(iq_corrector_q.i_offset)
            awg_Qdrive_Q.dc_offset(iq_corrector_q.q_offset)
            awg_fogi_I.dc_offset(iq_corrector_fogi.i_offset)
            awg_fogi_Q.dc_offset(iq_corrector_fogi.q_offset)
            awg_JPA_I.dc_offset(iq_corrector_JPA.i_offset)
            awg_JPA_Q.dc_offset(iq_corrector_JPA.q_offset)
            awg2.start_all()
            dig_ch.start()
            hvi_trigger.output(True)
            res.append(dig_ch.read()* voltage_step)
            hvi_trigger.output(False)
            dig_ch.stop()
        res = np.array(res).reshape(4*repetition, 1, acquisition_period//2)
        g_plus_e_I = res[::4].mean(axis=0).ravel()
        g_plus_e_Q = res[1::4].mean(axis=0).ravel()
        g_minus_e_I = res[2::4].mean(axis=0).ravel()
        g_minus_e_Q = res[3::4].mean(axis=0).ravel()
        waveform_I = (g_plus_e_I - g_minus_e_I)/2
        waveform_Q = (g_plus_e_Q - g_minus_e_Q)/2
        waveform = (waveform_I + waveform_Q) / 2
        datasaver.add_result(
            (time_param, np.arange(len(waveform))*dig_ch.sampling_interval()), 
            (waveform_param,waveform),
            (g_plus_e_I_param, g_plus_e_I),(g_plus_e_Q_param, g_plus_e_Q),(g_minus_e_I_param, g_minus_e_I),(g_minus_e_Q_param, g_minus_e_Q),
        )
finally:
    off()
    print('finished')
