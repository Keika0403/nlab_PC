from TD_setups_for_tomography import *
from qcodes import load_by_run_spec, initialise_or_create_database_at

with open(__file__) as file:
    setup_script_JPA = file.read()

# readout threshold
qc.initialise_or_create_database_at(db_dir+f"\\tomography.db")
dataset = load_by_run_spec(captured_run_id=557)
mean_g=dataset.get_parameter_data()['pulse_g']['pulse_g'].ravel()
mean_e=dataset.get_parameter_data()['pulse_e']['pulse_e'].ravel()
ein_vec = (mean_e - mean_g)/np.sum((mean_e - mean_g)**2)

# photon to be tomographed
run_id = 640 # ctrl #
target_freq = 10.51 # check yokogawa current
center = 496 # related to hvitrigger digitizer delay
const = 2e-3 * 2 * pi
form="sech"
JPA_port.if_freq = readout_freq*2 - readout_lo_freq*2
JPA_if_freq_for_photon = target_freq*2 - readout_lo_freq*2
initialise_or_create_database_at("D:\\miyamura\\waveform_data\\waveform_at_AWG.db")
dataset = load_by_run_spec(captured_run_id=run_id)
waveform_AWG=dataset.get_parameter_data()['waveform']['waveform'].ravel()
drive_length = len(waveform_AWG) + 100
print(drive_length)
time_AWG=np.unique(dataset.get_parameter_data()['waveform']['time'])
measurement_name=str(run_id) +'_' + dataset.name

# target waveform for coherent state
time = np.arange(1000)
center_of_AWG_pulse = 500
target = (1/np.cosh(const * (time - center_of_AWG_pulse))) * np.cos(2 * np.pi * (target_freq-readout_lo_freq) * (time - center_of_AWG_pulse))

# angle correction for <X>, <Y> 
theta = -0.4839923678066558 * np.pi

# JPA pump
JPA_DC = 90.e-6 # A
width_JPA_pump = 100 # ns
seq_JPA = Sequence(port_list=[JPA_port])
JPA_phase = ResetPhase(np.pi*1.25)
JPA_pump=Square(amplitude=0.94, duration=width_JPA_pump)
seq_JPA.add(JPA_phase, JPA_port, copy=False)
seq_JPA.add(JPA_pump, JPA_port, copy=False)

# readout pulse
width_readout_pulse = 40
readout_pulse = Square(amplitude=0.85, duration=width_readout_pulse)
readout_seq = Sequence(port_list=[readout_port, dig_port])
dig_acquire = Acquire(width_readout_pulse)
readout_seq.trigger([readout_port, dig_port])
readout_seq.add(ResetPhase(0), readout_port, copy=False)
readout_seq.add(readout_pulse, readout_port, copy=False)
readout_seq.add(dig_acquire, dig_port, copy=False)
readout_seq.trigger([readout_port, dig_port])
readout_seq.add(Delay(10), readout_port, copy=False)

# readout pulse & JPA pump
readout_JPA_seq = Sequence(port_list=[readout_port, JPA_port, dig_port])
readout_JPA_seq.trigger([readout_port, JPA_port, dig_port])
readout_JPA_seq.call(readout_seq)
readout_JPA_seq.call(seq_JPA)
readout_JPA_seq.trigger([readout_port, JPA_port, dig_port])
readout_JPA_seq.add(Delay(20), readout_port)
# readout_JPA_seq.draw()

# probe sequence
acquire_length = 300
probe_seq = Sequence(port_list=[readout_port, dig_port])
probe_pulse = Square(amplitude=0.3, duration=1000)
probe_seq.trigger([readout_port, dig_port])
probe_seq.add(ResetPhase(0), readout_port)
probe_seq.add(probe_pulse, readout_port)
# probe_seq.add(Delay((probe_pulse.params["duration"]-acquire_length)//2), dig_port)
# probe_seq.add(Acquire(acquire_length), dig_port)
# probe_seq.add(Delay((probe_pulse.params["duration"]-acquire_length)//2), dig_port)
probe_seq.add(Acquire(probe_pulse.params["duration"]), dig_port)


def fourier_tr(x,y, sort=False):
    off_ini = np.mean(y)
    y_mod = y - off_ini
    N = len(y)
    x_fft = np.fft.fftfreq(N,d=x[1]-x[0])
    y_fft = np.fft.fft(y_mod)
    if sort:
        sorted_idx = np.argsort(x_fft)
        x_fft = x_fft[sorted_idx]
        y_fft = y_fft[sorted_idx]
    return x_fft,y_fft

def s11_g_(f, f_r=10.5105408 , k_ex=0.03962612, k_in=7.2950e-04, kai=-0.00613717, demo_LO=readout_lo_freq):
    return 1 - k_ex/((k_ex+k_in)/2 + 1j*(f - np.sign(f)*(f_r-demo_LO)))

def load_fogi_wavefrom(seq:Sequence, id:int, cycles:int, acquisition_period=1000, withJPA=1, JPA_phase=0):
    qc.initialise_or_create_database_at("D:\\miyamura\\waveform_data\\waveform_at_AWG.db")
    dataset = qc.load_by_run_spec(captured_run_id=id)
    waveform_AWG=dataset.get_parameter_data()['waveform']['waveform']

    seq.compile()
    i, q = iq_corrector_q.correct(qubit_drive_port.waveform.conj())
    plt.plot(i)
    awg2.load_waveform(i, 1, append_zeros=True)
    awg2.load_waveform(q, 2, append_zeros=True)
    fogi_waveform = np.append(fogi_port.waveform, waveform_AWG)
    i,q = iq_corrector_fogi.correct(fogi_waveform.conj())
    awg2.load_waveform(i, 3, append_zeros=True)
    awg2.load_waveform(q, 4, append_zeros=True)
    plt.plot(i)

    awg_Qdrive_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
    awg_Qdrive_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
    awg_fogi_I.queue_waveform(3, trigger="software/hvi", cycles=cycles)
    awg_fogi_Q.queue_waveform(4, trigger="software/hvi", cycles=cycles)
    dig_ch.cycles(cycles)
    dig_ch.delay(len(qubit_drive_port.waveform)//dig_ch.sampling_interval())
    dig_ch.points_per_cycle(acquisition_period//dig_ch.sampling_interval())

    if withJPA:
        seq_JPA = Sequence(port_list=[JPA_port])
        seq_JPA.add(Square(amplitude=JPA_pump.params["amplitude"], duration=2000), JPA_port, copy=False)
        
        seq_for_JPA = Sequence([qubit_drive_port, fogi_port, JPA_port, dig_port])
        seq_for_JPA.call(seq)
        seq_for_JPA.trigger([qubit_drive_port, fogi_port, JPA_port])
        seq_for_JPA.add(VirtualZ(JPA_phase), JPA_port)
        seq_for_JPA.call(seq_JPA)
        seq_for_JPA.compile()
        i, q = iq_corrector_JPA.correct(JPA_port.waveform.conj())
        awg1.load_waveform(i, 1, append_zeros=True)
        awg1.load_waveform(q, 2, append_zeros=True)
        awg_JPA_I.queue_waveform(1, trigger="software/hvi", cycles=cycles)
        awg_JPA_Q.queue_waveform(2, trigger="software/hvi", cycles=cycles)
        plt.plot(i)

def mode_function(center=center, const=const, form=form):
    center = center / dig_ch.sampling_interval()
    time = np.arange((drive_length) // dig_ch.sampling_interval())
    if form=="sech":
        return np.sqrt(const/2)/np.cosh(const*(time - center))
    elif form=="gaussian":return np.sqrt(np.exp(-(time-center)**2/const**2)/const/np.sqrt(np.pi))
mode = mode_function()