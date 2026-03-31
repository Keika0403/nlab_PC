from setup_files.setup_td import *
import sys
sys.path.append('D:\\miyamura\\analysis\\programs')
from Analysis import (
    fourier_tr_padding, start_stop, lowpass,
)
from scipy import interpolate

measurement_name = os.path.basename(__file__)[:-3]
tags.append("photon")

hvi_trigger.digitizer_delay(410)
hvi_trigger.trigger_period(10000)
lo1.frequency(9.47e9)
JPA_amp = 0.78
JPA_port_tx.if_freq = (9.34 - 9.47) * 2
# JPA_amp = 1.1
# JPA_port_tx.if_freq = (9.41 - 9.47) * 2
passing_band = 0.01
phase_coeff = 0.5
const = 3e-3*2*np.pi


def IQ_sym_(wave,time,):
    norm=np.sqrt(np.sum(np.abs(wave)**2) * (time[1]-time[0]))
    m_t = wave / norm
    m_t_star_rev = np.flip(m_t).conj() 
    m=0
    for i in range(len(wave)):
        m_t_star_rev_i = np.roll(m_t_star_rev, i)
        mult = m_t * m_t_star_rev_i
        mult_sum=np.sum(mult) * (time[1] - time[0])
        cand = np.abs(mult_sum)**2
        if m<cand:
            m=cand
    return m

def target_overlap(wave, time, if_fixed=False):
    norm = np.sqrt(np.sum(np.abs(wave)**2) * (time[1] - time[0]))
    m_t = wave / norm
    target = 1 / np.cosh(const * (time-time[-1]/2), dtype=complex)
    norm_target = np.sqrt(np.sum(np.abs(target)**2) * (time[1] - time[0]))
    m_t_target = target / norm_target
    if if_fixed: 
        target = 1j / np.cosh(const * (time-250));norm_target = np.sqrt(np.sum(np.abs(target)**2) * (time[1] - time[0]))
        return np.abs(np.sum(m_t * target/norm_target) * (time[1] - time[0]))**2
    m=0 ;id=None
    for i in range(len(wave)):
        m_t_i = np.roll(m_t, i)
        mult = m_t_target*m_t_i
        mult_sum=np.sum(mult) * (time[1] - time[0])
        cand = np.abs(mult_sum)**2
        if m<cand:
            m=cand
    return m

def reverese_pulse(control_pulse):
    phase = np.unwrap(np.angle(control_pulse))
    env = np.abs(control_pulse)
    env_rvs = np.flip(env)
    phase_rvs = np.flip(phase)
    inst_omega_minus = -np.diff(phase_rvs)
    phase_receive = np.zeros(phase.shape)
    for i in range(1, len(phase_receive)):
        phase_receive[i] = phase_receive[i-1] + inst_omega_minus[i-1]
    receive_pulse = env_rvs * np.exp(1j * phase_receive)
    return receive_pulse

def calibrate_shape(
        control_pulse,
        measure_which,
        target_frequency,
        reverse = False,
        cycles = 50000,
        repetition = 2,
        iteration = 15,
        JPA_amp = JPA_amp,
    ):
    hvi_trigger.digitizer_delay(410)
    hvi_trigger.trigger_period(10000)
    def fogi_sequence(half_pi:bool, JPA_direction=1) -> Sequence:
        assert JPA_direction == -1 or 1
        if measure_which == "tx":
            seq = Sequence(port_list=ports_tx)
            seq.call(reset_sequence_tx)
            seq.trigger(ports_tx)
            seq.add(Square(amplitude=JPA_direction*JPA_amp, duration=len(control_pulse)+100), JPA_port_tx)
            if half_pi:
                seq.call(ge_half_pi_seq_tx)
            else:
                seq.add(VirtualZ(np.pi), qubit_drive_port_tx)
                seq.call(ge_half_pi_seq_tx)
                seq.add(VirtualZ(np.pi), qubit_drive_port_tx)
            seq.call(ef_pi_seq_tx)
            seq.trigger(ports_tx_wo_JPA)
            seq.add(Acquire(len(control_pulse)), fogi_port_tx)
            seq.add(Acquire(len(control_pulse) + 100), dig_port_tx)
            return seq
        elif measure_which == "txQrx":
            seq = Sequence(port_list=ports_txQrx)
            seq.call(reset_sequence_rx)
            seq.trigger(ports_txQrx)
            seq.add(Square(amplitude=JPA_direction*JPA_amp, duration=len(control_pulse)+100), JPA_port_tx)
            if half_pi:
                seq.call(ge_half_pi_seq_rx)
            else:
                seq.add(VirtualZ(np.pi), qubit_drive_port_rx)
                seq.call(ge_half_pi_seq_rx)
                seq.add(VirtualZ(np.pi), qubit_drive_port_rx)
            seq.call(ef_pi_seq_rx)
            seq.trigger(ports_txQrx_wo_JPA)
            seq.add(Acquire(len(control_pulse)), fogi_port_rx)
            seq.add(Acquire(len(control_pulse) + 100), dig_port_tx)
            return seq
    max_sym = 0
    for i in tqdm(range(iteration)):
        g_plus_e_I=[]
        g_plus_e_Q=[]
        g_minus_e_I=[]
        g_minus_e_Q=[]
        for _ in  (range(repetition)):
            for state in ["0+1_i", "0+1_q", "0-1_i", "0-1_q"]:
                awg1.flush_waveform()
                awg2.flush_waveform()
                if state=="0+1_i":
                    seq = fogi_sequence(half_pi=True, JPA_direction=1)
                if state=="0+1_q":
                    seq = fogi_sequence(half_pi=True, JPA_direction=-1)
                if state=="0-1_i":
                    seq = fogi_sequence(half_pi=False, JPA_direction=1)
                if state=="0-1_q":
                    seq = fogi_sequence(half_pi=False, JPA_direction=-1)
                if measure_which == "tx":
                    load_sequence_w_append(seq, append_ports=[fogi_port_tx], waveforms_added=[control_pulse], cycles=cycles)
                elif measure_which == "txQrx":
                    load_sequence_w_append(seq, append_ports=[fogi_port_rx], waveforms_added=[control_pulse], cycles=cycles)
                data = run(seq, which="tx").mean(axis=0) * voltage_step_tx
                if state=="0+1_i": g_plus_e_I = np.append(g_plus_e_I, data)
                if state=="0+1_q": g_plus_e_Q = np.append(g_plus_e_Q, data)
                if state=="0-1_i": g_minus_e_I = np.append(g_minus_e_I, data)
                if state=="0-1_q": g_minus_e_Q = np.append(g_minus_e_Q, data)
        g_plus_e_I = g_plus_e_I.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
        g_plus_e_Q = g_plus_e_Q.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
        g_minus_e_I = g_minus_e_I.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
        g_minus_e_Q = g_minus_e_Q.reshape(int(repetition), int(dig_ch_tx.points_per_cycle())).mean(axis=0)
        waveform_I = (g_plus_e_I - g_minus_e_I)/2
        waveform_Q = (g_plus_e_Q - g_minus_e_Q)/2
        waveform = (waveform_I + waveform_Q) / 2
        time = np.arange(len(waveform)) * dig_ch_tx.sampling_interval()

        phase = np.angle(demodulate(waveform, lo1.frequency()*1e-9-target_frequency))
        xfft, fourier = fourier_tr_padding(time, waveform)
        xfft, fourier = start_stop(xfft, fourier, xfft[0], 0)
        photon_freq = lo1.frequency()*1e-9 + xfft

        signal = 2 * lowpass(time, 
            waveform * np.exp(2j*np.pi*(lo1.frequency()*1e-9-target_frequency)*time),
            passing_band, 0.045, 0.1, 90) * np.exp(-1j*phase)
        
        phase = np.angle(signal)
        phase_of_time = interpolate.interp1d(time, np.unwrap(phase))
        phase_for_correction = phase_of_time(np.arange(len(control_pulse)))
        control_pulse = control_pulse*np.exp(1j*phase_for_correction*phase_coeff)
        if IQ_sym_(signal, time) > max_sym:
            max_sym = IQ_sym_(signal, time)
            max_waveform = waveform
            max_control_pulse = control_pulse
    print(f"max symmetry : {max_sym:.4f}")
    if reverse:
        return reverese_pulse(max_control_pulse), max_waveform
    return max_control_pulse, max_waveform
