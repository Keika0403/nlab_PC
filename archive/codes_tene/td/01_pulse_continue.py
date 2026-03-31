from setup_td import *


# seq = Sequence(port_list = [qubit_drive_port]) 
# seq.add(Square(amplitude=1, duration=2000), qubit_drive_port)
# seq.compile()
# waveform_i , waveform_q = iq_corrector_q.correct(qubit_drive_port.waveform)
# awg2.load_waveform(waveform_i, 0, append_zeros=True)
# awg2.load_waveform(waveform_q, 1, append_zeros=True)
# awg_Qdrive_I.queue_waveform(0, trigger="software/hvi", cycles=0, per_cycle=False)
# awg_Qdrive_Q.queue_waveform(1, trigger="software/hvi", cycles=0, per_cycle=False)
# awg2.start_all()


seq1 = Sequence(port_list = [readout_port]) 
seq1.add(Square(amplitude=1, duration=10000), readout_port)
seq1.compile()
awg1.load_waveform(readout_port.waveform.real, 0, append_zeros=True)
awg_readout_I.queue_waveform(0, trigger="software/hvi", cycles=0, per_cycle=False)
awg1.start_all()

# seq2 = Sequence(port_list = [fogi_port]) 
# seq2.add(Square(amplitude=1, duration=2000), fogi_port)
# seq2.compile()
# i, q = iq_corrector_fogi.correct(fogi_port.waveform)
# awg2.load_waveform(i, 0, append_zeros=True)
# awg2.load_waveform(q, 1, append_zeros=True)
# awg_fogi_I.queue_waveform(0, trigger="software/hvi", cycles=0, per_cycle=False)
# awg_fogi_Q.queue_waveform(1, trigger="software/hvi", cycles=0, per_cycle=False)
# awg2.start_all()

#####
# awg_i = awg1.ch3
# awg_q = awg1.ch4
# target_freq = 10.3
# lo_freq = 10.38
# lo1.frequency(lo_freq*1e9)
# JPA_if_freq = target_freq*2 - lo_freq*2

# seq = Sequence(port_list=[JPA_port])
# seq.add(Square(amplitude=0.5, duration=100000), JPA_port, copy=False)
# seq.compile()
# i, q = iq_corrector_JPA.correct(JPA_port.waveform, cyclic=True)
# awg1.load_waveform(i, 1, append_zeros=True)
# awg1.load_waveform(q, 2, append_zeros=True)
# awg_i.queue_waveform(1, trigger="software/hvi", cycles=0, per_cycle=False)
# awg_q.queue_waveform(2, trigger="software/hvi", cycles=0, per_cycle=False)
# awg_i.dc_offset(iq_corrector_JPA.i_offset)
# awg_q.dc_offset(iq_corrector_JPA.q_offset)
# awg1.start_all()
####


lo1.output(True)
# lo2.output(True)
# lo3.output(True)
hvi_trigger.output(True)
hvi_trigger.output(False)
try:
    while True:
        'continue to send pulses'
except KeyboardInterrupt:
    awg1.stop_all()
    lo1.output(False)
    # awg2.stop_all()
    # lo2.output(False)
    # awg2.stop_all()
    # lo3.output(False)
    print('AWG and LO are turned off')
