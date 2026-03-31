from calendar import timegm
import time
from qcodes import initialise_or_create_database_at, load_or_create_experiment, Measurement
from qcodes.instrument.parameter import Parameter
from qcodes.utils.validators import ComplexNumbers
from tqdm import tqdm
from TD_setups_JPA import *
strt=time.time()

database_name="tomography"
initialise_or_create_database_at(db_dir+f"\\{database_name}.db")
starttime=time.time()
with open(__file__) as file:
    script = file.read()

shot_count = 50000
acquisition_period = 600
run_id=534  # 

seq = Sequence([qubit_drive_port, fogi_port])
seq.call(reset_sequence)
seq.call(ge_half_pi_seq)
seq.call(ef_pi_seq)

measurement_name = f"waveform id {run_id}"
experiment_name = "JPA photon single shot"
exp = load_or_create_experiment(experiment_name, sample_name)
meas = Measurement(exp, station, measurement_name)
shot_num_param = Parameter(name='shot_number', label='shot number', unit=' ', set_cmd=None, get_cmd=None)
s11_param = Parameter(name='s11', label='s11', unit=' ', set_cmd=None, get_cmd=None, vals=ComplexNumbers())
meas.register_parameter(shot_num_param, paramtype="array")
meas.register_parameter(s11_param, setpoints=[shot_num_param], paramtype="array")

dig_ch.cycles(shot_count)
try:
    with meas.run() as datasaver:
        datasaver.dataset.add_metadata("wiring", wiring)
        datasaver.dataset.add_metadata("setup_script", setup_script)
        datasaver.dataset.add_metadata("setup_script_JPA", setup_script_JPA)
        datasaver.dataset.add_metadata("script", script)
        load_fogi_wavefrom(seq, run_id, shot_count, JPA_phase=0)
        data = run_photon_gen(mode="each")
        s11 = demodulate(data)
        datasaver.add_result(
            (shot_num_param, np.arange(shot_count)), 
            (s11_param, s11),
        )
        awg1.flush_waveform()
        dig_ch.stop()
except:print('An error occurred')
finally:
    off()
    print('finished')

try:
    run_id = datasaver.dataset.run_id
except:run_id=None
Notify=SlackNotify()
Notify.notify(experiment_name, measurement_name, strt=strt, run_id=run_id)