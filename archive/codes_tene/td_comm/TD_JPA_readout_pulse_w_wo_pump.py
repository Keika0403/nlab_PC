from calendar import timegm
from qcodes import initialise_or_create_database_at, load_or_create_experiment, Measurement
from qcodes.instrument.parameter import Parameter
from qcodes.utils.validators import ComplexNumbers
from tqdm import tqdm
from TD_setups_JPA import *

database_name="tomography"
initialise_or_create_database_at(db_dir+f"\\{database_name}.db")
with open(__file__) as file:
    script = file.read()

cycles = 10000
hvi_trigger.digitizer_delay(400)

var = Variables()
amplitude = Variable("amplitude", np.append([0], np.linspace(0.8, 1.0, 41)), "V")
var.add(amplitude)
var.compile()

seq_JPA = Sequence(port_list=[JPA_port])
seq_JPA.add(Square(amplitude=amplitude, duration=100), JPA_port, copy=False)

seq = Sequence(port_list=[dig_port, JPA_port, readout_port])
seq.call(seq_JPA)
seq.call(readout_seq)
seq.add(Acquire(100), dig_port)

measurement_name = f"trigger delay {hvi_trigger.digitizer_delay()}"
experiment_name = "JPA readout pulse check"
exp = load_or_create_experiment(experiment_name, sample_name)
meas = Measurement(exp, station, measurement_name)
time_param = Parameter(name='time', label='Time', unit='ns', set_cmd=None, get_cmd=None)
waveform_param = Parameter(name='waveform', label='amplitude', unit='V', set_cmd=None, get_cmd=None, vals=ComplexNumbers())
amplitude_param = Parameter(name='amplitude', label='amplitude', unit='V', set_cmd=None, get_cmd=None, vals=ComplexNumbers())
meas.register_parameter(time_param, paramtype="array")
meas.register_parameter(amplitude_param, paramtype="array")
meas.register_parameter(waveform_param, setpoints=[amplitude_param, time_param], paramtype="array")

dig_ch.cycles(cycles)
# lo1.output(True)
try:
    with meas.run() as datasaver:
        datasaver.dataset.add_metadata("wiring", wiring)
        datasaver.dataset.add_metadata("setup_script", setup_script)
        datasaver.dataset.add_metadata("setup_script_JPA", setup_script_JPA)
        datasaver.dataset.add_metadata("script", script)
        for update_command in tqdm(var.update_command_list):
            seq.update_variables(update_command)
            load_sequence(seq, cycles=cycles)
            if seq.variable_dict["amplitude"][0].value != 0 and seq.variable_dict["amplitude"][0].value < 0.6:
                continue
            waveform = run(seq).mean(axis=0)*voltage_step
            datasaver.add_result(
                (amplitude_param, seq.variable_dict["amplitude"][0].value),
                (time_param, np.arange(len(waveform))*dig_ch.sampling_interval()), 
                (waveform_param, waveform),
            )
            awg1.flush_waveform()
            dig_ch.stop()
except:print('An error occurred')
finally:
    off()
    print('finished')