from qcodes.instrument_drivers.yokogawa.GS200 import GS200
current_source = GS200("current_source", "TCPIP0::192.168.100.98::inst0::INSTR")
current_source.ramp_current(108e-6, step=1e-8, delay=0)
print("Completed")