from qcodes_drivers.E4407B import E4407B

spectrum_analyzer = E4407B("spectrum_analyzer", "GPIB0::18::INSTR")
print(spectrum_analyzer )