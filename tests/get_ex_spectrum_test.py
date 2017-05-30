import sys
sys.path.append('..')
from cntspectrum import cntSpectrum
import os
import numpy as np
from numpy import pi

"""
This script tests the current get_ex_spectrums function against experimental
data for a shell in the conduction band. The data is published in
"Noncollinear Spin-Orbit Magnetic Fields in a Carbon Nanotube Double
Quantum Dot", M. C. Hels et al. Phys. Rev. Lett. 117, 276802 (2016).
"""

model_kw = {
    'deltaSO': 0.15,
    'deltaKK': 0.07,
    'mu_orb': -0.15,
    'J': 0.12,
}
model = cntSpectrum(**model_kw)
bias_offset = -0.05
B_fields_dict = {
    'B_field': np.linspace(-3, 3, 50),
    'B_field_perp': np.linspace(-2, 2, 50),
    'B_angle': 2,
}
tube_Bx_angle = 83.0*pi/180
B_angles_dict = {
    'B_field': 72*pi/180 - tube_Bx_angle,
    'B_field_perp': 162*pi/180 - tube_Bx_angle,
    'B_angle': np.linspace(62*pi/180, 262*pi/180, 50) - tube_Bx_angle,
}
for B_type in ('B_field', 'B_field_perp', 'B_angle'):
    B_fields = B_fields_dict[B_type]
    B_angles = B_angles_dict[B_type]
    for filling in (1, 2, 3):
        ex_spectrums = model.get_ex_spectrums(
            B_fields,
            B_angles,
            filling,
            bias_offset=bias_offset,
            deltaSC=0.02,
            BC=0.8,
        )
        fname = '{}{}.dat'.format(filling, B_type)
        fpath = os.path.relpath('reference_data_get_ex_spectrum/' + fname)
        reference_data = np.loadtxt(fpath)
        np.testing.assert_allclose(ex_spectrums.squeeze(), reference_data)
