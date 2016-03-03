import sys
sys.path.append('..')
from cntspectrum import cntSpectrum
from cntspin import cntSpin
import numpy as np
import matplotlib.pyplot as plt

"""
For deltaSO==0 The expectation value for the spin operator must always be
parallel or anti-parallel with the effective magnetic field B_ff:
      B_ff_x = B_xt_x           = B*sin(B_angle)
      B_ff_z = B_xt_z + tau*BSO = B*cos(B_angle) + tau*BSO
where B_ff_i (B_xt_i) is the i'th component of the effective (external)
magnetic field, B_angle is the angle between the external magnetic field and
the nanotube axis, tau is the valley number (+1 or -1) and BSO is the
spin-orbit magnetic field in the nanotube (=0 for deltaSO=0). The z-direction
is parallel to the nanotube axis.

This script tests that the calculated expectation value for the spin operatorfor a deltaSO==0 and deltaKK' not zero and a range of magnetic fields is parallel or anti-parallel with the effective magnetic field for every state.
"""

model_kw = {
    'deltaSO': 0.0,
    'deltaKK': 0.3,
    'mu_orb': -0.15,
    'J': 0.12,
}
model = cntSpectrum(**model_kw)
spin = cntSpin()
filling = 1
B_fields = np.linspace(0, 5, 101)
# The upper limit is np.pi*1.01 to avoid having an angle of exactly pi/2. At
# this angle the states are degenerate in pairs at any magnitude of the
# B_field. Since any combination of the two spin states within a degenerate
# pair is an energy eigenstate the spin expectation value is zero (I think).
B_angles = np.linspace(0, np.pi*1.01, 91)
spectrum, eigenvectors = model.get_spectrum(B_fields, B_angles, filling,
                                            get_eigenvectors=True)
spin_vectors = spin.get_spin_vectors_from_eigenstates(eigenvectors)
for i, B_field in enumerate(B_fields):
    for j, B_angle in enumerate(B_angles):
        eigenmatrix = eigenvectors[i,j]
        B_ff_x = B_field * np.sin(B_angle)
        B_ff_z = B_field * np.cos(B_angle)
        for k, eigenvector in enumerate(eigenmatrix.T):
            spin_vector = spin_vectors[i,j,k]
            # If the magnetic field magnitude is zero we get a finite value
            # for the spin expectation value and a comparison between the two
            # will result in False. Physically, the spin expectation value
            # should be zero because of degeneracy as noted above.
            if np.isclose(B_ff_x, 0) and np.isclose(B_ff_z, 0):
                continue
            # If the x component of either the effective magnetic field or the
            # spin vector is zero we can't use them in the denominator below.
            # In this case ensuring that BOTH x-components are zero and BOTH
            # z-components are non-zero is sufficient to say that the two
            # vectors are (anti-)parallel.
            if np.isclose(B_ff_x, 0) or np.isclose(spin_vector[0], 0):
                assert np.isclose(B_ff_x, 0)
                assert not np.isclose(B_ff_z, 0)
                assert np.isclose(spin_vector[0], 0)
                assert not np.isclose(spin_vector[2], 0)
                continue
            assert np.isclose(B_ff_z/B_ff_x, spin_vector[2]/spin_vector[0])
