import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')
from cntspectrum import cntSpectrum

# Plot the energy of an excitation in the nanotube spectrum as a function of
# magnetic field strength and angle as a polar plot.
model_kw = {
   'deltaSO': 0.15,
   'deltaKK': 0.07,
   'g_orb': 2.6,
   'J': 0.12,
}
model = cntSpectrum(**model_kw)
B_fields = np.linspace(0, 3, 41)
B_angles = np.linspace(0, 2*np.pi, 181)
filling = 3
ex_spectrum = model.get_ex_spectrums(B_fields, B_angles, filling)
excitation_n = 0
excitation_energy = ex_spectrum[...,excitation_n]
ax = plt.subplot(111, projection='polar')
B_angles_mesh, B_fields_mesh = np.meshgrid(B_angles, B_fields)
ax.pcolormesh(
    B_angles_mesh, B_fields_mesh, excitation_energy,
    cmap=plt.get_cmap('Reds')
)
ax.grid(True)
plt.show()
