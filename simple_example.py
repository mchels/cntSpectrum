# cntspectrum.py must be located in the same folder as this file.
from cntspectrum import cntSpectrum
import numpy as np
import matplotlib.pyplot as plt
model_kw = {
    'deltaSO': 0.15,
    'deltaKK': 0.07,
    'mu_orb': 0.15,
    'J': 0.12,
}
model = cntSpectrum(**model_kw)
B_fields = 2
B_angles = np.linspace(0, np.pi, 46)
spectrums = model.get_spectrums(B_fields, B_angles, two_electron=False)
# Plot spectrums.squeeze() instead of spectrums to remove single-dimensional
# entries that arise because our B_fields is a number and not a list.
fig, ax = plt.subplots()
ax.plot(B_angles, spectrums.squeeze())
ax.set_xlabel('Magnetic field angle (radians)')
ax.set_ylabel('Energy (meV)')
plt.savefig('example.png')
plt.show()
