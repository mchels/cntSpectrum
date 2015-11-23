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
B_angles = np.linspace(0, np.pi, 20)
filling = 1
spectrum = model.get_spectrum(B_fields, B_angles, filling)
# Plot spectrum.squeeze() instead of spectrum to remove single-dimensional
# entries that arise because our B_fields is a number and not a list.
plt.plot(B_angles, spectrum.squeeze())
plt.show()
