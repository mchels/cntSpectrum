import sys
sys.path.append('..')
from cntspectrum import cntSpectrum
from cntspin import cntSpin
import numpy as np
import matplotlib.pyplot as plt
model_kw = {
    'deltaSO': 0.15,
    'deltaKK': 0.07,
    'mu_orb': 0.15,
    'J': 0.12,
}
model = cntSpectrum(**model_kw)
spin = cntSpin()
B_fields = 1
B_angles = np.linspace(0, np.pi, 45)
spectrum, eigenvectors = model.get_spectrums(B_fields, B_angles,
                                             two_electron=False,
                                             get_eigenvectors=True)
# Plot spectrum.squeeze() instead of spectrum to remove single-dimensional
# entries that arise because our B_fields is a number and not a list.
fig, ax = plt.subplots()
ax.plot(B_angles, spectrum.squeeze())
n_states = 4
spin_dimensions = 3
shape = (len(np.ravel(B_fields)), len(B_angles), n_states, spin_dimensions)
spin_components_array = np.zeros(shape=shape, dtype=float)
for i, _ in enumerate(np.ravel(B_fields)):
    for j, _ in enumerate(B_angles):
        eigenmatrix = eigenvectors[i,j]
        for k, eigenvector in enumerate(eigenmatrix.T):
            spin_vector = spin.get_spin_vector(eigenvector)
            spin_components_array[i,j,k] = spin_vector
colors = ('b', 'g', 'r', 'c')
for k in range(n_states):
    ax.quiver(
        B_angles, spectrum[:,:,k],
        spin_components_array[:,:,k,0], spin_components_array[:,:,k,2],
        pivot='middle',
        color=colors[k],
        width=0.002,
    )
plt.show()
