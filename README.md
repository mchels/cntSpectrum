Carbon nanotube spectrum
========================

Calculates carbon nanotube spectrum and excitation spectrum using the model
in equation (25) in

E. Laird et al., Reviews of Modern Physics, 87, 703 (2015)

All energy values are in milli-electronvolts.

## Example
````python
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
plt.plot(B_angles, spectrum)
````

## Citation
1. E. Laird et al., Reviews of Modern Physics, 87, 703 (2015)
2. D. H. Douglass, Phys Rev Lett, 6, 7 (1961)

## Help
Type `help(cntSpectrum)`


## Credit and Copyright
Morten Canth Hels
