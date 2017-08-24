import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')
from cntspectrum import cntSpectrum

# Make a 3x3 plot of the nanotube spectrum or excitation spectrum with rows
# representing parallel, perpendicular and angle sweeps and columns
# representing filling.

# Instantiate model.
model_kw = {
    'deltaSO': 0.15,
    'deltaKK': 0.07,
    'g_orb': 2.6,
    'J': 0.12,
}
model = cntSpectrum(**model_kw)

# When doing magnetic field sweeps of a nanotube there are three relevant
# angles:
# B_tube: Angle between magnetic field and nanotube.
# B_Bx: Angle between magnetic field and the magnetic field x-axis.
# tube_Bx: Angle between nanotube axis and magnetic field x-axis.
# The angles are related like so:
#       B_tube = B_Bx - tube_Bx
# Typically, in experimental data the magnetic field angle is measured from Bx,
# i.e., the experimental data is B_Bx above. The cntSpectrum class takes B_tube
# as input so we must offset the experimentally used B_Bx values accordingly.
# The numbers below represent an actual experiment in which the angle of the
# nanotube axis was initially unknown. This is the cause of the misalignment of
# 72-83 = -11 degrees.
# If you just want to see what the spectrum looks like for perfect alignment
# set tube_Bx_angle=0, 'par': 0, 'perp': np.pi/2 and 'angle' to whatever range
# you want to plot.
tube_Bx_angle = 83 * np.pi / 180
B_angles_dict = {
    'par': 72*np.pi/180 - tube_Bx_angle,
    'perp': 162*np.pi/180 - tube_Bx_angle,
    'angle': np.linspace(0, np.pi, 46) - tube_Bx_angle,
}

# Set magnetic field magnitude.
B_fields_dict = {
    'par': np.linspace(-3, 3, 41),
    'perp': np.linspace(-2, 2, 21),
    'angle': 2,
}

# Choose whether to plot spectrum or ex_spectrum.
# plot_func = model.get_spectrum
plot_func = model.get_ex_spectrums

fillings = (1, 2, 3)
types = ('par', 'perp', 'angle')
fig, axes = plt.subplots(3, 3, sharey='row')
for i, type in enumerate(types):
    for j, filling in enumerate(fillings):
        ax = axes[i,j]
        B_fields = B_fields_dict[type]
        B_angles = B_angles_dict[type]
        if type in ('par', 'perp'):
            B_list_for_plotting = B_fields
        elif type == 'angle':
            B_list_for_plotting = B_angles
        data_for_plotting = plot_func(B_fields, B_angles, filling)
        # Plot data_for_plotting.squeeze() instead of data_for_plotting to
        # remove single-dimensional entries that arise because our B_fields is
        # a number and not a list.
        ax.plot(B_list_for_plotting, data_for_plotting.squeeze())

fig.tight_layout()
plt.show()
