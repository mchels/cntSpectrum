import numpy as np
from numpy import cos, sin, sqrt

# mu_B in units of milli electronvolts per Tesla
MU_B = 0.0578

class cntSpectrum:
    r"""
    Get spectrum and excitation spectrum for a carbon nanotube.

    Parameters
    ----------
    deltaSO, deltaKK, J : int or float
        Value of the parameter in milli-electronvolts.
    g_orb : int or float, optional
        Orbital g-factor (unitless).
        Either g_orb exclusive or mu_orb must be set.
    mu_orb : int or float, optional
        Orbital magnetic moment in units of milli-electronvolts per tesla.
        Either g_orb exclusive or mu_orb must be set.
    bias_offset : int or float, optional
        Specify the bias offset to take into account when calculating
        the excitation spectrum.

    Notes
    -----
    The spectrum is from
        E. Laird et al., Reviews of Modern Physics, 87, 703 (2015)
    page 722 equation (25). This spectrum is only valid in the limit where the
    band gap is much larger than the spin-orbit energy and the parallel
    magnetic field splitting.

    We use the same convention for g_orb as in the paper above:
        g_orb = mu_orb / mu_B.

    deltaSO, deltaKK, g_orb and J are set at class instantiation and should
    not be changed afterwards.
    """
    def __init__(self, deltaSO, deltaKK, J, g_orb=None, mu_orb=None,
                 bias_offset=0):
        self.deltaSO = deltaSO
        self.deltaKK = deltaKK
        self.J = J
        self.bias_offset = bias_offset
        if type(g_orb) in (int, float) and mu_orb is None:
            self.g_orb = g_orb
            self.mu_orb = g_orb * MU_B
        elif type(mu_orb) in (int, float) and g_orb is None:
            self.g_orb = mu_orb / MU_B
            self.mu_orb = mu_orb
        else:
            err_str = 'Either g_orb exclusive or mu_orb must be a number.'
            raise TypeError(err_str)

    def get_spectrum(self, B_fields, B_angles, filling):
        r"""
        Parameters
        ----------
        B_fields, B_angles : array-like or int or float.
            B_fields is a list of magnetic field strengths.
            B_angles is a list of the magnetic field angles which are
            measured from the axis of the nanotube.
        filling : int, must be 1, 2 or 3

        Returns
        -------
        spectrum : ndarray
            If B_fields and B_angles are arrays:
            The spectrum array has dimensions of
            B_fields x B_angles x n_states
            where filling=(1,3): n_states=4. filling=2: n_states=6
            If either or both B_fields and B_angles are int or float the
            corresponding 0-d dimension is squeezed out of the output
            array.

        Notes
        -----
        If an angle offset is desired (e.g., because the experimental
        magnetic field x-axis does not coincide with the tube axis)
        this must be arranged by the user's own code.

        Examples
        --------
        # Import cntSpectrum however you want
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
        """
        assert filling in (1,2,3)
        # ravel ensures that B_fields and B_angles are iterable if they are int
        # or float.
        B_fields = np.ravel(B_fields)
        B_angles = np.ravel(B_angles)
        n_states = 4
        if filling == 2:
            n_states = 6
        shape = (len(B_fields), len(B_angles), n_states)
        spectrum = np.zeros(shape=shape, dtype=float)
        for i, B_field in enumerate(B_fields):
            for j, B_angle in enumerate(B_angles):
                hamil = self._get_hamil(B_field, B_angle, filling)
                eigvalues = np.linalg.eigvalsh(hamil)
                spectrum[i,j] = np.sort(eigvalues)
        # squeeze removes 0-d arrays that arise if either or both B_fields and
        # B_angles are integers or floats.
        spectrum = spectrum.squeeze()
        return spectrum

    def get_ex_spectrum(self, B_fields, B_angles, filling, bias_offset=0,
                        deltaSC=None, BC=None):
        r"""
        Parameters
        ----------
        B_fields, B_angles, filling : Same as for get_spectrum.
        bias_offset : int or float
            Bias offset in meV.
        deltaSC : int or float
            Superconducting gap in meV.
        BC : int or float
            The critical magnetic field for the superconductor.
            Both deltaSC and BC must be provided for the superconducting gap to
            be added to the spectrum.

        Returns
        -------
        ex_spectrum : ndarray
            Excitation spectrum.

        Notes
        -----
        - If
            1) the nanotube forms a quantum dot
            2) its spectrum is being probed by inelastic cotunneling excitation spectroscopy
            3) one of the leads is a superconductor
        a correction must be added to the excitation spectrum to account for
        the suppression of cotunneling inside the superconducting gap.
        - If both leads are identical superconductors, use 2*deltaSC for
        deltaSC.
        - This function does not support two superconducting leads with
        dissimilar deltaSC or BC, although it can probably be hacked by taking
        averages of the two deltaSC respectively BC.
        - The superconducting gap as a function of magnetic field is calculated
        as
            deltaSC(B_field) = deltaSC(B_field=0) * sqrt(1-(B_field/BC)^2)
        This equation is from D. H. Douglass, Phys Rev Lett, 6, 7 (1961).
        """
        spectrum = self.get_spectrum(B_fields, B_angles, filling)
        lowest_energies = spectrum[...,0][...,np.newaxis]
        non_lowest_energies = spectrum[...,1:]
        ex_spectrum = non_lowest_energies - lowest_energies
        if deltaSC is not None and BC is not None:
            SC_gap = self._SC_gap(deltaSC, BC, B_fields, B_angles)
            ex_spectrum += SC_gap[...,np.newaxis]
        # Stack negative excitation energies with the positive ones.
        axis = len(ex_spectrum.shape) - 1
        ex_spectrum = np.concatenate([ex_spectrum, -ex_spectrum], axis=axis)
        ex_spectrum += bias_offset
        return ex_spectrum

    def _get_hamil(self, B_field, B_tube_angle, filling):
        if filling == 1:
            hamil = h_0(self.deltaSO, self.deltaKK) + \
                    B_field * h_B(B_tube_angle, self.g_orb)
        elif filling == 2:
            hamil = self.deltaSO * h_SO + \
                    self.deltaKK * h_KK + \
                    self.g_orb * B_field * MU_B * cos(B_tube_angle) * h_borb + \
                    self.J * h_ex + \
                    MU_B * B_field * cos(B_tube_angle) * h_bs + \
                    MU_B * B_field * sin(B_tube_angle) * h_perp
        elif filling == 3:
            hamil = h_0(-self.deltaSO, self.deltaKK) + \
                    B_field * h_B(B_tube_angle, self.g_orb)
        else:
            raise ValueError('self.filling is not 1, 2 or 3. Aborting.')
        return hamil

    @staticmethod
    def _SC_gap(deltaSC, BC, B_fields, B_angles):
        _, B_field_mesh = np.meshgrid(B_angles, B_fields)
        temp = 1 - (B_field_mesh/BC)**2
        temp = temp.clip(min=0)
        SC_gap = deltaSC * np.sqrt(temp)
        return SC_gap.squeeze()


# 1-electron matrices
def h_0(deltaSO, deltaKK):
    matrix = 0.5*np.matrix([
        [-deltaSO, 0       , 0      , deltaKK],
        [0       , -deltaSO, deltaKK, 0      ],
        [0       , deltaKK , deltaSO, 0      ],
        [deltaKK , 0       , 0      , deltaSO]
    ])
    return matrix

def h_B(angle, g_orb):
    gs = 2
    def diag(spin, tau):
        return cos(angle) * (0.5*spin*gs+tau*g_orb)
    matrix = MU_B * np.matrix([
        [diag(1,1) , 0          , sin(angle), 0         ],
        [0         , diag(-1,-1), 0         , sin(angle)],
        [sin(angle), 0          , diag(-1,1), 0         ],
        [0         , sin(angle) , 0         , diag(1,-1)]
    ])
    return matrix

# 2-electron Hamiltonians.
h_SO = np.matrix([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
])

h_KK = 1/sqrt(2.0)*np.matrix([
        [0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
])

h_ex = 0.5*np.matrix([
        [0, 0, 0,  0,  0,  0],
        [0, 1, 0,  0,  0,  0],
        [0, 0, 0,  0,  0,  0],
        [0, 0, 0, -1,  0,  0],
        [0, 0, 0,  0, -1,  0],
        [0, 0, 0,  0,  0, -1],
])

h_borb = 2*np.matrix([
        [1, 0,  0, 0, 0, 0],
        [0, 0,  0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0],
        [0, 0,  0, 0, 0, 0],
        [0, 0,  0, 0, 0, 0],
        [0, 0,  0, 0, 0, 0],
])

h_bs = 2*np.matrix([
        [0, 0, 0, 0, 0,  0],
        [0, 0, 0, 0, 0,  0],
        [0, 0, 0, 0, 0,  0],
        [0, 0, 0, 1, 0,  0],
        [0, 0, 0, 0, 0,  0],
        [0, 0, 0, 0, 0, -1],
])

h_perp = sqrt(2.0)*np.matrix([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0],
])
