import numpy as np
import sympy as sy
from sympy.physics.matrices import msigma
from sympy.physics.quantum import TensorProduct
from basis import simple_to_def, def_to_simple, sy_mat_to_np


class cntSpectrum(object):
    r"""
    Get spectrum, excitation spectrum and Hamiltonians for a carbon nanotube
    model.

    Parameters
    ----------
    deltaSO, deltaKK, J : int or float
        Value of the parameter in milli-electronvolts.
    g_orb : int or float, optional
        Orbital g-factor (unitless).
        In the CONDUCTION band this is a NEGATIVE number
        In the VALENCE band this is a POSITIVE number
        Either g_orb exclusive or mu_orb must be set.
    mu_orb : int or float, optional
        Orbital magnetic moment in units of milli-electronvolts per tesla.
        In the CONDUCTION band this is a NEGATIVE number
        In the VALENCE band this is a POSITIVE number
        Either g_orb exclusive or mu_orb must be set.
    bias_offset : int or float, optional
        Specify the bias offset to take into account when calculating
        the excitation spectrum.

    Attributes
    ----------
    Attributes include the parameters above and the following:
    BSO : float
        Magnitude of the spin-orbit magnetic field calculated as
        BSO = self.deltaSO / (self.g_s*self.mu_B)

    Notes
    -----
    The spectrum is from
        E. Laird et al., Reviews of Modern Physics, 87, 703 (2015)
    page 722 equation (25). This spectrum is only valid in the limit where the
    band gap is much larger than the spin-orbit energy and the parallel
    magnetic field splitting.

    The Hamiltonian is written in the basis Kup K'down Kdown K'up which is
    called the 'default' basis in this module.

    We use the same convention for g_orb as in the paper above:
        g_orb = mu_orb / mu_B.

    deltaSO, deltaKK, g_orb and J are set at class instantiation and should
    not be changed afterwards.
    """

    # mu_B in units of milli electronvolts per Tesla
    mu_B = 0.0578

    # Spin electron g-factor
    g_s = 2.0

    # 1 and 3-electron Hamiltonian matrices.
    # The first (second) Pauli matrices below works in valley (spin) space.
    h_pauli = {
        'SO': (msigma(3), msigma(3)),
        'KK': (msigma(1), sy.eye(2)),
        'orb': (msigma(3), sy.eye(2)),
        'par': (sy.eye(2), msigma(3)),
        'perp': (sy.eye(2), msigma(1)),
    }
    sub_Hs = {k: simple_to_def(TensorProduct(*v)) for k, v in h_pauli.items()}
    sub_Hs_np = {k: sy_mat_to_np(v) for k, v in sub_Hs.items()}

    # 2-electron Hamiltonian matrices.
    sub_Hs_N2 = {
        'SO': np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]),
        'KK': np.array([
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]),
        'orb': np.array([
            [1, 0,  0, 0, 0, 0],
            [0, 0,  0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0],
            [0, 0,  0, 0, 0, 0],
            [0, 0,  0, 0, 0, 0],
            [0, 0,  0, 0, 0, 0],
        ]),
        'par': np.array([
            [0, 0, 0, 0, 0,  0],
            [0, 0, 0, 0, 0,  0],
            [0, 0, 0, 0, 0,  0],
            [0, 0, 0, 1, 0,  0],
            [0, 0, 0, 0, 0,  0],
            [0, 0, 0, 0, 0, -1],
        ]),
        'perp': np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ]),
        'ex': np.array([
            [0, 0, 0,  0,  0,  0],
            [0, 1, 0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0],
            [0, 0, 0, -1,  0,  0],
            [0, 0, 0,  0, -1,  0],
            [0, 0, 0,  0,  0, -1],
        ]),
    }

    def __init__(self, deltaKK, J, deltaSO=None, BSO=None, g_orb=None,
                 mu_orb=None, bias_offset=0):
        self.deltaKK = deltaKK
        self.J = J
        self.bias_offset = bias_offset
        assert (deltaSO is None) ^ (BSO is None)
        self._deltaSO = deltaSO
        self._BSO = BSO
        assert (g_orb is None) ^ (mu_orb is None)
        self._g_orb = g_orb
        self._mu_orb = mu_orb

    def get_spectrums(self, B_fields, B_angles, filling,
                      get_eigenvectors=False):
        r"""
        Get spectrums and eigenvectors for the given Hamiltonians.
        The basis is Kup K'down Kdown K'up.

        Parameters
        ----------
        B_fields, B_angles : 1D arrays or int or float.
            B_fields is a list of magnetic field strengths in Tesla.
            B_angles is a list of the magnetic field angles in radians. The
            nanotube is assumed to be oriented along 0 radians.
        filling : int, must be 1, 2 or 3
            filling=1 and filling=3 yield identical outputs.
        get_eigenvectors : Boolean
            Specify whether to return eigenvectors (states) along with
            spectrum.

        Returns
        -------
        spectrums : ndarray
            If B_fields and B_angles are arrays:
            The spectrums array has shape
            len(B_fields) x len(B_angles) x n_states
            where filling=(1,3): n_states=4; filling=2: n_states=6
        states : ndarray
            Eigenvectors for the system which are returned if get_eigenvectors
            is True. If B_fields and B_angles are arrays states has shape
            len(B_fields) x len(B_angles) x n_states x n_states.

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
        spectrums = model.get_spectrums(B_fields, B_angles, filling)
        plt.plot(B_angles, spectrums.squeeze())
        """
        hamils = self.get_hamils(B_fields, B_angles, filling)
        if get_eigenvectors:
            spectrums, eigvecs = np.linalg.eigh(hamils)
            return (spectrums, eigvecs)
        else:
            spectrums = np.linalg.eigvalsh(hamils)
            return spectrums

    def get_ex_spectrums(self, B_fields, B_angles, filling, bias_offset=0,
                         deltaSC=None, BC=None):
        r"""
        Parameters
        ----------
        B_fields, B_angles : 1D arrays or int or float.
            B_fields is a list of magnetic field strengths in Tesla.
            B_angles is a list of the magnetic field angles in radians. The
            nanotube is assumed to be oriented along 0 radians.
        filling: : int, must be 1, 2 or 3
            filling=1 has different behavior from filling=3. In get_spectrum
            they have the same behavior.
        bias_offset : int or float
            Bias offset in meV.
        deltaSC : int or float
            Superconducting gap in meV.
        BC : int or float
            The critical magnetic field for the superconductor in Tesla.
            Both deltaSC and BC must be provided for the superconducting gap to
            be added to the spectrum.

        Returns
        -------
        ex_spectrums : ndarray
            Excitation spectrums.

        Notes
        -----
        - If
            1) the nanotube forms a quantum dot,
            2) spectrum of the nanotube is being probed by inelastic cotunneling
               excitation spectroscopy, and
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
        spectrums = self.get_spectrums(B_fields, B_angles, filling)
        assert isinstance(filling, int)
        # filling==2 has its own Hamiltonian where each state holds two
        # electrons. Thus, for filling==2 ONE state is occupied.
        if filling == 2:
            n_occ_states = 1
        else:
            n_occ_states = filling
        assert n_occ_states > 0
        occ_Es = spectrums[...,:n_occ_states]
        non_occ_Es = spectrums[...,n_occ_states:]
        # Insert np.newaxis to calculate all combinations of occupied and
        # unoccupied energies.
        ex_spectrums = non_occ_Es[...,:,np.newaxis] - occ_Es[...,np.newaxis,:]
        # Flatten the inner-most two dimensions.
        newshape = spectrums.shape[:-1] + (-1,)
        ex_spectrums = np.reshape(ex_spectrums, newshape=newshape)
        if deltaSC is not None and BC is not None:
            SC_gap = self._SC_gap(deltaSC, BC, B_fields, B_angles)
            ex_spectrums += SC_gap[...,np.newaxis]
        ex_spectrums = np.sort(ex_spectrums, axis=-1)
        # Stack negative excitation energies with the positive ones.
        ex_spectrums = np.concatenate([ex_spectrums, -ex_spectrums], axis=-1)
        ex_spectrums += bias_offset
        return ex_spectrums

    def get_hamils(self, B_fields, B_angles, filling):
        """
        Get Hamiltonians for the given parameters.
        The basis is Kup K'down Kdown K'up.

        Parameters
        ----------
        B_fields, B_angles : 1D arrays or int or float.
            B_fields is a list of magnetic field strengths in Tesla.
            B_angles is a list of the magnetic field angles in radians. The
            nanotube is assumed to be oriented along 0 radians.
        filling : int, must be 1, 2 or 3
            filling=1 and filling=3 yield identical outputs.

        Returns
        -------
        hamils: ndarray
            Hamiltonians for the given parameters.
        """
        # ravel ensures that B_fields and B_angles are iterable if they are not
        # already.
        B_fields = np.ravel(B_fields)
        B_angles = np.ravel(B_angles)
        B_fields_4D = B_fields[:,np.newaxis,np.newaxis,np.newaxis]
        B_angles_4D = B_angles[np.newaxis,:,np.newaxis,np.newaxis]
        g_orb = self.g_orb
        deltaSO = self.deltaSO
        deltaKK = self.deltaKK
        if filling in (1, 3):
            hamils = self.H_total(B_fields_4D, B_angles_4D, deltaSO, deltaKK,
                                  g_orb)
        elif filling == 2:
            J = self.J
            hamils = self.H_total_N2(B_fields_4D, B_angles_4D, deltaSO, deltaKK,
                                     g_orb, J)
        else:
            raise ValueError('self.filling is not 1, 2 or 3. Aborting.')
        return hamils

    @staticmethod
    def _SC_gap(deltaSC, BC, B_fields, B_angles):
        _, B_field_mesh = np.meshgrid(B_angles, B_fields)
        temp = 1 - (B_field_mesh/BC)**2
        temp = temp.clip(min=0)
        SC_gap = deltaSC * np.sqrt(temp)
        return SC_gap

    @classmethod
    def H_SO(cls, deltaSO):
        mat = cls.sub_Hs_np['SO']
        return 1 / 2 * deltaSO * mat

    @classmethod
    def H_KK(cls, deltaKK):
        mat = cls.sub_Hs_np['KK']
        return 1 / 2 * deltaKK * mat

    @classmethod
    def H_orb(cls, B_fields, B_angles, g_orb):
        mat = cls.sub_Hs_np['orb']
        mu_B = cls.mu_B
        return B_fields * np.cos(B_angles) * g_orb * mu_B * mat

    @classmethod
    def H_par(cls, B_fields, B_angles):
        mat = cls.sub_Hs_np['par']
        g_s = cls.g_s
        mu_B = cls.mu_B
        return 1 / 2 * B_fields * np.cos(B_angles) * g_s * mu_B * mat

    @classmethod
    def H_perp(cls, B_fields, B_angles):
        mat = cls.sub_Hs_np['perp']
        g_s = cls.g_s
        mu_B = cls.mu_B
        return 1 / 2 * B_fields * np.sin(B_angles) * g_s * mu_B * mat

    @classmethod
    def H_total(cls, B_fields, B_angles, deltaSO, deltaKK, g_orb):
        Bf = B_fields
        Ba = B_angles
        H_SO = cls.H_SO(deltaSO)
        H_KK = cls.H_KK(deltaKK)
        H_orb = cls.H_orb(Bf, Ba, g_orb)
        H_par = cls.H_par(Bf, Ba)
        H_perp = cls.H_perp(Bf, Ba)
        # Insert np.newaxis in H_SO and H_KK that do not depend on magnetic
        # field and thus do not have magnetic field dimensions.
        H_total = H_SO[np.newaxis,np.newaxis,:,:] + \
                  H_KK[np.newaxis,np.newaxis,:,:] + H_orb + H_par + H_perp
        return H_total

    @classmethod
    def H_SO_N2(cls, deltaSO):
        mat = cls.sub_Hs_N2['SO']
        return deltaSO * mat

    @classmethod
    def H_KK_N2(cls, deltaKK):
        mat = cls.sub_Hs_N2['KK']
        return 1 / np.sqrt(2.0) * deltaKK * mat

    @classmethod
    def H_orb_N2(cls, B_fields, B_angles, g_orb):
        mat = cls.sub_Hs_N2['orb']
        mu_B = cls.mu_B
        return 2 * B_fields * np.cos(B_angles) * g_orb * mu_B * mat

    @classmethod
    def H_par_N2(cls, B_fields, B_angles):
        mat = cls.sub_Hs_N2['par']
        g_s = cls.g_s
        mu_B = cls.mu_B
        return B_fields * np.cos(B_angles) * g_s * mu_B * mat

    @classmethod
    def H_perp_N2(cls, B_fields, B_angles):
        mat = cls.sub_Hs_N2['perp']
        g_s = cls.g_s
        mu_B = cls.mu_B
        return 1 / np.sqrt(2.0) * B_fields * np.sin(B_angles) * g_s * mu_B * mat

    @classmethod
    def H_ex_N2(cls, J):
        mat = cls.sub_Hs_N2['ex']
        return 1 / 2 * J * mat

    @classmethod
    def H_total_N2(cls, B_fields, B_angles, deltaSO, deltaKK, g_orb, J):
        Bf = B_fields
        Ba = B_angles
        H_SO = cls.H_SO_N2(deltaSO)
        H_KK = cls.H_KK_N2(deltaKK)
        H_orb = cls.H_orb_N2(Bf, Ba, g_orb)
        H_par = cls.H_par_N2(Bf, Ba)
        H_perp = cls.H_perp_N2(Bf, Ba)
        H_ex = cls.H_ex_N2(J)
        # Insert np.newaxis in H_SO, H_KK and H_ex that do not depend on
        # magnetic field and thus do not have magnetic field dimensions.
        H_total = H_SO[np.newaxis,np.newaxis,:,:] + \
                  H_KK[np.newaxis,np.newaxis,:,:] + \
                  H_ex[np.newaxis,np.newaxis,:,:] + H_orb + H_par + H_perp
        return H_total

    @property
    def deltaSO(self):
        if self._deltaSO is None:
            return self._BSO * self.g_s * self.mu_B
        else:
            return self._deltaSO

    @property
    def BSO(self):
        if self._BSO is None:
            return self._deltaSO / (self.g_s*self.mu_B)
        else:
            return self._BSO

    @property
    def g_orb(self):
        if self._g_orb is None:
            return self._mu_orb / self.mu_B
        else:
            return self._g_orb

    @property
    def mu_orb(self):
        if self._mu_orb is None:
            return self._g_orb * self.mu_B
        else:
            return self._mu_orb
