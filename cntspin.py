import numpy as np

class cntSpin(object):
    r"""
    Class for getting expected value of <S_vector> = (<Sx>, <Sy>, <Sz>) for
    carbon nanotube states given a basis.
    """
    # Pauli matrices
    pauliX = np.matrix([[0,1],[1,0]], dtype=np.complex128)
    pauliY = np.matrix([[0,-1j],[1j,0]], dtype=np.complex128)
    pauliZ = np.matrix([[1,0],[0,-1]], dtype=np.complex128)
    pauliI = np.eye(2, dtype=np.complex128)
    # Spin matrices in the basis defined in this class as default:
    # Kup Kdown K'up K'down
    def_Sx = np.kron(pauliI, pauliX)
    def_Sy = np.kron(pauliI, pauliY)
    def_Sz = np.kron(pauliI, pauliZ)

    def __init__(self, basis=None):
        """
        Parameters
        ==========
        basis : ndarray
            4x4 array which changes the basis of the spin matrices relative to
            the default basis Kup Kdown K'up K'down.
            The basis satisfies the equation
                basis * v_user = v_default
            where v_default is in the default basis and v_user is in the user
            basis.
        """
        if basis is None:
            basis = np.eye(4)
        self.basis = basis
        self.set_spin_matrices()

    def set_spin_matrices(self):
        """
        Set the spin matrices in the basis specified by self.basis.
        """
        basis_inv = np.linalg.inv(self.basis)
        self.Sx = basis_inv @ self.def_Sx @ self.basis
        self.Sy = basis_inv @ self.def_Sy @ self.basis
        self.Sz = basis_inv @ self.def_Sz @ self.basis

    def get_spin_vector(self, state):
        """
        Parameters
        ==========
        state : ndarray
            Column vector with four entries specifying a state in the basis
            specified by self.basis.

        Returns
        =======
        spin_exp_vector : ndarray
            Expectation value of the spin vector <S_vec> = (<Sx>, <Sy>, <Sz>).
        """
        state_H = state.conj().T
        Sx_exp = state_H @ self.Sx @ state
        Sy_exp = state_H @ self.Sy @ state
        Sz_exp = state_H @ self.Sz @ state
        spin_exp_vector = np.array([Sx_exp, Sy_exp, Sz_exp]).squeeze()
        assert not any(spin_exp_vector.imag)
        return spin_exp_vector.real
