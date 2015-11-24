import numpy as np

class cntSpin(object):
    r"""
    Class for getting expected value of <S_vector> = (<Sx>, <Sy>, <Sz>) for
    carbon nanotube states given a basis.

    Notes
    -----
    Notation:
    In this class the basis change matrix a_basis_b satisfies the equation
    v_a = a_basis_b * v_b
    where a and b are bases and v_i is a vector representing a nanotube state
    in the basis i.

    Bases used in this class:
    'default'
        Kup K'down Kdown K'up
    'simple'
        Kup Kdown K'up K'down
    'user'
        Specified by the user at class instantiation as the parameter basis
        in the __init__ function. If basis is not specified the 'user' basis
        is identical to the 'default' basis.
    """
    # Pauli matrices
    pauliX = np.matrix([[0,1],[1,0]], dtype=np.complex128)
    pauliY = np.matrix([[0,-1j],[1j,0]], dtype=np.complex128)
    pauliZ = np.matrix([[1,0],[0,-1]], dtype=np.complex128)
    pauliI = np.eye(2, dtype=np.complex128)
    # Matrix that changes from the 'default' basis
    # Kup K'down Kdown K'up
    # to the 'simple' basis
    # Kup Kdown K'up K'down
    simple_basis_def = np.matrix([
                           [1,0,0,0],
                           [0,0,1,0],
                           [0,0,0,1],
                           [0,1,0,0],
                       ])
    def_basis_simple = np.linalg.inv(simple_basis_def)
    # Spin matrices in the 'default' basis. The inner np.kron matrices are in
    # the 'simple' basis.
    Sx_def = def_basis_simple @ np.kron(pauliI, pauliX) @ simple_basis_def
    Sy_def = def_basis_simple @ np.kron(pauliI, pauliY) @ simple_basis_def
    Sz_def = def_basis_simple @ np.kron(pauliI, pauliZ) @ simple_basis_def

    def __init__(self, basis=None):
        """
        Parameters
        ==========
        basis : ndarray
            4x4 matrix which changes the basis of the spin matrices relative to
            the default basis Kup K'down Kdown K'up.
            basis satisfies the equation
                v_default = basis * v_user
            where v_default is a state in the default basis and v_user is a
            state in the user basis.
        """
        if basis is None:
            basis = np.eye(4)
        self.def_basis_user = basis
        self.set_spin_matrices()

    def set_spin_matrices(self):
        """
        Set the spin matrices in the basis specified by self.basis.
        """
        user_basis_def = np.linalg.inv(self.def_basis_user)
        self.Sx = user_basis_def @ self.Sx_def @ self.def_basis_user
        self.Sy = user_basis_def @ self.Sy_def @ self.def_basis_user
        self.Sz = user_basis_def @ self.Sz_def @ self.def_basis_user

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
