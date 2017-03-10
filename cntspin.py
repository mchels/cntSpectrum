import numpy as np

class cntSpin(object):
    r"""
    Class for getting expected value of <S_vector> = (<Sx>, <Sy>, <Sz>) for
    carbon nanotube states given a basis.
    Currently, only fillings 1 and 3 are supported, that is, all matrices are
    of size 4x4.

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
    pauliX = np.array([[0,1],[1,0]], dtype=np.complex128)
    pauliY = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
    pauliZ = np.array([[1,0],[0,-1]], dtype=np.complex128)
    pauliI = np.eye(2, dtype=np.complex128)
    # Matrix that changes from the 'default' basis
    # Kup K'down Kdown K'up
    # to the 'simple' basis
    # Kup Kdown K'up K'down
    simple_basis_def = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
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
        assert np.allclose(spin_exp_vector.imag, 0)
        return spin_exp_vector.real

    def get_spin_vectors_from_eigenstates(self, eigenstates):
        """
        Calculate expectation value of the spin vector for all states in
        eigenstates.

        Parameters
        ==========
        eigenstates : ndarray
            Any array for which the two inner-most dimensions both have size 4.
            The inner-most dimension should contain the eigenstate components.
            This parameter is designed to be provided by the 'states' output
            from the cntspectrum.get_spectrums function. The 'states' output has
            dimensions of
                n_B_steps x n_B_angle_steps x n_states x n_states

        Returns
        =======
        spin_vectors : ndarray
            Contains the expectation value of the spin vector for all states in
            the input eigenstates array.
            If the 'states' output from the cntspectrum.get_spectrums function
            is used as the eigenstates array above the spin_vectors array will
            have dimensions of
                n_B_steps x n_B_angle_steps x n_states x spatial_dimensions
            where spatial_dimensions is 3 (obviously).

        Notes
        =====
        This function depends on self through the basis used in
        self.get_spin_vector.
        """
        spin_dimensions = 3
        # Use the same shape for the spin_vectors array as for the eigenstates
        # array EXCEPT that we want three entries (<Sx>, <Sy>, <Sz>) instead of
        # four entries (the components of the four nanotube states) in the last
        # dimension.
        shape = eigenstates.shape[:-1] + (spin_dimensions,)
        spin_vectors = np.zeros(shape=shape)
        def loop_recursively(eigenstates, spin_vectors, indices=None):
            """
            For all states in the eigenstates array save the corresponding spin
            vector in array spin_vectors.
            A spin vector will have the same indices in spin_vectors as the
            corresponding state in eigenstates.
            The lowest two dimensions of the eigenstate array must contain a
            n_states x n_states matrix with eigenvectors specified as columns.
            """
            n_states = 4
            assert eigenstates.shape[-2:] == (n_states,n_states)
            if indices is None:
                indices = ()
            if eigenstates.shape == (n_states,n_states):
                for k, state in enumerate(eigenstates.T):
                    spin_vector = self.get_spin_vector(state)
                    spin_vectors[indices+(k,)] = spin_vector
            elif len(eigenstates.shape) > 2:
                for i in range(len(eigenstates)):
                    loop_recursively(eigenstates[i], spin_vectors, indices=indices+(i,))
            else:
                error_str = (
                    'This should never occur because of the assertion in the '
                    'first line of the loop_recursively function.'
                )
                raise RuntimeError(error_str)
        loop_recursively(eigenstates, spin_vectors)
        return spin_vectors
