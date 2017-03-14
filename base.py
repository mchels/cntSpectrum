import numpy as np

def get_ex_spectrums(spectrums, n_occ_states, get_neg=True):
    r"""
    Parameters
    ----------
    spectrums : array-like
        An array containing energies (eigenvalues) corresponding to some
        matrix.
    n_occ_states : integer
        Number of occupied states.
    get_neg : Boolean
        Decides whether negative excitation energies are returned as well. The
        negative excitation energies are obtained by simply mirroring the
        positive ones.

    Returns
    -------
    ex_spectrums : ndarray
        Excitation spectrum.
    """
    assert isinstance(n_occ_states, int)
    assert n_occ_states > 0
    occ_Es = spectrums[...,:n_occ_states]
    non_occ_Es = spectrums[...,n_occ_states:]
    # Insert np.newaxis to calculate all combinations of occupied and
    # unoccupied energies.
    ex_spectrums = non_occ_Es[...,:,np.newaxis] - occ_Es[...,np.newaxis,:]
    # Flatten the inner-most two dimensions.
    newshape = spectrums.shape[:-1] + (-1,)
    ex_spectrums = np.reshape(ex_spectrums, newshape=newshape)
    ex_spectrums = np.sort(ex_spectrums, axis=-1)
    if get_neg:
        # Stack negative excitation energies with the positive ones.
        ex_spectrums = np.concatenate([ex_spectrums, -ex_spectrums], axis=-1)
    return ex_spectrums
