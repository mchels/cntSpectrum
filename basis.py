import numpy as np
import sympy as sy

# Matrix that changes from the 'default' basis
# Kup K'down Kdown K'up
# to the 'simple' basis
# Kup Kdown K'up K'down
simple_basis_def = sy.Matrix([
    [1,0,0,0],
    [0,0,1,0],
    [0,0,0,1],
    [0,1,0,0],
])
def_basis_simple = simple_basis_def.inv()

def simple_to_def(arr):
    mat1 = def_basis_simple
    mat2 = simple_basis_def
    if isinstance(arr, np.ndarray):
        mat1 = sy_mat_to_np(mat1)
        mat2 = sy_mat_to_np(mat2)
    return mat1 @ arr @ mat2

def def_to_simple(arr):
    mat1 = simple_basis_def
    mat2 = def_basis_simple
    if isinstance(arr, np.ndarray):
        mat1 = sy_mat_to_np(mat1)
        mat2 = sy_mat_to_np(mat2)
    return mat1 @ arr @ mat2

def sy_mat_to_np(sy_mat, dtype=float):
    return np.array(sy_mat).astype(dtype)
