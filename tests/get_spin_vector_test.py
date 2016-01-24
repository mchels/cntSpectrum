import sys
sys.path.append('..')
import numpy as np
from cntspin import cntSpin

simple_basis_def = np.array([
                       [1,0,0,0],
                       [0,0,1,0],
                       [0,0,0,1],
                       [0,1,0,0],
                   ])
def_basis_simple = np.linalg.inv(simple_basis_def)

# Tests for basis Kup Kdown K'up K'down
def do_tests():
    cases = [
        # These states are simple basis states.
        {'state': np.array([1,0,0,0]), 'output': np.array([0,0,1])},
        {'state': np.array([0,1,0,0]), 'output': np.array([0,0,-1])},
        {'state': np.array([0,0,1,0]), 'output': np.array([0,0,1])},
        {'state': np.array([0,0,0,1]), 'output': np.array([0,0,-1])},
        # These states are equal mixes of two basis states.
        {'state': np.array([1,1,0,0]), 'output': np.array([2,0,0])},
        {'state': np.array([1,0,1,0]), 'output': np.array([0,0,2])},
        {'state': np.array([1,0,0,1]), 'output': np.array([0,0,0])},
        {'state': np.array([0,1,1,0]), 'output': np.array([0,0,0])},
        {'state': np.array([0,1,0,1]), 'output': np.array([0,0,-2])},
        {'state': np.array([0,0,1,1]), 'output': np.array([2,0,0])},
    ]
    bases = [
        np.eye(4),
    ]
    for basis in bases:
        foo = cntSpin(basis=basis)
        for case in cases:
            state = def_basis_simple @ case['state']
            actual_output = foo.get_spin_vector(state)
            desired_output = case['output']
            try:
                assert (actual_output == desired_output).all()
            except AssertionError:
                print('Assertion failed')
                print('State ' + str(case['state']))
                print('Actual output ' + str(actual_output))
                print('Desired output ' + str(desired_output))
                print('---------------------------')

if __name__ == '__main__':
    do_tests()
