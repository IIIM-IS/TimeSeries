"""
Permutation entropy brute force computation

Created : Mon Nov  5 10:43:54 GMT 2018
Author : David Orn, david@iiim.is


Description:
Using papers like
Brand et al 2002 : Permuation entropy - a natural complexity measure for time series

and
Mariano Matilla Garcia et al, 2008 : A non-Parametric independence test using permutation entropy
(url : https://www.sciencedirect.com/science/article/pii/S030440760800002X)

We reconstruct a brute force approach to the Permutation Entropy computation
"""
import itertools  # To generate the permutation list of a range
import numpy as np


# Debug
import pdb


def pe_bf(data, d, tau=1):
    """
    Perform a linear brute force expression of the complexity of a
    time series object using the permutation entropy as the
    measurement in question.

    *args*
        data : np.array: Time series object of length N
        d : integer : Embedding dimension >= 2
        tau : integer : Time delay, distance between selected points
    """
    p = np.zeros((np.math.factorial(d)), dtype=np.int32)  # Compute possible types of permutations
    sym = list(itertools.permutations(range(0,d)))  # List all available permutations

    n = len(data)
    for indx in range(0,n-(d-1)*tau):
        # Create an set of indexes to select the next values
        indx_set = range(indx, indx+(d*tau), tau)
        # Select the indexed values into a numpy array
        vals = np.array(data[indx_set])
        # Get the sorted index value of the values
        set_indx = np.argsort(vals)

        # Compare the value set with the symbols
        sym_indx = 0
        while True:
            if (sym[sym_indx] == set_indx).all():
                break
            else:
                sym_indx += 1
        p[sym_indx] += 1

    # Remove all values from p which are zero, since log(0) isn't defined
    p = p[p != 0]
    print(p)
    # Get overall number of samples
    sum = np.sum(p)
    # Compute each sample size, normalized
    p = p/sum
    # Compute entropy
    EP = [-(x*np.log(x)) for x in p]
    return np.sum(EP)


def test_function():
    import matplotlib.pyplot as plt
    from timeit import default_timer as timer

    # Sample from Brandt et al 2002
    sample = np.array([4, 7, 9, 10, 6, 11, 3])
    val = pe_bf(sample, 3, tau=1)
    print("Brand et al: Sample entropy= {:.2f}".format(val))
    # Sample from https://www.sciencedirect.com/science/article/pii/S030440760800002X
    X = np.array([2, 8, 6, 5, 4, 9, 3])
    val = pe_bf(X, 3, tau=1)
    print("Matilla-Garcia et al : Permutation entropy = {:.3f}".format(val))


if __name__ == "__main__":
    test_function()
