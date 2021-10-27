# sparse_haar_play.py
import torch
# import matplotlib.pyplot as plt
# from sparse_haar_3 import SparseHaarEstimate
from sparse_haar.schemata import Schemata, Schema, SchemataConstructor
from framework import haar_basis_functions as hbf

torch.set_default_dtype(torch.float64)
ALPHAS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvxyz"\
    + "!£$%^&*()"


# parameters
dim = 1
max_j = 30
max_k = 30

# alphabets
j_alphas = ALPHAS[:2*max_j + 1]
k_alphas = ALPHAS[:2*max_k + 1]

x = torch.Tensor([[0.23, 0.26],
                  [0.23, 0.23],
                  [0.24, 0.26],
                  [4.23, 3.26],
                  [6.23, 5.26],
                  [0.23, 3.26],
                  [4.23, 9.26]])

y = torch.Tensor([[0.23, 0.26],
                  [0.23, 0.23]])

z = torch.Tensor([[0.23],
                  [0.26]])
a = torch.Tensor([[0.23]])
x = z

basis = hbf.HaarWavelet()


def value(x, j, k, max_j, max_k):
    return torch.pow(2, torch.tensor((j-max_j)/2))\
         * basis(torch.pow(2, torch.tensor(j-max_j)) * x - (k - max_k))


"""
def value(x, j, k, max_j, max_k):
    j2term = torch.log(torch.tensor(2)) * torch.tensor(j-max_j)
    return torch.exp(j2term/2) * basis(torch.exp(j2term) * x - (k - max_k))
"""

def augment_schemata(schemata, max_j, max_k):
    """
    In order to build the Kronecker product, we need to augment the schema
    so that the indices that are common (i.e. less than the min_j) are built.
    This means that we need to take, in each dimension, the lowest loc for each
    relevant input, and add that input to a key s.t. k=0 and j<{minimum
    relevant j}. Then we can build the Kronecker using this augmented Schemata.

    This is irrelevant if we just place in the min js as being -max_j
    """
    for d, schema in enumerate(schemata):
        # get the last element of the thing.
        pass


def evaluate(schemata,
             new_input,
             x,
             last_inputs={0, 1, 2, 3, 4, 5, 6, 7},
             d=0):
    """
    The aim of this function is to calculate, for a given set of inputs that
    have been observed, the evaluation of the estimate function. That is, to
    take the sum of the coefficients times the basis functions.

    This means that we are taking the hadamard product of the kronecker for
    the previous inputs and the kronecker for the evaluated input.

    This is equivalent to taking the kronecker product of the per-dim hadamard
    products.

    """
    # get the common keys for the new input and the base inputs
    # prepare an empty tensor
    full_dim = schemata.dim

    # matrix sizes
    matrix_width = (2 * max_j + 1) ** (full_dim - d)
    matrix_height = (2 * max_k + 1) ** (full_dim - d)
    nm_width = (2 * max_j + 1) ** (full_dim - d - 1)
    nm_height = (2 * max_k + 1) ** (full_dim - d - 1)
    this_term = 0

    # for each key in this dimension, build a matrix
    for key in sorted(schemata[d].keys()):
        j = ALPHAS.index(key[0])
        k = ALPHAS.index(key[1])
        inputs = schemata[d][key].intersection(last_inputs)

        new_input_value = value(new_input[0, d], j, k, max_j, max_k)
        sorted_inputs = sorted(list(inputs))
        for index in sorted_inputs:
            if d == full_dim - 1:
                this_term += value(x[index, d],
                                   j,
                                   k,
                                   max_j,
                                   max_k) * new_input_value
            else:
                result = evaluate(schemata,
                                  new_input,
                                  x,
                                  last_inputs={index},
                                  d=d+1) * new_input_value
                this_term += value(x[index, d],
                                   j,
                                   k,
                                   max_j,
                                   max_k) * result

    return this_term


def build_kronecker(schemata,
                    x,
                    last_inputs={0, 1, 2, 3, 4, 5, 6, 7}, d=0):
    """
    Builds the basic Kronecker product recursively.
    This part of the process builds up the Kronecker product matrix in a depth-
    first formulation.

    For each key in the first dimension, it places a matrix that is built from
    the relevant keys in the next dimension. As it goes down the tree, it
    will first build the matrix for the "first" index in each of the
    levels.

    The matrix in any given level is built as an appropriately sized matrix
    of zeros, and then each relevant input's contribution is added (i.e. with
    the summation operator) to this tensor of zeros.

    Since if there are overlapping locations in a given matrix, this means that
    the product and summation steps are in the wrong order. For this reason,
    we store separately the locations that hve summation so that they can
    be calculated separately.

    It now appears this may not be the case and I am wrong... need to check
    this!
    """
    # prepare an empty tensor
    full_dim = schemata.dim

    # matrix sizes
    matrix_width = (2 * max_j + 1) ** (full_dim - d)
    matrix_height = (2 * max_k + 1) ** (full_dim - d)
    nm_width = (2 * max_j + 1) ** (full_dim - d - 1)
    nm_height = (2 * max_k + 1) ** (full_dim - d - 1)
    this_tensor = torch.zeros(matrix_width, matrix_height)

    # for each key in this dimension, build a matrix
    for key in sorted(schemata[d].keys()):
        j = ALPHAS.index(key[0])
        k = ALPHAS.index(key[1])
        inputs = schemata[d][key].intersection(last_inputs)
        sorted_inputs = sorted(list(inputs))
        for index in sorted_inputs:
            if d == full_dim - 1:
                this_tensor[j, k] += value(x[index, d], j, k, max_j, max_k)
            else:
                try:
                    result = build_kronecker(schemata, x, last_inputs={index},
                                             d=d+1)
                    this_tensor[(j * nm_width):((j + 1) * nm_width),
                                (k * nm_height):((k + 1) * nm_height)] +=\
                         value(x[index, d], j, k, max_j, max_k) * result # build_kronecker(schemata,
                                                           #        x,
                                                           #        last_inputs=
                                                           #        {index},
                                                           #        d=d+1)
                except IndexError:
                    print("IndexError!")
                    breakpoint()
    return this_tensor
    """
        value = haar_basis_for_j_and_k
        this_tensor[(j * matrix_width):(j+1)*matrix_width,
                    (k * matrix_height):(k+1)*matrix_height] =
        {sum over the inputs in
        this_tensor[(j*matrix_width):(j+1)*matrix_width,
                (k*matrix_height):(k+1)*matrix_height]
    """


if __name__ == "__main__":
    new_input = torch.Tensor([[0.24, 0.26]])

    schemata_constructor = SchemataConstructor(dim, max_j, max_k)
    base_schemata = schemata_constructor.get_schemata(x[0, :].unsqueeze(0))
    # second_schemata = schemata_constructor.get_schemata(x[1, :].unsqueeze(0))
    # base_schemata = base_schemata + second_schemata

    schemata_list = []
    for this_input in x:
        schemata = schemata_constructor.get_schemata(this_input.unsqueeze(0))
        schemata_list.append(schemata)

    for sc in schemata_list[1:]:
        base_schemata = base_schemata + sc

    new_basis = hbf.HaarWaveletBasis(max_j, max_k)
    # print(base_schemata)
    kronecker = build_kronecker(base_schemata)
    real_kronecker = new_basis(0.23) + new_basis(0.26)
    print(kronecker - real_kronecker)
    # print(kronecker[kronecker != 0])
    interval = int(max_j/2)
    breakpoint()
