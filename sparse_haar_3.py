# sparse_haar_3.py
import torch
import collections
from framework.utils import print_dict
from sparse_haar.schemata import Schema, Schemata, SchemaConstructor, SchemataConstructor,\
            IndexConstructor
from framework.haar_basis_functions import HaarWaveletBasis

ALPHAS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvxyz"\
    + "!£$%^&*()"


class Christ(list):
    def __mul__(self, other):
        if type(other) == int:
            new_list = []
            for s in self:
                new_list.append(other * s)
            return new_list


class SparseHaarEstimate:
    def __init__(self,
                 dim,
                 max_j,
                 max_k,
                 basis):

        self.dim = dim

        # alphabets
        self.j_alphas = ALPHAS[:2*max_j + 2]
        self.k_alphas = ALPHAS[:2*max_k + 2]
        # breakpoint()
        # maximum indices
        self.max_j = max_j
        self.max_k = max_k

        self.input_count = 0

        self.word_sets = []  # this will expand in the N dimension

        # initialise the Schemata for storing the data
        self.schemata_constructor = SchemataConstructor(dim,
                                                        max_j,
                                                        max_k)

        # factories
        self.schema_constructor = SchemaConstructor(dim, max_j, max_k)
        self.commons = Schema()

        self.basis = basis
        self.x = torch.Tensor([])
        return

    def get_input_indices(self):
        return self.word_sets

    def add_inputs(self, x):
        """
        Constructs the set, in 'jkv' notation, of locations for the semi COO
        format of the tensor product matrix. Then, it updates the sets
        on the class.

        Step 1 - build the set of inputs in 'jkv' format for the new input set

        Step 2 - find the common indices and add their inputs
        """

        # check dimension of input corresponds to dim of estimate
        if x.shape[1] != self.dim:
            raise RuntimeError("The dimension of the input must match that " +
                               "of the estimate. Current estimate dim: " +
                               " {} ".format(self.dim))
        # add the data - this should be a different thing
        self.x = torch.cat((self.x, x))

        # process the new input to get the two components
        new_schemata = self.schemata_constructor.get_schemata(x)

        if self.input_count == 0:
            self.schemata = new_schemata
        else:
            # add the new input; the set component
            self.schemata = self.schemata + new_schemata
            new_commons = self.schemata.combine(new_schemata)

            # store the common locations; the powerset component
            self.commons = self.commons + new_commons

        # increment the thing.
        self.input_count += x.shape[0]
        return

    def get_kronecker(self):
        """
        Returns the Kronecker product matrix for the current data. It will
        do this by using separately the 'orthogonal' aspects of the data repre-
        sentation.
        """
        inputs_set = set(range(self.input_count))
        kronecker = self._build_kronecker(set(range(self.input_count)))
        return kronecker

    def get_schemata(self):
        return self.schemata

    def _build_kronecker(self,
                         last_inputs,
                         d=0):
        """
        Builds the basic Kronecker product recursively.
        This part of the process builds up the Kronecker product matrix in a
        depth-first formulation.

        For each key in the first dimension, it places a matrix that is built
        from the relevant keys in the next dimension. As it goes down the tree,
        it will first build the matrix for the "first" index in each of the
        levels.

        The matrix in any given level is built as an appropriately sized matrix
        of zeros, and then each relevant input's contribution is added (i.e.
        with the summation operator) to this tensor of zeros.

        Since if there are overlapping locations in a given matrix, this means
        that the product and summation steps are in the wrong order. For this
        reason, we store separately the locations that hve summation so that
        they can be calculated separately.

        It now appears this may not be the case and I am wrong... need to check
        this!
        """
        # prepare an empty tensor
        full_dim = self.dim

        # matrix sizes
        matrix_width = (2 * self.max_j + 1) ** (full_dim - d)
        matrix_height = (2 * self.max_k + 1) ** (full_dim - d)
        nm_width = (2 * self.max_j + 1) ** (full_dim - d - 1)
        nm_height = (2 * self.max_k + 1) ** (full_dim - d - 1)
        this_tensor = torch.zeros(matrix_width, matrix_height)

        # for each key in this dimension, build a matrix
        for key in self.schemata[d]:
            j = self.j_alphas.index(key[0])
            k = self.k_alphas.index(key[1])
            inputs = self.schemata[d][key].intersection(last_inputs)

            for index in inputs:
                if d == full_dim - 1:
                    this_tensor[j, k] += self.basis(self.x[index, d],
                                                    j-self.max_j,
                                                    k-self.max_k)
                else:
                    try:
                        result = self._build_kronecker({index}, d=d+1)
                        this_tensor[(j * nm_width):((j + 1) * nm_width),
                                    (k * nm_height):((k + 1) * nm_height)] +=\
                                self.basis(self.x[index, d], j-self.max_j,
                                           k-self.max_k) * result


                    except IndexError:
                        print("IndexError!")
                        breakpoint()

        return this_tensor


if __name__ == "__main__":
    y1 = torch.Tensor([[1.23, 1.23]])
    y2 = torch.Tensor([[2.23, 2.23]])
    y3 = torch.Tensor([[3.25, 3.25]])

    y = torch.Tensor([[0.23, 0.23],
                     #  [0.23, 0.22],
                      #  [0.23, 0.23],
                      #  [0.23, 1.24],
                      #  [0.23, 2.24],
                      #  [0.23, 3.24],
                      #  [0.23, 2.59],
                      #  [0.23, 3.42],
                      #  [0.23, 4.89],
                      #  [0.23, 2.14],
                      #  [0.23, 8.94],
                      #  [0.23, 5.14],
                      #  [0.23, 6.88],
                      #  [0.23, 0.24],
                      [5.6, 8.32]])
    x1 = torch.Tensor([[0.23, 0.43]])
    x2 = torch.Tensor([[0.23, 0.26]])
    #  x3 = torch.Tensor([[0.23, 0.23]])
    x = torch.Tensor([[0.23, 0.29],
                      [0.25, 0.29]])
    dim = 2
    max_j = 20
    max_k = 20
    estimate = SparseHaarEstimate(dim, max_j, max_k)
    estimate_2 = SparseHaarEstimate(dim, max_j, max_k)

    print("About to add the inputs")
    #    estimate.add_inputs(x1)
    estimate.add_inputs(x2)
    # estimate_2.add_inputs(x)
    # estimate.check_input(x3)
    # estimate.add_inputs(y)
    final_schemata = estimate.schemata
    final_schemata2 = estimate_2.schemata
    for diction in final_schemata:
        print("Next dict:")
        print_dict(diction)

    # for diction in final_schemata2:
    #    print("Next dict:")
    #    print_dict(diction)
    breakpoint()
