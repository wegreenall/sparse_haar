# sparse_haar_test.py
import unittest
import torch
from sparse_haar.sparse_haar_3 import SparseHaarEstimate
import framework.haar_basis_functions as hbf
from framework import utils
from sparse_haar.schemata import SchemataConstructor


class TestSparseHaarEstimate(unittest.TestCase):
    """
    Tests to run:
        - add an input - check that the estimate
        - kronecker matrix generation
    """
    def setUp(self):
        # setup parameters for the estimate
        self.dim = 2
        self.max_j = 20
        self.max_k = 20

        # two test_inputs
        self.test_input_1 = torch.Tensor([[0.23, 0.26]])
        self.test_input_2 = torch.Tensor([[0.23, 0.23]])

        # amalgamated test input
        self.test_input_3 = torch.Tensor([[0.23, 0.26],
                                          [0.23, 0.23]])

        # basis functions
        self.new_basis = hbf.HaarWaveletBasis(self.max_j, self.max_k)
        self.basis_function = hbf.HaarWaveletBasisFunction()
        self.sparse_haar_estimate = SparseHaarEstimate(self.dim,
                                                       self.max_j,
                                                       self.max_k,
                                                       self.basis_function)

        self.schemata_constructor = SchemataConstructor(self.dim,
                                                        self.max_j,
                                                        self.max_k)
        return

    def test_kronecker(self):
        # to test the Kronecker - add two inputs to it and get the result

        this_sparse_haar_estimate = SparseHaarEstimate(self.dim,
                                                       self.max_j,
                                                       self.max_k,
                                                       self.basis_function)

        this_sparse_haar_estimate.add_inputs(self.test_input_1)
        this_sparse_haar_estimate.add_inputs(self.test_input_2)

        # generate the Kronecker
        kronecker_matrix = this_sparse_haar_estimate.get_kronecker()

        # check if the kronecker is the same
        real_kronecker = utils.kronecker(self.new_basis(0.23),
                                         self.new_basis(0.26)) +\
            utils.kronecker(self.new_basis(0.23),
                            self.new_basis(0.23))  # +\

        # self.assertTrue((real_kronecker == kronecker_matrix).all())
        self.assertTrue((abs(real_kronecker - kronecker_matrix) < 1e-14).all())
        return

    def test_add_inputs(self):
        """
        Tests that addition of inputs results in correct updating of the
        estimate data.
        """
        this_sparse_haar_estimate = SparseHaarEstimate(self.dim,
                                                       self.max_j,
                                                       self.max_k,
                                                       self.basis_function)
        # add the inputs
        this_sparse_haar_estimate.add_inputs(self.test_input_1)
        this_sparse_haar_estimate.add_inputs(self.test_input_2)

        # check whether the addition occurred as expected
        haar_schemata = this_sparse_haar_estimate.get_schemata()

        schemata1 = self.schemata_constructor.get_schemata(self.test_input_1)
        schemata2 = self.schemata_constructor.get_schemata(self.test_input_2)
        net_schemata = schemata1 + schemata2

        self.assertEqual(haar_schemata, net_schemata)
        self.assertEqual(this_sparse_haar_estimate.input_count, 2)
        return

    def test_add_multiple_inputs(self):
        """
        Tests that addition of inputs results in correct updating of the
        estimate data.
        """
        # add the inputs
        first_sparse_haar_estimate = SparseHaarEstimate(self.dim,
                                                        self.max_j,
                                                        self.max_k,
                                                        self.basis_function)

        second_sparse_haar_estimate = SparseHaarEstimate(self.dim,
                                                         self.max_j,
                                                         self.max_k,
                                                         self.basis_function)
        # check whether the addition occurred as expected
        # (the result should be the same as the previous one as well)

        first_sparse_haar_estimate.add_inputs(self.test_input_3)

        second_sparse_haar_estimate.add_inputs(self.test_input_1)
        second_sparse_haar_estimate.add_inputs(self.test_input_2)

        self.assertEqual(first_sparse_haar_estimate.get_schemata(),
                         second_sparse_haar_estimate.get_schemata())
        self.assertEqual(first_sparse_haar_estimate.input_count, 2)
        return

    @unittest.skip("Too long")
    def test_predict(self):
        """
        Should be able to generate a prediction that is correct given the
        inputs that have been added.
        """
        # build the estimate
        sparse_haar_estimate = SparseHaarEstimate(self.dim,
                                                  self.max_j,
                                                  self.max_k,
                                                  self.basis_function)
        sparse_haar_estimate.add_inputs(self.test_input_3)

        # build the basis
        basis = hbf.HaarWaveletBasis(self.max_j, self.max_k)

        real_kronecker = utils.kronecker(self.new_basis(0.23),
                                         self.new_basis(0.26)) +\
            utils.kronecker(self.new_basis(0.23),
                            self.new_basis(0.23))  # +\

        test_input = torch.Tensor([[0.29, 0.3]])
        test_kronecker = utils.kronecker(self.new_basis(0.29),
                                         self.new_basis(0.3))

        matrix = basis(self.test_input_3)

        return


if __name__ == "__main__":
    unittest.main()
