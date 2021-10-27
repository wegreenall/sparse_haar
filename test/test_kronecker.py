# kronecker_test.py
import unittest
import torch
from sparse_haar.schemata import SchemataConstructor
from sparse_haar.sparse_haar_play import build_kronecker, value, evaluate
from framework import haar_basis_functions as hbf
from framework import utils
# torch.set_default_dtype(torch.float64)

ALPHAS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvxyz"\
    + "!£$%^&*()"


def get_schemata(x, schemata_constructor):
    base_schemata = schemata_constructor.get_schemata(x[0, :].unsqueeze(0))
    schemata_list = []
    for this_input in x:
        schemata = schemata_constructor.get_schemata(this_input.unsqueeze(0))
        schemata_list.append(schemata)

    for sc in schemata_list[1:]:
        base_schemata = base_schemata + sc

    return base_schemata


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        # parameters
        self.max_j = 30
        self.max_k = 30

        # parameters
        self.x1 = torch.Tensor([[0.23, 0.23]])

        # haar basis stuff
        self.new_basis = hbf.HaarWaveletBasis(self.max_j, self.max_k)

        self.one_d_schemata_constructor = SchemataConstructor(1,
                                                              self.max_j,
                                                              self.max_k)

        self.two_d_schemata_constructor = SchemataConstructor(2,
                                                              self.max_j,
                                                              self.max_k)

        self.three_d_schemata_constructor = SchemataConstructor(3,
                                                                self.max_j,
                                                                self.max_k)

    def test_evaluation(self):
        """
        First, get the standard, true parameter kronecker
        """
        x = torch.Tensor([[0.23, 0.26],
                          [0.23, 0.23],
                          [0.58, 0.99]])

        # get the kronecker for the seen data
        data_kronecker = utils.kronecker(self.new_basis(0.23),
                                         self.new_basis(0.26)) +\
            utils.kronecker(self.new_basis(0.58), self.new_basis(0.99)) +\
            utils.kronecker(self.new_basis(0.23),
                            self.new_basis(0.23))

        # get the kronecker for the new input for evaluation
        new_input = torch.Tensor([[0.24, 0.24]])
        new_kronecker = utils.kronecker(self.new_basis(0.24),
                                        self.new_basis(0.24))

        # calculate the evaluation
        result_kronecker = data_kronecker * new_kronecker
        evaluation = torch.sum(data_kronecker * new_kronecker)

        # get the testing kronecker
        base_schemata = get_schemata(x, self.two_d_schemata_constructor)
        new_schemata = get_schemata(x, self.two_d_schemata_constructor)
        edited_schemata = base_schemata.intersection(new_schemata)
        test_evaluation = evaluate(edited_schemata, new_input, x)
        print(evaluation)
        print(test_evaluation)
        self.assertTrue(((evaluation - test_evaluation) < 1e-14).all())
        return


class TestKroneckerBuilder(unittest.TestCase):
    def setUp(self):
        # parameters
        self.max_j = 30
        self.max_k = 30

        # alphabets
        # j_alphas = ALPHAS[:2*self.max_j + 1]
        # k_alphas = ALPHAS[:2*self.max_k + 1]

        # parameters
        self.x1 = torch.Tensor([[0.23, 0.23]])

        # haar basis stuff
        self.new_basis = hbf.HaarWaveletBasis(self.max_j, self.max_k)

        self.one_d_schemata_constructor = SchemataConstructor(1,
                                                              self.max_j,
                                                              self.max_k)

        self.two_d_schemata_constructor = SchemataConstructor(2,
                                                              self.max_j,
                                                              self.max_k)

        self.three_d_schemata_constructor = SchemataConstructor(3,
                                                                self.max_j,
                                                                self.max_k)

    def test_1d_1input(self):
        x = torch.Tensor([[0.23]])

        # get the real kronecker
        real_kronecker = self.new_basis(x)

        # get the testing kronecker
        base_schemata = get_schemata(x, self.one_d_schemata_constructor)
        kronecker = build_kronecker(base_schemata, x)

        self.assertTrue((abs(real_kronecker - kronecker) < 1e-14).all())
        return

    def test_1d_2input(self):

        x = torch.Tensor([[0.23],
                          [0.26],
                          [0.99]])
        # get the real kronecker
        real_kronecker = self.new_basis(0.23) \
            + self.new_basis(0.26) \
            + self.new_basis(0.99)

        # get the testing kronecker
        base_schemata = get_schemata(x, self.one_d_schemata_constructor)
        kronecker = build_kronecker(base_schemata, x)

        # self.assertTrue((real_kronecker == kronecker).all())
        self.assertTrue((abs(real_kronecker - kronecker) < 1e-14).all())
        return

    def test_2d_1input(self):
        x = torch.Tensor([[0.23, 0.26]])

        # get the real kronecker
        real_kronecker = utils.kronecker(self.new_basis(0.23),
                                         self.new_basis(0.26))

        # get the testing kronecker

        base_schemata = get_schemata(x, self.two_d_schemata_constructor)
        kronecker = build_kronecker(base_schemata, x)
        # self.assertTrue((real_kronecker == kronecker).all())
        self.assertTrue((abs(real_kronecker - kronecker) < 1e-14).all())
        return

    # @unittest.skipIf((__name__ != "__main__"), "Too long")
    @unittest.skip("Too long")
    def test_3d_1input(self):
        x = torch.Tensor([[0.23, 0.26, 0.99]])

        # get the real kronecker
        real_kronecker = utils.kronecker(utils.kronecker(self.new_basis(0.23),
                                         self.new_basis(0.26)),
                                         self.new_basis(0.99))

        # get the testing kronecker

        base_schemata = get_schemata(x, self.three_d_schemata_constructor)
        kronecker = build_kronecker(base_schemata, x)
        # self.assertTrue((real_kronecker == kronecker).all())
        self.assertTrue((abs(real_kronecker - kronecker) < 1e-14).all())
        return

    def test_2d_2input(self):
        x = torch.Tensor([[0.23, 0.26],
                          [0.23, 0.23]])

        # get the real kronecker
        real_kronecker = utils.kronecker(self.new_basis(0.23),
                                         self.new_basis(0.26)) +\
            utils.kronecker(self.new_basis(0.23),
                            self.new_basis(0.23))  # +\
#            utils.kronecker(self.new_basis(0.25),
#                            self.new_basis(0.99))

        # get the testing kronecker
        base_schemata = get_schemata(x, self.two_d_schemata_constructor)
        kronecker = build_kronecker(base_schemata, x)

        # self.assertTrue((real_kronecker == kronecker).all())
        self.assertTrue((abs(real_kronecker - kronecker) < 1e-14).all())
        return

    @unittest.skipIf((__name__ != "__main__"), "Too long")
    def test_2d_3input(self):
        x = torch.Tensor([[0.23, 0.26],
                          [0.23, 0.23],
                          [0.556, 0.99]])

        # get the real kronecker
        real_kronecker = utils.kronecker(self.new_basis(0.23),
                                         self.new_basis(0.26)) +\
            utils.kronecker(self.new_basis(0.556),
                            self.new_basis(0.99)) +\
            utils.kronecker(self.new_basis(0.23),
                            self.new_basis(0.23))
        # get the testing kronecker
        base_schemata = get_schemata(x, self.two_d_schemata_constructor)
        kronecker = build_kronecker(base_schemata, x)

        # self.assertTrue((real_kronecker == kronecker).all())
        self.assertTrue((abs(real_kronecker - kronecker) < 1e-14).all())
        return

#    @unittest.skipIf((__name__ != "__main__"), "Too long")
    @unittest.skip("Too long")
    def test_3d_3input(self):
        x = torch.Tensor([[0.23, 0.26, 0.44],
                          [0.23, 0.23, 0.23],
                          [0.556, 0.99, 0.678]])

        # get the real kronecker
        real_kronecker = utils.kronecker(utils.kronecker(self.new_basis(0.23),
                                         self.new_basis(0.26)),
                                         self.new_basis(0.44)) +\
            utils.kronecker(utils.kronecker(self.new_basis(0.556),
                            self.new_basis(0.99)), self.new_basis(0.678)) +\
            utils.kronecker(utils.kronecker(self.new_basis(0.23),
                            self.new_basis(0.23)), self.new_basis(0.23))
        # get the testing kronecker
        base_schemata = get_schemata(x, self.three_d_schemata_constructor)
        kronecker = build_kronecker(base_schemata, x)

        # self.assertTrue((real_kronecker == kronecker).all())
        self.assertTrue((abs(real_kronecker - kronecker) < 1e-14).all())
        return

    def test_basis_vs_value(self):
        # x = torch.Tensor([0.23])
        random_nums = torch.distributions.Normal(0, 1).sample([10])
        test_inputs = [x for x in random_nums]
        j = 25
        k = 27
        basis = hbf.HaarWaveletBasisFunction()
        for input in test_inputs:
            print(";", end='')
            value_result = value(input, j, k, self.max_j, self.max_k)
            basis_result = basis(input, j - self.max_j, k - self.max_k)
            self.assertEqual(value_result, basis_result)


if __name__ == "__main__":
    print("Testing Kronecker Builder")
    unittest.main()
    print("Testing Complete!")
