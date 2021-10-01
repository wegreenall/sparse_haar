# haar_basis_functions.py
# from eigenvalue_solution import WSM
import torch
# import matplotlib as mpl
# mpl.use('Qt4Agg')
import matplotlib.pyplot as plt

from framework.WGPP import PoissonProcess
from framework.wgcycler import wcyclerRTB
# from collections import namedtuple
from framework.sparse_haar_kronecker import Edge


# implementing Haar wavelets
def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),
                                                  A.size(1)*B.size(1))


def plot_matrix(matrix, interval, show=True, this_marker='+'):
    for j in range(-interval, interval):
    #    plt.scatter(range(-interval, interval),
    #                2*interval*[j], marker='_',
    #                linewidth=0.1)
        for k in range(-interval, interval):
            # plt.scatter(range(-interval,
            # interval), k*[1], marker='_', linewidth=0.1)
            var_1 = matrix[j + interval, k + interval]
            if var_1 > 0:
                plt.scatter(k, -j, marker=this_marker, linewidth=0.5)
                # print("where not zero, k is:", k)
            elif var_1 < 0:
                plt.scatter(k, -j, marker='o', linewidth=0.5)
            # else:
            #     plt.scatter(k, -j, marker='_', linewidth=0.1)
    if show:
        plt.show()
    return


def get_matrix(x, max_j, max_k):
    my_basis = HaarWaveletBasis(max_j, max_k)
    results = torch.sign(my_basis(x))

    return results


class Wavelet:
    def __init__(self):
        pass

    def __call__(self):
        pass


class Scaler:
    def __init__(self):
        pass

    def __call__(self):
        pass


class HaarWavelet(Wavelet):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return torch.zeros(x.shape) + 1 * (x >= 0) * (x < 0.5) \
            - 1 * (x >= 0.5) * (x < 1)


class HaarScaler(Scaler):
    def __call__(self, x):
        return torch.zeros(x.shape) + 1 * (x > 0) * (x < 1)


class WaveletBasis():
    def __init__(self, max_j, max_k):
        self.max_j = max_j
        self.max_k = max_k
        js = torch.Tensor([j for j in range(-max_j, max_j)])
        ks = torch.Tensor([k for k in range(-max_k, max_k)])

        self.grid_js, self.grid_ks = torch.meshgrid(js, ks)
        return

    def __call__(self, x):
        first = torch.pow(torch.tensor(2), self.grid_js/2)
        psi = first * self.wavelet((2**self.grid_js) * x - self.grid_ks)
        return psi


class HaarWaveletBasis(WaveletBasis):
    def __init__(self, max_j, max_k):
        super().__init__(max_j, max_k)
        self.wavelet = HaarWavelet()
        return super(HaarWaveletBasis, self).__init__(max_j, max_k)


class HaarScalerBasis(WaveletBasis):
    def __init__(self, max_j, max_k):
        super().__init__(max_j, max_k)
        self.scaler = HaarScaler()
        return super(HaarScalerBasis, self).__init__(max_j, max_k)

    def __call__(self, x):
        first = torch.pow(torch.tensor(2), self.grid_js/2)
        first = 1
        psi = first * self.scaler((2**self.grid_js)*x - self.grid_ks)
        return psi


class HaarEstimate(object):
    def __init__(self, sample, max_j, max_k, window):
        super(HaarEstimate, self).__init__()
        self.basis = HaarWaveletBasis(max_j, max_k)
        self.scaler = HaarScalerBasis(max_j, max_k)
        self.max_j = max_j
        self.max_k = max_k
        self.sample = sample
        self.window = window

        self.wavelet_coeffics, self.scaler_coeffics = self._get_coefficients()
        return

    def _format_input(self, x):
        """
        Formats potential inputs to be of the shape required for the Haar basis
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(1)

        x = x.expand(2*self.max_j, len(x),  2*self.max_k)
        x = torch.einsum('ijk->jik', x)
        return x

    def __call__(self, x):
        # return the value of the estimate with the given sample
        # (and thus coefficients)
        x = self._format_input(x)
        estimate = torch.einsum('ijk->i',
                                self.wavelet_coeffics *
                                self.basis(x))

        #  breakpoint()
        # estimate += torch.einsum('ijk->i',
        #                         self.scaler_coeffics *
        #                         self.scaler(x,
        #                                     self.grid_js,
        #                                     self.grid_ks))
        return estimate

    def _get_coefficients(self):
        """
        Returns the coefficients for a Haar basis estimate of a given intensity
        function. To do this we take the Haar function system and calculate the
        inner product between "the function" we're building and the Haar
        function, just as in the standard basis method.while

        This will involve summing up the values of the corresponding basis
        function.
        """
        # get local variables
        sample = self._format_input(self.sample)

        wavelet_basis = self.basis(sample)
        scaler_basis = self.scaler(sample)
        wavelet_coeffics = torch.sum(wavelet_basis,
                                     dim=0)
        # breakpoint()
        scaler_coeffics = torch.sum(scaler_basis,
                                    dim=0)
        return wavelet_coeffics, scaler_coeffics


class SparseHaarKronecker(object):
    def __init__(self, js, ks, signs):
        super(self, SparseHaarKronecker).__init__()
        """
        Sets up the edges data.
        """
        self.edges = {}

        return

    def get_edges(self):
        pass

    def _add_input(self, x):
        """
        adds an input in the form of a SparseHaarKronecker value for an input.
        I.e. it conducts the necessary operation on the edge-set for the
        purpose of storing the graph.
        """


class EdgeSet:
    def __init__(self):
        self.edges = []
        pass

    def update_edges(self, js, new_haar_data):
        input_count = new_haar_data.shape[1]
        for i in range(input_count):
            this_input = ks[:, i, :]  # [jcount, dim]
            self.edges.append(Edge((this_input)))


class SparseHaarEstimate(object):
    def __init__(self, index_dict, dim, inputs=torch.Tensor([])):
        super(SparseHaarEstimate, self).__init__()
        self.lj = index_dict['lower_j']
        self.uj = index_dict['upper_j']
        self.lk = index_dict['lower_k']
        self.uk = index_dict['upper_k']

        self.dim = dim
        self.haar_data = torch.Tensor([])
        self.edges = EdgeSet()
        self.input_count = 0
        if isinstance(inputs, torch.Tensor):
            self.inputs = inputs
        else:
            raise ValueError("inputs needs to be of type torch.Tensor.")

    def add_inputs(self, new_inputs):
        """
        Update inputs list
        """
        self.inputs = torch.cat((self.inputs, new_inputs))
        self.input_count += new_inputs.shape[0]

        # update the edge set
        self.update_edges(new_inputs)

    def get_inputs(self):
        return self.inputs

    def get_ks(self, x, js):
        """
        Returns the set of k indices for a corresponding set of j indices, for
        an x of whatever dimension necessary
        """
        js2 = torch.pow(2, js+1).repeat([self.dim, x.shape[0], 1])
        js2 = torch.einsum('ijk -> jki', js2)  # so [n, relevant_js, d]
        xview = torch.einsum('ijk->jik', x.repeat(js.shape[0], 1, 1))

        ks = torch.einsum('ijk->jik', torch.ceil(xview * js2)/2)

        return ks  # [relevant_js, n, d]

    def update_edges(self, new_inputs):
        self.haar_data = torch.cat((self.haar_data, new_inputs))

    def evaluate(self, x):
        """
        Returns the set of js and corresponding ks for the Haar wavelet tensor
        implied by the input x.

        This is a sparse tensor and so can be described using relatively few
        terms. When using it to get coefficients, the redundant coefficients
        are added.
        """

        self.add_inputs(x)

        if x.shape[1] != self.dim:
            raise ValueError("Dimension of input not matched to\
                              dimension of model")

        max_zero_j = torch.min(torch.floor(- torch.log2(x) - 1))
        min_nonzero_j = torch.max(torch.floor(torch.log2((1 + self.uk) / x)))
        js = torch.linspace(int(max_zero_j),
                            int(min_nonzero_j),
                            int(min_nonzero_j - max_zero_j + 1)
                            )

        ks = self.get_ks(x, js)

        values = torch.where(ks % 1 != 0,
                             torch.ones(ks.shape),
                             -torch.ones(ks.shape))

        # having acquired the signs information, push back to integers
        ks = torch.floor(ks)
        # breakpoint()
        haar_data = torch.stack((ks, values), dim=3)

        #  haar_data represent the haar tensor for a set of inputs (dim 1) and
        #  as many dimensions as necessary.
        #  for each input, the j k pairs will be unique in each dim.
        #  if you concatenate two inputs, you will get duplicate indices for
        #  the
        #  two inputs and you will
        #  get the data for each pair. You want to sum these, I think.

        # breakpoint()
        # js_shaped = js * torch.
        return js, haar_data

    # def __getitem__(self, loc):
    #     """
    #     Having supplied the estimate with data,
    #     allow one to get the value at a given index. To do this we can mimic
    #     indexing behaviour for the implied Kronecker by 'travelling' to
    #     indices,  returning the corresponding value.

    #     The Kronecker tensor will be indexed from lower j to upper j, so
    #     SparseHaareEstimate[-1, -5] refers to the j, k
    #     associated with these numbers, rather than the negative indexing
    #      paradigm in pytorch.
    #     """

    #     # check if too many dimensions
    #     if len(loc) != self.dim:
    #         raise IndexError("Not enough indices. Expecting {expected},
    #         received {actual}".format(expected=self.dim, actual=len(loc)))

    #     # check if the indices chosen are out of bounds

    #     return loc


if __name__ == "__main__":

    def intensity(x):
        return 32000*torch.exp(torch.distributions.gamma.Gamma(
                                                torch.Tensor([11.]),
                                                torch.Tensor([8.])).log_prob(x)
                               )
    test_estimate = False
    draw_diagram = True
    test_sparse_haar = False
    test_sparse_kron = False

    if test_estimate:
        # parameters
        window = torch.Tensor([[0], [5]])

        estimate_window = torch.Tensor([[0], [5]])
        pp1 = PoissonProcess(intensity, window, 1)
        sample = pp1.sample()

        max_k = 30
        max_j = 40
        estimate = HaarEstimate(sample, max_j, max_k, estimate_window)

        """
        When the support of the function is almost all in the interval over
        the range of the points, the estimate can be accurately fixed
        (comparing the mean height of the function with the mean height of
        the estimate) by multiplying by the size of the window. However, I do
        not understand exactly why....
        """
        # breakpoint()
        # estimate.get_coefficients()
        x = torch.linspace(0.01, 15, 1000)

        print(torch.mean(intensity(x)/torch.mean(estimate(x))))
        print(max(sample))
        plt.plot(x, estimate(x))
        plt.plot(x, intensity(x))
        plt.show()

    if draw_diagram:
        z = torch.linspace(-20, 40, 1000)
        x = 1.234
        y = 0.8
        z = 2.468
        a = 0.5
        k = 1
        j = 1

        max_j = 30
        max_k = 20
        my_basis = HaarWaveletBasis(max_j, max_k)
        line_count = 25
        cycler = wcyclerRTB(int(line_count/4))
        interval = int(max_j/2)

        # mpl.rcParams['axes.prop_cycle'] = cycler
        results_1 = get_matrix(x, max_j, max_k)
        results_2 = get_matrix(y, max_j, max_k)
        results_3 = get_matrix(z, max_j, max_k)
        results_4 = get_matrix(a, max_j, max_k)

        # plot_scatter = True
        # print(results_1)
        kron_1 = kronecker(results_1, results_2)
        kron_2 = kronecker(results_3, results_4)
        kron_sum = kron_1 + kron_2

        plot_matrix(results_2, interval, show=True, this_marker='x')
        plot_matrix(results_1, interval)
        plot_matrix(kron_sum, 2*interval*interval, show=True)

        # breakpoint()
        # plt.plot(y * torch.ones(10),
        #        torch.linspace(-3, 3, 10),
        #        color='black',
        #        linewidth=0.5)
        plt.show()

    if test_sparse_haar:
        index_dict = {
            'lower_j': -20,
            'upper_j': 20,
            'lower_k': -50,
            'upper_k': 50
        }
        dim = 2
        x = torch.Tensor([[0.4455, 6.82476], [1.23, 1.992], [1.456, 2.39568]])

        she = SparseHaarEstimate(index_dict, dim)
        breakpoint()
        js, ks = she.evaluate(x)

        # testing some things related to the kronecker and getting the
        # appropriate strides
        z = torch.linspace(0, 15, 16)
        z = z.reshape([4, 4])
        z2 = kronecker(z, z)
        z3 = kronecker(z2, z)

        breakpoint()
        # print(kronecker[1, 3, 2])

    if test_sparse_kron:
        j1 = 1
        k1 = 3
        j2 = 2
        k2 = 3
        j3 = 1
        k3 = 3
        val1 = 1
        val2 = -1
        edge_1 = Edge((j1, k1), (j2, k2), val1)
        edge_2 = Edge((j1, k1), (j3, k3), val2)
        breakpoint()
