# sparse_haar_kronecker.py
import torch
# from collections import namedtuple
from itertools import product
# location = namedtuple("j", "k")


class Edge:
    def __init__(self, start_point, end_point, value):
        if isinstance(start_point, tuple) and isinstance(end_point, tuple):
            self.start_point = start_point
            self.end_point = end_point
            self.value = value
        else:
            raise ValueError("start_point and end_point should be tuples of\
length 2, containing tensors.")
        return

    def __eq__(self, other):
        if isinstance(other, Edge):
            return (self.start_point == other.get_start_point()
                    and self.end_point == other.get_end_point())

    def get_start_point(self):
        return self.start_point

    def get_end_point(self):
        return self.end_point

    def get_value(self):
        return self.value

    def __key(self):
        # this means that two edges with the same start and end points will
        # appear the same in the set of edges for the graph
        return (self.get_start_point(), self.get_end_point())

    def __hash__(self):
        # hashes based on the key
        return hash(self.__key())

    def __add__(self, other):
        if self == other:  # i.e. they have the same start and end point
            return Edge(self.start_point,
                        self.end_point,
                        self.value + other.value)


class Graph:
    """
    This class handles the storage and updating of the graph describing the
    dependency relations for the Kronecker representation. The hope is that
    the logic based on graphs is correct for d dimensions and it will work
    correctly.


    It contains a sequence of _dim_ sets that contain Edge objects.
    Each Edge object has a start point, end point and a value.
    The Edge class is designed so that addition implies addition of the value
    if and only if the start and end point are the same.

    """
    def __init__(self, dim):
        self.dim = dim
        self.edges = []
        return

    def update_edges(self, js, ks, values):
        """
        Updates the graph corresponding to the Kronecker estimate, building
        the edges and values up
        """
        for d in range(self.dim):
            # get the edges for the given dimension
            dim_edges = self._construct_edges(js, ks, values, d)
            self.edges.append(dim_edges)

        # now handle the last 'dimension' i.e. the values in the final layer
        self.last_values = values[:, :, self.dim-1]
        return

    def _construct_edges(self, js, ks, values, d) -> set():
        """
        Returns a set of Edge objects for the given js, ks and values,
        corresponding to dimension d.

        An edge must have a start point, an end point and value.

        ks is of shape [js, n, d]
        and so you presumably loop over n and add the sets to each other
        without problem...
        """
        new_input_count = ks.shape[1]

        # for given dimension d, amalgamate the set of edges over the inputs
        for n in range(new_input_count):  # for each of the inpyts;
            this_input_indices = zip(js, ks[:, n, d], values[:, n, d])

            # to handle the last dimension, which has only edges to nothing,
            # I use a set of zeros instead of the locations of the 'next'
            # dimension's inputs.
            if d < self.dim - 1:
                next_input_indices = zip(js, ks[:, n, d+1])
            else:
                next_input_indices = zip(torch.zeros(js.shape),
                                         torch.zeros(ks.shape))

            # initialise the set for this dim
            this_input_set = set()
            indices = product(this_input_indices, next_input_indices)
            breakpoint()

            # loop over set of start and end points, and build the edge set
            for start_and_value, end in indices:
                start = (int(start_and_value[0]), int(start_and_value[1]))
                value = int(start_and_value[2])
                int_end = (int(end[0]), int(end[1]))
                edge = Edge(start, int_end, value)
                this_input_set.add(edge)

            if (n == 0):
                this_dim_set = this_input_set
            else:
                this_dim_set = self.add_edges(this_dim_set, this_input_set)

        return this_dim_set

    def add_edges(self, orig_set, new_set):
        """
        Takes an old set of edges and adds a new set, via the formulation
        wherein:

        - if the start point and end point of an edge are the same, then
          the values are added
        - if the start point and end point are not the same, then the sets
          get appended

        It is important to note that added_set gets unioned
        with a set including its elements but with different values. Because
        the edges are hashed according only to the start and end point, this
        should mean that the one from the added_set is kept.

        This may be a source of bugs in the future but it appears to work at
        the moment
        """
        # get the set of common edges, with values added
        added_set = set(map(lambda e1, e2: e1+e2, orig_set, new_set))

        # get the set of unique edges across both new and old
        unique_set = orig_set.union(new_set)

        # union on the unique set which pulls in the ones not already
        # in according to start and end point.
        # Use filter to clear out a None that appears for some reason
        final_set = set(filter(None, added_set.union(unique_set)))

        return final_set

    def get_edges(self):
        return self.edges


class SparseHaarEstimate:
    """
    A Sparse Haar estimate.

    The Sparse Haar estimate can be used to store the coefficients of an
    orthogonal series estimate for a given intensity function, for the
    purpose of classification.

    For an input, we have to evaluate the whole of the Haar basis, truncated
    up to a given j and k. The Haar basis in multiple dimensions consists of
    the successive Kronecker product of the individual coefficient matrices.
    The quickly-expanding scale of a naive approach means that we have to work
    using a sparse method.

    For an input in d dimensions, each of the dimensions of the input
    represents an entire matrix of coefficients in the Haar basis.
    However each of these values is just a 1 or -1, multiplied by 2^j.
    The storage of the Kronecker tensor can thus be done by storing
    the individual matrices and defining operations that match the equivalent
    operation on the Kronecker matrix. I.e. if we can match the operation
    of addition in 'Kronecker' space with operations on the constituent
    matrices (or some derivative object) then we can store the Kronecker
    representation sparsely.

    """
    def __init__(self,
                 index_dict,
                 dim,
                 inputs=torch.Tensor([])):
        self.lj = index_dict['lower_j']
        self.uj = index_dict['upper_j']
        self.lk = index_dict['lower_k']
        self.uk = index_dict['upper_k']

        self.dim = dim
        self.haar_data = torch.Tensor([])
        self.input_count = 0

        # js and ks initialisation
        self.js = torch.Tensor([])

        self.graph = Graph(dim)  # create a graph for handling the kronecker
        # representation

        if isinstance(inputs, torch.Tensor):
            self.inputs = inputs
        else:
            raise ValueError("inputs needs to be of type torch.Tensor.")

        return

    def add_inputs(self, x):
        """
        Adds inputs to the estimate

        This involves:
            - incrementing the input count so we know how many have been added
            - getting the ks and js for the new input
            - constructing the set of edges/graph information so we can store
              and evaluate
        """

        # self._update_js_and_ks(x)

        # increment the input count
        new_input_count = x.shape[0]
        self.input_count += new_input_count

        # get the js, ks and values for the new input
        js, ks, values = self._get_ks_js_values(x)

        # update the edges, with new js and ks
        self.graph.update_edges(js, ks, values)
        return

    def _get_ks_js_values(self, x):
        """
        Returns the set of js, ks and values for a given input.

        Acquisition of the set of js here requires processing in the sense that
        we want the
        """
        js = self._construct_js(x)
        ks, values = self._construct_ks_and_values(x, js)
        return js, ks, values

    def _construct_js(self, x):
        """
        For a given x, calculates the sequence of js such that:
            - for j below_min_j, all inputs have a 1 at k = 0
            - for j above max_j, the relevant k solving the inequality
                is greater than the maximum k
        """

        # acquire relevant js:
        # calculate the smallest j such that the value is 1, for each input
        min_js = torch.floor(-torch.log2(x) - 1)  # [n, d]

        # calculate  the largest possible j given the k budget
        """
        If we have max_k = 20, theis will calculate the largest relevant
        j in the diagram - so this is the one such that the next non-zero value
        no longer has a k less than max_k.

        """
        max_js = torch.floor(torch.log2((1 + self.uk)/x))  # [n,d]

        net_min_j = int(torch.min(min_js))  # the minimum max_j across all xs
        net_max_j = int(torch.max(max_js))  # the maximum max_j across all xs

        # so js is the sequence from min_j to max_j
        js = torch.linspace(net_min_j,
                            net_max_j,
                            (net_max_j - net_min_j + 1))

        return js

    def _construct_ks_and_values(self, x, js):
        """
        Returns the ks and corresponding values for the set of js built from
        the input x. The js set is the minimum set of js required to capture
        the relevant information (i.e. from min_j to max_j)
        """
        if (js.shape[0] != 0):  # save the js we have
            self.js = js
        else:
            raise ValueError("The set of j-indices is empty")

        js2 = torch.pow(2, js+1).repeat([self.dim, x.shape[0], 1])
        js2 = torch.einsum('ijk -> jki', js2)  # so [n, relevant_js, d]
        xview = torch.einsum('ijk->jik', x.repeat(js.shape[0], 1, 1))

        ks = torch.einsum('ijk->jik', torch.ceil(xview * js2)/2)

        values = torch.where(ks % 1 != 0,
                             torch.ones(ks.shape),
                             -torch.ones(ks.shape))
        ks = torch.floor(ks)

        return ks, values  # [relevant_js, n, d]

    def get_kronecker(self):
        """
        Constructs the Kronecker from the graph associated with the estimate.
        This will allow me to test whether my idea for the method is correct!
        """
        base_height = (self.uj - self.lj)
        base_width = (self.uk - self.lk)

        kronecker_height = base_height ** self.dim
        kronecker_width = base_width ** self.dim

        kronecker = torch.zeros([kronecker_height,
                                 kronecker_width])

        edges = self.graph.get_edges()

        kronecker = self._get_kronecker_recurse(self,
                                                base_height,
                                                base_width,
                                                0)

        for j in range(base_height):
            for k in range(base_width):
                # get the value at that location and plop it in
                break
        return

    def _get_kronecker_recurse(self, base_height, base_width, level):
        """
        This function recursively builds the Kronecker.

        Because at each level of the kronecker we build a matrix using the
        next level of data we should be able to construct some recursive
        schema that builds up the resulting kronecker matrix.
        """
        for j in range(base_height):
            for k in range(base_height):
                if level == self.dim:  # if we're at the finest resolution
                    # get the last edges
                    edges = self.graph.get_edges()[level]

        return

    def update_ks_and_values(self, x, js_deltas):  # setter
        """
        Appends the ks and values for the passed input
        to the set of preexisting ks and values, whilst padding
        to take account of the new smallest j.
        """

        ks, values = self._construct_ks_and_values(x)
        # breakpoint()
        # having got the ks and values for the input
        # we need to also take into account the extension of the js...

        old_ks = self.get_ks()
        old_values = self.get_values()

        # calculate size difference of new input
        # expand the ks and values to account for the new input
        ks_pad = torch.zeros(new_js, self.input_count, self.dim)
        values_pad = torch.ones(new_js, self.input_count, self.dim)

        ks = torch.stack((old_ks, ks_pad), dim=0)
        values = torch.stack((old_values, values_pad), dim=0)

        # update the class state
        self.set_values(values)
        self.set_ks(ks)

        return

    def _update_edges(self):
        """
        Pulls the set of js and ks from the class and builds the edge set for
        the graph formulation
        """
        raise NotImplementedError("Not yet Implemented: _update_edges")
        # js = self.get_js()
        # ks = self.get_ks()
        # return

    def print_edges(self):
        edge_sets = self.graph.get_edges()
        for edge_set in edge_sets:
            for edge in edge_set:
                print(edge.get_start_point(),
                      edge.get_end_point(),
                      edge.get_value())
        return

    def get_js(self):
        return self.js

    def get_ks(self):
        return self.ks

    def get_values(self):
        return self.values

    # def set_js(self):
    #     return self.js

    # def set_ks(self):
    #     return self.ks

    # def set_values(self):
    #     return self.values




if __name__ == "__main__":
    edge_1 = Edge((torch.tensor(1.), torch.tensor(2.)),
                  (torch.tensor(2.), torch.tensor(3.)),
                  torch.tensor(3.))
    edge_2 = Edge((torch.tensor(1.), torch.tensor(2.)),
                  (torch.tensor(2.), torch.tensor(3.)),
                  torch.tensor(4.))
    edge_3 = Edge((torch.tensor(3.), torch.tensor(2.)),
                  (torch.tensor(4.), torch.tensor(3.)),
                  torch.tensor(3.))
    edge_4 = Edge((torch.tensor(4.), torch.tensor(2.)),
                  (torch.tensor(5.), torch.tensor(3.)),
                  torch.tensor(3.))

    my_set_1 = set()
    my_set_2 = set()

    my_set_1.add(edge_1)
    my_set_1.add(edge_3)
    my_set_2.add(edge_2)
    my_set_2.add(edge_4)

    net_set = set(map(lambda e1, e2: e1+e2, my_set_1, my_set_2))
    unique_set = my_set_1.union(my_set_2)
    final_set = filter(None, net_set.union(unique_set))

    # for edge in final_set:
    #     print(edge.start_point, edge.end_point, edge.value)
    # breakpoint()
    index_dict = {
        'lower_j': -20,
        'upper_j': 20,
        'lower_k': -50,
        'upper_k': 50
    }
    dim = 2
    x = torch.Tensor([[0.23, 0.21],
                    #   [0.23, 0.22],
                    #   [0.23, 0.23],
                    #   [0.23, 1.24],
                    #   [0.23, 2.24],
                    #   [0.23, 3.24],
                    #   [0.23, 2.59],
                    #   [0.23, 3.42],
                    #   [0.23, 4.89],
                    #   [0.23, 2.14],
                    #   [0.23, 8.94],
                    #   [0.23, 5.14],
                    #   [0.23, 6.88],
                      [0.23, 0.24],
                      [0.23, 0.25]])


    estimate = SparseHaarEstimate(index_dict, dim)
    # breakpoint()
    estimate.add_inputs(x)
    # print(len(estimate.graph.edges))
    # breakpoint()
    # estimate.print_edges()


    """
    THINGS WE NEED TO TEST:
        DOES THIS PROPERLY HANDLE THE ADDITION OF JS TO INCLUDE THAT THE
        MINIMUM J WE HAVE MUST BE THE LOWEST SUCH THAT ALL INPUTS EVER SEEN
        ARE 0?
    """

