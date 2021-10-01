# sparse_haar_3.py
import torch
import collections
from framework.utils import print_dict
from schemata import Schema, Schemata, SchemaConstructor, IndexConstructor

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
                 max_k):

        self.dim = dim

        # alphabets
        self.j_alphas = ALPHAS[:2*max_j + 1]
        self.k_alphas = ALPHAS[:2*max_k + 1]
        breakpoint()
        # maximum indices
        self.max_j = max_j
        self.max_k = max_k

        self.input_count = 0

        self.word_sets = []  # this will expand in the N dimension

        # self.schemata = []
        # for i in range(self.dim):
        #    self.schemata.append(dict())
        #    # initialise the sets
        self.schemata = Schemata(self.dim)

        self.commons = []

        self.index_constructor = IndexConstructor(dim, max_j, max_k)

        return

    def get_input_indices(self):
        return self.word_sets

    def check_input(self, x):
        """
        Takes an input and checks whether it has any common locations with the
        other inputs so far.

        Steps:
            - take common keys between the new input and the extant inputs
            - union the relevant sets with in each dimension
            - intersection across dimensions
        the resulting set contains the index of each input that matches with
        the new input in some location

        Returns:
            - set of inputs that have coincidence
            - the locations at which they coincide

        """
        # get the set of keys for the new input
        js, ks, vs = self.index_constructor.get_jkvs(x)  # ??

        for d in range(self.dim):
            # select appropriate indices for this dimension
            new_schema = self._get_schema(js, ks[:, :, d], vs[:, :, d])
            old_schema = self.schemata[d]

            # get the common keys between the new input and the extant inputs
            common_keys = new_schema.keys() & old_schema.keys()
            redundant_set = set(range(self.input_count))
            common_inputs = old_schema[common_keys]

            # remove the all-input keys, as they are redundant for the next
            # calculation
            joint_set = set()
            for i in common_inputs:
                joint_set = joint_set.union(i)

            breakpoint()

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

        js, ks, vs = self.index_constructor.get_jkvs(x)  # ??
        breakpoint()
        # initialise the set to calculate the common inputs
        # now build A∪B and A∩B
        for d in range(self.dim):
            dim_ks = ks[:, :, d]
            dim_vs = vs[:, :, d]
            old_schema = self.schemata[d]
            new_schema = self._get_schema(js, dim_ks, dim_vs)
            self.schemata[d] = self._update_schema(old_schema,
                                                   new_schema)
            print(new_schema)
            common_values = set()
            common_keys = new_schema.keys() & old_schema.keys()
            for key in common_keys:
                common_values = common_values.union(old_schema[key])
            print("common set", common_values, "dimension", d)
        # self.schemata now contains AUB where A is the original schema, and B
        # is the schema from the new one
        # we also have to get the

        # how do we build A∩B?
        """
        # step 1
        for i in range(x.shape[0]):
            ks, js, values = self.index_constructor.get_jkvs(x[i, :])
            this_word_sets = []
            for d in range(self.dim):
                dim_ks = ks[:, :, d]
                dim_vs = values[:, :, d]

                words = self._get_words(js, dim_ks, dim_vs)
                breakpoint()
                print("i and words are:", i, words)
                this_word_sets.append(words)
            self.word_sets.append(this_word_sets)
        """
        # step 2
        self.input_count += x.shape[0]

        return

    def _update_schema(self, old_schema, new_schema) -> Schema:
        """
        Accepts an Schema, and js and ks from a new input.

        Returns a new Schema, with the keys from both new and old, and
        updates the sets containing relevant inputs.

        The final Schema contains keys as index locations in the form of words,
        and values as sets of relevant inputs
        """
        # temp_net_min_j = self.net_min_j  # this will only differ when the new
        # new_net_min_j = int(min(js).item())

        # input comes in, for all dimensions - once processing beyond the
        # zero-th, to the d-th dimension, this will be zero. However we are
        # fixing the dth dimension IF this is not zero...

        # get the new schema for the inputs

        temp_old_schema = old_schema.copy()
        # breakpoint()
        # if this is a new input, then use just the new_schema
        if len(old_schema) == 0:
            print("This time, just pulling in the new schema!")
            final_schema = new_schema.copy()
        else:
            final_schema = temp_old_schema + new_schema

        return final_schema

    def _get_schema(self, js, dim_ks, dim_vs) -> Schema:
        """
        Returns, for dimension d, the Schema containing:
            - a jk word as the key for the given set of relevant inputs
            - a list with the indices of the input relevant for the given js
              and ks

        :param js:
            the js that are relevant to be stored (i.e. not redundantly all 1s
            for k = 0
        :param dim_ks:
            the ks for a given dimension for a given set of inputs - built in
            self.add_inputs
        :param dim_vs:
            the valuess for a given dimension for a given set of inputs - build
            in self.add_inputs

        """
        words = self._get_words(js, dim_ks, dim_vs)

        # construct an empty Schema with these words
        schema = self._get_blank_schema(words)

        # now update the set of inputs corresponding to each key
        for i in range(dim_ks.shape[1]):
            this_input_words = self._get_words(js,
                                               dim_ks[:, i].unsqueeze(1),
                                               dim_vs[:, i].unsqueeze(1))
            # you're not checking if it's in one or the other!
            # get the corresponding keys for this input!
            for word in this_input_words:
                schema[word].add(i + self.input_count)
        # breakpoint()
        return schema

    def _get_blank_schema(self, keys, arg=None):
        """
        From a  set of keys, i.e. per-dim jk sybmol pairs,
        constructs a Schema and returns it.

        :param keys:
            an iterable containing relevant keys for building the Schema
        :param arg:
            allows one to fill the blank Schema's sets with some specific
            object
        """
        dictionary = Schema().fromkeys(keys)
        if arg is None:
            for i in dictionary.keys():
                dictionary[i] = set()
        else:
            for i in dictionary.keys():
                dictionary[i] = set(arg)
        return dictionary

    def _get_words(self, js, dim_ks, dim_vs) -> set:
        """
        DOES NOT CURRENTLY INCLUDE THE VALUES
        Produces the set of words for a given input whose js, ks and vs have
        been passed to the function.

        The result is a set containing a string for index in this dimension's
        matrix. The string is of the form jkv, where:
            - j is from the j_alphabet
            - k is from the k-alphabet
            - v is from the values
        """

        # extend the set of js along the length of the input
        extended_js = js.repeat_interleave(dim_ks.shape[1]).unsqueeze(1)
        extended_ks = dim_ks.reshape([len(js) * dim_ks.shape[1], 1])
        extended_vs = dim_vs.reshape([len(js) * dim_vs.shape[1], 1])
        concat_jsksvs = torch.cat([extended_js,
                                   extended_ks,
                                   extended_vs], dim=1)

        # prepare the set of words
        this_set = set()

        for word in concat_jsksvs[:]:
            j_index = int(word[0] + self.max_j)
            k_index = int(word[1] + self.max_k)
            if k_index < 2*self.max_k+1:
                try:
                    j = self.j_alphas[j_index]
                    k = self.k_alphas[k_index]
                except IndexError:
                    print("Index error!")
                    breakpoint()
                #  v = str(int(0 * (word[2] > 0) + 1 * (word[2] < 0)))
                wordtext = j+k  # +v
                this_set.add(wordtext)

        return this_set


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
    x1 = torch.Tensor([[0.23, 0.23]])
    x2 = torch.Tensor([[0.23, 0.26]])
    #  x3 = torch.Tensor([[0.23, 0.23]])
    x = torch.Tensor([[0.23, 0.23],
                      [0.23, 0.26]])
    dim = 2
    max_j = 20
    max_k = 20
    estimate = SparseHaarEstimate(dim, max_j, max_k)
    estimate_2 = SparseHaarEstimate(dim, max_j, max_k)

    print("About to add the inputs")
    estimate.add_inputs(x1)
    estimate.add_inputs(x2)
    estimate_2.add_inputs(x)
    # estimate.check_input(x3)
    #estimate.add_inputs(y)
    final_schemata = estimate.schemata
    final_schemata2 = estimate_2.schemata
    for diction in final_schemata:
        print("Next dict:")
        print_dict(diction)

    for diction in final_schemata2:
        print("Next dict:")
        print_dict(diction)
    breakpoint()
