import collections.abc as collections
import torch


ALPHAS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvxyz"\
    + "!£$%^&*()"


class Schema(dict):
    def __init__(self, init=None):
        if init is not None:
            super().__init__(init)
        else:
            super().__init__()

    def __add__(self, other):
        """
        Implementing the add function allows us to specify that the result
        of adding a new input is to add the sets for the correpsonding
        js and ks with that input
        """
        this_dict = self.copy()
        if not isinstance(other, dict):
            raise TypeError("The second operand should be a subclass of a" +
                            "dictionary")

        # amalgamate common keys
        for key in self.keys() & other.keys():
            if isinstance(this_dict[key], set):
                # this_dict[key] = this_dict[key].extend(other[key])
                intermediate_set = this_dict[key]
                intermediate_set.update(other[key])
                this_dict[key] = intermediate_set
            else:
                raise TypeError("The elements in the Schemata should be lists")

        # append new keys and their values
        for key in other.keys() - this_dict.keys():
            # the set of keys that are in the other keys and not in self
            this_dict[key] = other[key]

        return this_dict

    def __getitem__(self, key):
        if isinstance(key, list)\
            or isinstance(key, tuple)\
            or isinstance(key, set)\
                or isinstance(key, collections.KeysView):
            print("ABOUT TO GET THE VALUES FOR A SET OF KEYS FROM THE DICT!")
            return {frozenset(self[k]) for k in key}

        else:
            return dict.get(self, key)

    def copy(self):
        dictionary = super().copy()
        return Schema(dictionary)

    def shift_inputs(self, n):
        """
        Shifts all the inputs in this Schema up by n.


        When adding a Schema to another Schema, each Schema has on it
        the input lists for itself. However these must be increased so that
        later inputs are treated as later inputs and earlier inputs are treated
        as earlier inputs.
        """
        for inputs_set in self.values():
            # for each of the input sets in self.values
            number = inputs_set.pop()
            number += n
            inputs_set.add(number)
        """ WARNING! I DON'T SEE WHY THIS WORKS PROPERLY!"""
        return


class SchemaConstructor:
    def __init__(self, dim, max_j, max_k):
        self.dim = dim
        self.max_j = max_j
        self.max_k = max_k

        self.j_alphas = ALPHAS[:2*max_j + 1]
        self.k_alphas = ALPHAS[:2*max_k + 1]
        return

    def get_schema(self, js, dim_ks, dim_vs) -> Schema:
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
                schema[word].add(i)  # this should handle the input_count
                # i guess when updating a schema...

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


class IndexConstructor:
    """
    The IndexConstructor is an object that holds a collection of methods
    together. Methods on this class produce the js, ks and vs for a given input
    or set of inputs - I separated this class from the SparseHaarEstimate to
    simplify the structure of that class.
    """
    def __init__(self, dim, max_j, max_k):
        self.max_j = max_j
        self.max_k = max_k
        self.dim = dim

    def get_jkvs(self, x):
        # I wil leave it with this structure so as to handle possible
        # processing
        ks, js, values = self._get_ks_js_values(x)
        return js, ks, values

    def _get_ks_js_values(self, x):
        """
        Returns the set of js, ks and values for a given input.

        """
        js = self._construct_js(x)
        ks, values = self._construct_ks_and_values(x, js)
        return ks, js, values

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
        max_js = torch.floor(torch.log2((1 + self.max_k)/x))  # [n,d]

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

        final_ks = torch.where(ks % 1 != 0,
                               torch.floor(ks),
                               ks - 1)

        return final_ks, values  # [relevant_js, n, d]


class Schemata:
    def __init__(self, dim, *args):
        self.dim = dim
        self.schemata = []
        if len(args) != 0:
            for schema in args:
                self.schemata.append(schema)
        else:
            for d in range(dim):
                self.schemata.append(Schema())
        self.input_count = 0

    def combine(self, new_schemata):
        """
        For a new schema, will calculate the locations at which common indices
        exist - and so the values must be added.

        Step 1) get the intersection in each dimension of the new schemata
        with the old one. - this limits us to points that will sum between
        both the new and old ones. It also limits us specifically to a smaller
        loop size.

        Step 2) For each key in the first dimension, calculate the intersection
                between the key-relevant inputs and the inputs for each key
                in the second dimension
        """
        base_schemata = self.schemata.copy()
        # youi have to do something in here with relevant keys...
        for d, s in enumerate(base_schemata):
            relevant_keys = new_schemata.keys() & s.keys()
            # for key in relevant_keys:

        d = 0
        while (not self._check_input_set_union(base_schemata))\
                and d < self.dim-1:
            base_schemata = self._forward_pass_intersect(base_schemata, d)
            d += 1
        # now base_schemata contains the keys and inputs that are connected to
        # the new input

    def _forward_pass_intersect(schemata, d):
        for d_key in schemata[d]:
            for next_d_key in schemata[d+1]:
                intersected_inputs = schemata[d+1][next_d_key]\
                    .intersection(schemata[d][d_key])
                if len(intersected_inputs) == 0:
                    schemata[d+1].pop(next_d_key)
                else:
                    schemata[d+1][next_d_key] = intersected_inputs
        return schemata

    def _check_input_set_union(self, schemata):
        """
        Tests whether the inputs that are in each dimension of a schemata are
        the same. This will be true when we have exhausted the  operation that
        intersections the sets in the first layer to the second layer, etc.
        """
        # first, get the union of the input sets across all the dimensions
        unions_set = set()

        # Now get the union in each dimension, and append it to the set.
        for s in schemata:  # one for each dimension
            # take each key in this dimension and get the intersection with
            # each key in the next dimension. How do we cut off the
            d_sets = s.values()  # sets for this dimension
            d_union = set()
            for this_set in d_sets():
                d_union.update(this_set)
            unions_set.update(frozenset(d_union))

        # different 'paths' to having the same location may exist via different
        # inputs  -  but the union in one idmension will contain all the paths
        # in that dimension, as will the second, etc. This means that if we
        # have completed the intersection process, the union of the unions will
        # contain exactly one element, because the set will be the same across
        # all dimensions.
        if len(unions_set) == 1:
            return True
        else:
            return False

    def __getitem__(self, key):
        return self.schemata[key]

    def __setitem__(self, d, item):
        self.schemata[d] = item
        return

    def __repr__(self):
        return self.schemata.__repr__()

    def __add__(self, other):
        """
        For each dimension, amalgamates teh incoming Schema of that dimension
        with the current Schema of this dimension. It is important that the
        inputs are all shifted up by the input count.
        """
        if not isinstance(other, Schemata):
            raise TypeError("the add operation is only permitted between" +
                            " Schemata and Schemata")
        new_schemata = Schemata(self.dim)

        for d in range(self.dim):
            other[d].shift_inputs(self.input_count)
            new_schemata[d] = self[d] + other[d]

        self.input_count += other.get_input_count()
        return new_schemata

    def __eq__(self, other):
        if not isinstance(other, Schemata):
            raise TypeError("the add operation is only permitted between" +
                            " Schemata and Schemata")
        for d in range(self.dim):
            if self[d] != other[d]:
                return False
        return True

    def get_input_count(self):
        return self.input_count

    def set_input_count(self, n):
        self.input_count = n


class SchemataConstructor:
    def __init__(self, dim, max_j, max_k):
        self.dim = dim
        self.index_constructor = IndexConstructor(dim, max_j, max_k)
        self.schema_constructor = SchemaConstructor(dim, max_j, max_k)
        return

    def get_schemata(self, x) -> Schemata:
        """
        Returns a Schemata object, with Schemata corresponding to the inputs
        passed in. It should naturally handle long inputs.
        """
        js, ks, vs = self.index_constructor.get_jkvs(x)
        schemata = Schemata(self.dim)
        input_count = x.shape[0]
        for d in range(self.dim):
            dim_ks = ks[:, :, d]
            dim_vs = vs[:, :, d]
            schema = self.schema_constructor.get_schema(js, dim_ks, dim_vs)
            schemata[d] = schema
        schemata.set_input_count(input_count)

        return schemata
