# sparse_haar_2.py
import torch
# import framework.haar_basis_functions
# from framework.haar_basis_functions import plot_matrix


def print_dict(dictionary):
    my_dict = dictionary.copy()
    keys_iter = my_dict.keys()
    for key in keys_iter:
        print(key, my_dict[key])
    return


def print_dict_diffs(dictionary1, dictionary2):
    print("Keys that differ between the two dicts:")
    for key in dictionary1.keys() ^ dictionary2.keys():
        print(key)

    print("Values that differ between the two dicts, at same key:")
    print("dictionary 1", end='')
    print("dictionary 2")
    for key in dictionary1.keys() & dictionary2.keys():
        if dictionary1[key] != dictionary2[key]:
            print("key: ", key)
            print(dictionary1[key], end='')
            print(dictionary2[key])


def get_schema(start_dict):
    if len(start_dict) > 0:
        new_schema = Schema.fromkeys(start_dict.keys())
        new_schema.update(start_dict)
        return new_schema
    else:
        return Schema()


class Schema(dict):
    def __add__(self, other):
        """
        Implementing the add function allows us to specify that the result
        of adding a new input is to extend the lists of the correpsonding
        js and ks with that input
        """
        this_dict = self.copy()
        if not isinstance(other, dict):
            raise TypeError("The second operand should be a subclass of a\
                            dictionary")

        # amalgamate common keys
        for key in self.keys() & other.keys():
            if isinstance(this_dict[key], list):
                # this_dict[key] = this_dict[key].extend(other[key])
                intermediate_list = this_dict[key]
                intermediate_list.extend(other[key])
                this_dict[key] = intermediate_list
            else:
                raise TypeError("The elements in the Schemata should be lists")

        # add new keys and their values
        for key in other.keys() - this_dict.keys():
            # the set of keys that are in the other keys and not in self
            this_dict[key] = other[key]
        return this_dict

    def copy(self):
        dictionary = super().copy()
        return get_schema(dictionary)


class SparseHaarEstimate:
    """
    This is an implementation of the new method for SparseHaarEstimates.

    The 'new method' involves storage of the js, ks and _relevant inputs_ on
    the class. This should then allow us to build the estimate coefficients
    with ease.
    """
    def __init__(self,
                 dim,
                 max_j,
                 max_k):
        self.dim = dim
        self.max_j = max_j
        self.max_k = max_k

        # set the initial minimum and maximum relevant j
        self.net_min_j = 0
        self.net_max_j = 0

        # without a loop, don't know how to get it not to just ref. the same
        # dict every time
        self.dicts = []
        for i in range(dim):
            self.dicts.append(Schema())

        # to the same dict
        self.input_count = 0
        print("at init, the dicts are:", self.dicts)
        return

    def get_dicts(self):
        return self.dicts

    def add_inputs(self, x):
        """
        Adds inputs to the estimate storage.

        This does the following:
            - for each dimension, adds the input indices to the list of input
              indices corresponding to each relevant j and k pair
            - increments the count of inputs
        """
        # for each dimension, build the corresponding dictionary
        if x.shape[1] != self.dim:
            raise ValueError("The input should be of\
                             dimension {}".format(self.dim))

        js, ks, _ = self._get_ks_js_values(x)

        if self.input_count == 0:
            self.net_min_j = int(min(js).item())

        # loop over dimensions, constructing the corresponding dictionary
        for d in range(self.dim):
            self.dicts[d] = self._update_dictionary(self.dicts[d],
                                                    js,
                                                    ks[:, :, d])

        # increment the input count
        self.input_count += x.shape[0]  # the N dimension

        # and update the min/max js
        self.net_min_j = min(int(min(js).item()), self.net_min_j)

        return

    def _update_dictionary(self, old_dictionary, js, dim_ks):
        """
        Updates the dictionary passed in, according to the dimension and the
        input x.

        In here is code that handles the fact that when adding later inputs,
        there my be now relevant js that have to be accounted for.
        This is dealt with by checking whether the minimum relevant j is
        changed by the new input. If it is, then
        """
        # temp_net_min_j = self.net_min_j  # this will only differ when the new
        # new_net_min_j = int(min(js).item())

        # input comes in, for all dimensions - once processing beyond the
        # zero-th, to the d-th dimension, this will be zero. However we are
        # fixing the dth dimension IF this is not zero...

        # get the new dictionary for the inputs
        new_dictionary = self._get_dictionary(js, dim_ks)

        temp_old_dictionary = old_dictionary.copy()

        # if this is a new input, then use just the new_dictionary
        if len(old_dictionary) == 0:
            print("This time, just pulling in the new dictionary!")
            final_dictionary = new_dictionary.copy()
        else:
            final_dictionary = temp_old_dictionary + new_dictionary

        return final_dictionary

    def _get_dictionary(self, js, dim_ks):
        """
        Returns, for dimension d, the dictionary containing:
            - a j, k pair tuple as the key for the given set of relevant inputs
            - a list with the indices of the inputs relevant for the given js
              and ks

        This needs to be done in a way that allows new inputs to be added
        to the dictionary without completely destroying the old information.
        The fromkeys() method on the dict essentially builds it anew - this is
        not what we want!
        """
        # get the keys for the dictionary
        keys = self._get_keys(js, dim_ks)
        # construct an 'empty' dictionary with those keys
        dictionary = self._get_blank_dictionary(keys)

        # breakpoint()
        # now update the set of inputs corresponding to each key
        # given the dictionary, this can be done with any input and keys

        # loop through inputs:
        for i in range(dim_ks.shape[1]):
            # build the keys that we will use to index the input lists
            input_concat_jsks = torch.hstack([js.unsqueeze(1),
                                              dim_ks[:, i]
                                              .unsqueeze(1)]).tolist()
            #  breakpoint()
            input_keys = [tuple(i) for i in input_concat_jsks]

            # now we can set the input lists for
            for key in input_keys:
                # print(key)
                dictionary[key].append(i + self.input_count)
        # breakpoint()
        return dictionary

    def _get_blank_dictionary(self, keys, arg=None):
        dictionary = Schema().fromkeys(keys[:])
        if arg is None:
            for i in dictionary.keys():
                dictionary[i] = list()
        else:
            for i in dictionary.keys():
                dictionary[i] = list(arg)
        return dictionary

    def _get_keys(self, js, dim_ks):
        """
        returns the keys corresponding to a given input x
        """
        extended_js = js.repeat_interleave(dim_ks.shape[1]).unsqueeze(1)
        extended_ks = dim_ks.reshape([len(js) * dim_ks.shape[1], 1])
        concat_jsks = torch.cat([extended_js, extended_ks], dim=1).tolist()
        return [tuple(i) for i in concat_jsks]

    def _get_ks_js_values(self, x):
        """
        Returns the set of js, ks and values for a given input.

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
        ks = torch.floor(ks)
        return ks, values  # [relevant_js, n, d]


if __name__ == "__main__":
    x = torch.Tensor([[0.23, 0.21],
                      [0.23, 0.22],
                      [0.23, 0.23],
                      [0.23, 1.24],
                      [0.23, 2.24],
                      [0.23, 3.24],
                      [0.23, 2.59],
                      [0.23, 3.42],
                      [0.23, 4.89],
                      [0.23, 2.14],
                      [0.23, 8.94],
                      [0.23, 5.14],
                      [0.23, 6.88],
                      [0.23, 0.24],
                      [0.23, 0.25]])

    y1 = torch.Tensor([[0.23, 0.21]])
    y2 = torch.Tensor([[0.23, 0.24]])
    y3 = torch.Tensor([[0.23, 0.25]])

    y = torch.Tensor([[0.23, 0.21],
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
                      [0.23, 0.25]])
    x = y

    # parameters for testing
    run_estimate = True
    run_estimate_2 = True
    dim = 2
    if dim == 1:
        breakpoint()
        x = x[:, 1]
        y = y[:, 1]

    max_j = 20
    max_k = 20
    if run_estimate:
        estimate = SparseHaarEstimate(dim, max_j, max_k)
        estimate.add_inputs(x)
        print("Printing estimate dict 1")
        print_dict(estimate.dicts[0])

    if run_estimate_2:
        estimate_2 = SparseHaarEstimate(dim, max_j, max_k)
        for index, i in enumerate(y):
            print("We're at index ", index)
            print(i.unsqueeze(0))
            estimate_2.add_inputs(i.unsqueeze(0))
        print("Printing estmate_2 dict 1")
        print_dict(estimate_2.dicts[0])
        # breakpoint()
    """
    for key in estimate_2.dicts[1].keys():
        if estimate_2.dicts[1][key] != estimate.dicts[1][key]:
            print("DIFF!", end='')
            print("key: ", key, end='')
            print("estimate_2 value: ", estimate_2.dicts[1][key])
            print("estimate value: ", estimate.dicts[1][key])
    """

    # print(estimate.dicts[0].keys() - estimate_2.dicts[0].keys())
    if run_estimate and run_estimate_2:
        success = (estimate_2.dicts[0] == estimate.dicts[0])
        if not success:
            print_dict_diffs(estimate.dicts[0], estimate_2.dicts[0])
            breakpoint()
        else:
            print("Success! The two estimates are the same.")
