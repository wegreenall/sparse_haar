import torch
from schemata import SchemataConstructor, Schema
from framework.utils import print_dict

inputs = torch.Tensor([[0.23, 0.23],
                       [0.23, 0.26]])
#                       [0.24, 0.26]])
#                       [4.23, 3.26],
#                       [6.23, 5.26],
#                       [0.23, 3.26],
#                       [4.23, 9.26]])

new_input = torch.Tensor([[0.24, 0.26]])

# parameters
dim = 2
max_j = 20
max_k = 20
schemata_constructor = SchemataConstructor(dim, max_j, max_k)
schemata_list = []

# base schemata
for this_input in inputs:
    schemata = schemata_constructor.get_schemata(this_input.unsqueeze(0))
    print(schemata)
    schemata_list.append(schemata)

new_schemata = schemata_constructor.get_schemata(new_input)

sc0 = schemata_list[0]
for sc in schemata_list[1:]:
    print("About to add a new schemata")
    print(sc0.get_input_count())
    sc0 = sc0 + sc
    print("new schemata added")
    print(sc0.get_input_count())

# build some methods to print the dicts for thinking about.

# get the new_schemata
print("Printing New Schemata:")
print("first dimension")
print_dict(new_schemata[0])
print("second dimension")
print_dict(new_schemata[1])

# get the base schemata
print("Printing Base Schemata:")
print("first dimension of sc0")
print_dict(sc0[0])
print("second dimension of sc0")
print_dict(sc0[1])

# get the final schemata
print("sc0 before adding something to it to get the next one")
print("Printing Final Schemata:")
print("first dimension")
print_dict(sc0[0])
print("second dimension")
print_dict(sc0[1])

sc_final = sc0 + new_schemata
print("sc0 after adding something to it to get the next one")
print("first dimension")
print_dict(sc0[0])
print("second dimension")
print_dict(sc0[1])

# get the base schemata but with common keys
print("Printing Common Keys Schemata:")
sc_common = sc0.intersection(new_schemata)
print("first dimension")
print_dict(sc_common[0])
print("second dimension")
print_dict(sc_common[1])

#    breakpoint()
schemata_2 = schemata_constructor.get_schemata(inputs[1].unsqueeze(0))

# now that I have the common behaviour...

breakpoint()


# combine testbed
def get_layer_list(schemata, d=0):
    # some kind of check if it is the last layer...
    js = []
    ks = []
    inputs = []
    if d == (schemata.dim - 1):
        for key in schemata[d].keys():
            #inputs.append(update(schemata[d][key]))
            js.append(key[0])
            ks.append(key[1])
            inputs.append(schemata[d][key])
    else:
        next_layer_js,\
            next_layer_ks,\
            next_layer_inputs = get_layer_list(schemata, d+1)
        print("Check next layer inputs")
        for key in schemata[d].keys():
            this_input = schemata[d][key]
            # get the js from the next layers
            this_j = key[0]
            this_k = key[1]
            these_js = [this_j + next_j for next_j in next_layer_js]
            these_ks = [this_k + next_k for next_k in next_layer_ks]
            these_inputs = [this_input.intersection(next_input) for
                            next_input in
                            next_layer_inputs]
            js.extend(these_js)
            ks.extend(these_ks)
            inputs.extend(these_inputs)
    return js, ks, inputs


final_js, final_ks, final_inputs = get_layer_list(sc_common)
breakpoint()
for final_input in final_inputs:
    final_input.add(2)

keys = zip(final_js, final_ks)
final_dict = dict()
for i, key in enumerate(keys):
    final_dict[key] = final_inputs[i]

print("final_dict", final_dict)
breakpoint()
