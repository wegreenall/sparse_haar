# sparse_haar/tests/sparse_haar_test.py
import unittest
import torch
# from sparse_haar.schemata import Schema, Schemata, SchemataConstructor,\
#                SchemaConstructor, IndexConstructor

from sparse_haar.schemata import SchemataConstructor,\
    IndexConstructor, SchemaConstructor, Schema


class TestIndexConstructor(unittest.TestCase):
    def setUp(self):
        # parameters
        self.dim = 2
        self.max_j = 20
        self.max_k = 20
        self.x1 = torch.Tensor([[0.23, 0.23]])

        # self.x2 = torch.Tensor([[0.23, 0.23],
        #                        [0.23, 0.23]])
        self.index_constructor = IndexConstructor(self.dim,
                                                  self.max_j,
                                                  self.max_k)

        self.js, self.ks, self.vs = self.index_constructor.get_jkvs(self.x1)

    def test_results(self):
        self.assertIsInstance(self.js, list)
        self.assertIsInstance(self.ks, list)
        self.assertIsInstance(self.vs, list)

    def test_jsks_length(self):
        self.assertTrue(len(self.js) == self.dim)
        self.assertTrue(len(self.ks) == self.dim)

    def test_js_correct(self):
        for d in range(self.dim):
            self.assertTrue((self.js[d] == torch.Tensor([-20.,
                                                         -19.,
                                                         -18.,
                                                         -17.,
                                                         -16.,
                                                         -15.,
                                                         -14.,
                                                         -13.,
                                                         -12.,
                                                         -11.,
                                                         -10.,
                                                         -9.,
                                                         -8.,
                                                         -7.,
                                                         -6.,
                                                         -5.,
                                                         -4.,
                                                         -3.,
                                                         -2.,
                                                         -1.,
                                                         0.,
                                                         1.,
                                                         2.,
                                                         3., 4., 5., 6.],
                                                        )).all())

    def test_ks_correct(self):
        js = self.index_constructor._construct_js(self.x1)
        ks, vs = self.index_constructor._construct_ks_and_values(self.x1, js)
        for d in range(self.dim):
            self.assertTrue((ks[d] == torch.Tensor([0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    0.,
                                                    1., 3., 7., 14.])).all())
            self.assertTrue((vs[d] == torch.Tensor([1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    1.,
                                                    -1.,
                                                    -1.,
                                                    -1.,
                                                    1.,
                                                    -1.])).all())


class TestSchemaConstructor(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.max_j = 20
        self.max_k = 20
        self.x1 = torch.Tensor([[0.23, 0.23]])

        # self.x2 = torch.Tensor([[0.23, 0.23],
        #                        [0.23, 0.23]])
        self.index_constructor = IndexConstructor(self.dim,
                                                  self.max_j,
                                                  self.max_k)

        self.js, self.ks, self.vs = self.index_constructor.get_jkvs(self.x1)
        self.schema_constructor = SchemaConstructor(self.dim,
                                                    self.max_j,
                                                    self.max_k)

    def test_words(self):
        d = 0
        words = self.schema_constructor._get_words(self.js[d],
                                                   self.ks[d],
                                                   self.vs[d])
        self.assertEqual(words, set(["0K",
                                     "1K",
                                     "2K",
                                     "3K",
                                     "4K",
                                     "5K",
                                     "6K",
                                     "7K",
                                     "8K",
                                     "9K",
                                     "AK",
                                     "BK",
                                     "CK",
                                     "DK",
                                     "EK",
                                     "FK",
                                     "GK",
                                     "HK",
                                     "IK",
                                     "JK",
                                     "KK",
                                     "LK",
                                     "MK", "NL", "ON", "PR", "QY"]))

    def test_get_blank_schema(self):
        test_output = self.schema_constructor._get_blank_schema({"NL",
                                                                 "ON",
                                                                 "PR",
                                                                 "QY"})
        self.assertTrue(isinstance(test_output, Schema))
        self.assertEqual(test_output.keys(),  {"NL", "ON", "PR", "QY"})
        for v in test_output.values():
            self.assertEqual(v, set())

    def test_get_schema(self):
        """ TESTING GET SCHEMA """
        d = 0
        test_output = self.schema_constructor.get_schema(self.js[d],
                                                         self.ks[d],
                                                         self.vs[d])
        self.assertTrue(isinstance(test_output, Schema))
        self.assertEqual(test_output.keys(), {"0K",
                                              "1K",
                                              "2K",
                                              "3K",
                                              "4K",
                                              "5K",
                                              "6K",
                                              "7K",
                                              "8K",
                                              "9K",
                                              "AK",
                                              "BK",
                                              "CK",
                                              "DK",
                                              "EK",
                                              "FK",
                                              "GK",
                                              "HK",
                                              "IK",
                                              "JK",
                                              "KK",
                                              "LK",
                                              "MK", "NL", "ON", "PR", "QY"})
        for v in test_output.values():
            self.assertEqual(v, {0})


class TestSchema(unittest.TestCase):
    def setUp(self):
        # parameters
        dim = 2
        max_j = 20
        max_k = 20

        # data
        self.index_constructor = IndexConstructor(dim,
                                                  max_j,
                                                  max_k)

        self.schema_constructor = SchemaConstructor(dim,
                                                    max_j,
                                                    max_k)

    def test_shift(self):
        """ THIS TEST IS NOT COMPLETE """
        x1 = torch.Tensor([[0.23, 0.23]])

        js1, ks1, vs1 = self.index_constructor.get_jkvs(x1)

        d = 0
        # breakpoint()
        schema_1 = self.schema_constructor.get_schema(js1[d],
                                                      ks1[d],
                                                      vs1[d])

        schema_2 = self.schema_constructor.get_schema(js1[d],
                                                      ks1[d],
                                                      vs1[d])

        schema_1 = schema_1.shift_inputs(1)
        for key in schema_2:
            schema_2[key] = set([1])

        self.assertEqual(schema_1, schema_2)

    @unittest.skip("This test is not currently working")
    def test_shift_multiple(self):
        d = 0
        # multiple_inputs
        x2 = torch.Tensor([[0.23, 0.23],
                           [0.23, 0.26]])

        js2, ks2, vs2 = self.index_constructor.get_jkvs(x2)
        schema_3 = self.schema_constructor.get_schema(js2[d],
                                                      ks2[d],
                                                      vs2[d])

        schema_4 = self.schema_constructor.get_schema(js2[d],
                                                      ks2[d],
                                                      vs2[d])
        for key in schema_4:
            schema_4[key] = set([1, 2])
        schema_3 = schema_3.shift_inputs(1)
        self.assertEqual(schema_3, schema_4)

    def test_add(self):
        # individual parts
        x1 = torch.Tensor([[0.23, 0.23]])
        x2 = torch.Tensor([[0.23, 0.26]])

        js1, ks1, vs1 = self.index_constructor.get_jkvs(x1)
        js2, ks2, vs2 = self.index_constructor.get_jkvs(x2)

        # combined parts
        d = 0
        schema_11 = self.schema_constructor.get_schema(js1[d],
                                                       ks1[d],
                                                       vs1[d])

        schema_21 = self.schema_constructor.get_schema(js2[d],
                                                       ks2[d],
                                                       vs2[d])
        schema_21 = schema_21.shift_inputs(1)
        test_schema_1 = schema_11 + schema_21

        # build a dictionary and create a Schema from it
        skeleton_dict = {"0K": {0, 1},
                         "1K": {0, 1},
                         "2K": {0, 1},
                         "3K": {0, 1},
                         "4K": {0, 1},
                         "5K": {0, 1},
                         "6K": {0, 1},
                         "7K": {0, 1},
                         "8K": {0, 1},
                         "9K": {0, 1},
                         "AK": {0, 1},
                         "BK": {0, 1},
                         "CK": {0, 1},
                         "DK": {0, 1},
                         "EK": {0, 1},
                         "FK": {0, 1},
                         "GK": {0, 1},
                         "HK": {0, 1},
                         "IK": {0, 1},
                         "JK": {0, 1},
                         "KK": {0, 1},
                         "LK": {0, 1},
                         "MK": {0, 1},
                         "NL": {0, 1},
                         "ON": {0, 1},
                         "PR": {0, 1},
                         "QY": {0, 1}}

        schema_1 = Schema(skeleton_dict)
        self.assertEqual(schema_1, test_schema_1)

        d = 1
        schema_12 = self.schema_constructor.get_schema(js1[d],
                                                       ks1[d],
                                                       vs1[d])

        schema_22 = self.schema_constructor.get_schema(js2[d],
                                                       ks2[d],
                                                       vs2[d])
        schema_22 = schema_22.shift_inputs(1)
        test_schema_2 = schema_12 + schema_22

        # build a dictionary and create a Schema from it
        skeleton_dict_2 = {"0K": {0, 1},
                           "1K": {0, 1},
                           "2K": {0, 1},
                           "3K": {0, 1},
                           "4K": {0, 1},
                           "5K": {0, 1},
                           "6K": {0, 1},
                           "7K": {0, 1},
                           "8K": {0, 1},
                           "9K": {0, 1},
                           "AK": {0, 1},
                           "BK": {0, 1},
                           "CK": {0, 1},
                           "DK": {0, 1},
                           "EK": {0, 1},
                           "FK": {0, 1},
                           "GK": {0, 1},
                           "HK": {0, 1},
                           "IK": {0, 1},
                           "JK": {0, 1},
                           "KK": {0, 1},
                           "LK": {0, 1},
                           "MK": {0},
                           "ML": {1},
                           "NL": {0},
                           "NM": {1},
                           "ON": {0},
                           "OO": {1},
                           "PR": {0},
                           "PS": {1},
                           "QY": {0},
                           "Qa": {1}
                           }

        schema_2 = Schema(skeleton_dict_2)
        self.assertEqual(schema_2, test_schema_2)

        # make sure the order doesn't matter
        self.assertEqual(schema_12 + schema_22, schema_22 + schema_12)
        self.assertEqual(schema_11 + schema_21, schema_21 + schema_11)


class TestSchemata(unittest.TestCase):
    def setUp(self):
        # parameters
        self.dim = 2
        self.max_j = 20
        self.max_k = 20

        # data
        self.x1 = torch.Tensor([[0.23, 0.23]])
        self.x2 = torch.Tensor([[0.23, 0.26]])
        self.x3 = torch.Tensor([[0.24, 0.26]])

        self.multiple_x = torch.Tensor([[0.23, 0.23],
                                        [0.23, 0.26]])
        self.schemata_constructor = SchemataConstructor(self.dim,
                                                        self.max_j,
                                                        self.max_k)

        # constructors and objects
        self.base_schemata = self.schemata_constructor.get_schemata(self.x1)
        self.new_schemata = self.schemata_constructor.get_schemata(self.x2)

    def test_add_multiple_inputs(self):
        # when adding multiple schemata, and adding multiple inputs, the
        # result will depend on the order because the order of the labels will
        # differ. The result is that two schemata will appear not equal even
        # if they technically represent the same information
        test_schemata = self.schemata_constructor.get_schemata(self.multiple_x)

        standard_schemata_1 = self.schemata_constructor.get_schemata(self.x1)
        standard_schemata_2 = self.schemata_constructor.get_schemata(self.x2)
        standard_schemata = standard_schemata_1 + standard_schemata_2
        self.assertEqual(standard_schemata, test_schemata)

    def test_add(self):
        # x = torch.Tensor([[0.23, 0.23]]),
        #                   [0.23, 0.23]])
        final_schemata = self.base_schemata + self.new_schemata

        # test that the order of addition makes no difference
        # the order of difference should not make a difference - but we also
        # need to take into account that they are just labels!
        # how can I check this?

        skeleton_dicts = [{"0K": {0, 1},
                           "1K": {0, 1},
                           "2K": {0, 1},
                           "3K": {0, 1},
                           "4K": {0, 1},
                           "5K": {0, 1},
                           "6K": {0, 1},
                           "7K": {0, 1},
                           "8K": {0, 1},
                           "9K": {0, 1},
                           "AK": {0, 1},
                           "BK": {0, 1},
                           "CK": {0, 1},
                           "DK": {0, 1},
                           "EK": {0, 1},
                           "FK": {0, 1},
                           "GK": {0, 1},
                           "HK": {0, 1},
                           "IK": {0, 1},
                           "JK": {0, 1},
                           "KK": {0, 1},
                           "LK": {0, 1},
                           "MK": {0, 1},
                           "ON": {0, 1},
                           "NL": {0, 1},
                           "PR": {0, 1},
                           "QY": {0, 1}},
                          {"0K": {0, 1},
                           "1K": {0, 1},
                           "2K": {0, 1},
                           "3K": {0, 1},
                           "4K": {0, 1},
                           "5K": {0, 1},
                           "6K": {0, 1},
                           "7K": {0, 1},
                           "8K": {0, 1},
                           "9K": {0, 1},
                           "AK": {0, 1},
                           "BK": {0, 1},
                           "CK": {0, 1},
                           "DK": {0, 1},
                           "EK": {0, 1},
                           "FK": {0, 1},
                           "GK": {0, 1},
                           "HK": {0, 1},
                           "IK": {0, 1},
                           "JK": {0, 1},
                           "KK": {0, 1},
                           "LK": {0, 1},
                           "MK": {0},
                           "ON": {0},
                           "NL": {0},
                           "PR": {0},
                           "QY": {0},
                           "ML": {1},
                           "NM": {1},
                           "OO": {1},
                           "PS": {1},
                           "Qa": {1}
                           }]

        test_schemata = self.schemata_constructor.get_schemata(self.x1)
        for d in range(self.dim):
            schema = Schema(skeleton_dicts[d])
            test_schemata[d] = schema.copy()

        self.assertEqual(final_schemata, test_schemata)

    @unittest.skip("Test Not Complete: test_combine")
    def test_combine(self):
        # build the combined schemata
        skeleton_dict = {("NM", "LL"): {1, 2},
                         ("NN", "LM"): {1, 2},
                         ("NO", "LO"): {1, 2},
                         ("NP", "LS"): {1, 2},
                         ("NQ", "La"): {1, 2},
                         ("OM", "NL"): {1, 2},
                         ("ON", "NM"): {1, 2},
                         ("OO", "NO"): {1, 2},
                         ("OP", "NS"): {1, 2},
                         ("OQ", "Na"): {1, 2},
                         ("PM", "RL"): {1, 2},
                         ("PN", "RM"): {1, 2},
                         ("PO", "RO"): {1, 2},
                         ("PP", "RS"): {1, 2},
                         ("PQ", "Ra"): {1, 2}}

        test_combined_schema = Schema(skeleton_dict)
        final_schemata = self.base_schemata + self.new_schemata

        # get the test version
        next_input_schemata = self.schemata_constructor.get_schemata(self.x3)
        combined_schema = final_schemata.combine(next_input_schemata)

        self.assertEqual(test_combined_schema, combined_schema)


if __name__ == "__main__":
    unittest.main()
