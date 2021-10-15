# sparse_haar/tests/sparse_haar_test.py
import unittest
import torch
# from sparse_haar.schemata import Schema, Schemata, SchemataConstructor,\
#                SchemaConstructor, IndexConstructor

from sparse_haar.schemata import SchemataConstructor,\
    IndexConstructor, SchemaConstructor, Schema, Schemata


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
            self.assertTrue((self.js[d] == torch.Tensor([3., 4., 5., 6.],
                                                        )).all())

    def test_ks_correct(self):
        js = self.index_constructor._construct_js(self.x1)
        ks, vs = self.index_constructor._construct_ks_and_values(self.x1, js)
        for d in range(self.dim):
            self.assertTrue((ks[d] == torch.Tensor([1., 3., 7., 14.])).all())
            self.assertTrue((vs[d] == torch.Tensor([-1.,
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
        self.assertEqual(words, set(["NL", "ON", "PR", "QY"]))

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
        self.assertEqual(test_output.keys(), {"NL", "ON", "PR", "QY"})
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

        schema_1.shift_inputs(1)
        for key in schema_2:
            schema_2[key] = set([1])

        self.assertEqual(schema_1, schema_2)

    @unittest.skip("Skipping Schema shift_multiple")
    def test_shift_multiple(self):
        d = 0
        # multiple_inputs
        x2 = torch.Tensor([[0.23, 0.23],
                           [0.23, 0.23]])

        js2, ks2, vs2 = self.index_constructor.get_jkvs(x2)
        schema_3 = self.schema_constructor.get_schema(js2[d],
                                                      ks2[d],
                                                      vs2[d])

        schema_4 = self.schema_constructor.get_schema(js2[d],
                                                      ks2[d],
                                                      vs2[d])
        for key in schema_4:
            schema_4[key] = set([1, 2])
        schema_3.shift_inputs(1)
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
        schema_21.shift_inputs(1)
        test_schema_1 = schema_11 + schema_21

        # build a dictionary and create a Schema from it
        skeleton_dict = {"ON": {0, 1},
                         "NL": {0, 1},
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
        schema_22.shift_inputs(1)
        test_schema_2 = schema_12 + schema_22

        # build a dictionary and create a Schema from it
        skeleton_dict_2 = {"ON": {0},
                           "NL": {0},
                           "PR": {0},
                           "QY": {0},
                           "ML": {1},
                           "NM": {1},
                           "OO": {1},
                           "PS": {1},
                           "Qa": {1}
                           }

        schema_2 = Schema(skeleton_dict_2)
        self.assertEqual(schema_2, test_schema_2)


class TestSchemata(unittest.TestCase):
    def setUp(self):
        # parameters
        self.dim = 2
        self.max_j = 20
        self.max_k = 20

        # data
        self.x1 = torch.Tensor([[0.23, 0.23]])
        self.x2 = torch.Tensor([[0.23, 0.26]])

        self.schemata_constructor = SchemataConstructor(self.dim,
                                                        self.max_j,
                                                        self.max_k)

        # constructors and objects
        self.base_schemata = self.schemata_constructor.get_schemata(self.x1)
        self.new_schemata = self.schemata_constructor.get_schemata(self.x2)

    @unittest.skip("Skipping Schemata combine")
    def test_combine_one(self):
        # since there are no common sets in each of the
        x2 = torch.Tensor([[0.23, 0.26]])

        # what is the resulting case?
        copied_base_schemata = self.base_schemata.copy()
        test_schemata = self.schemata_constructor.get_schemata(self.x2)

        # the last layer of test_schemata should now be empty
        self.base_schemata.combine(test_schemata)
        diction = copied_base_schemata[-1]
        for key in diction:
            self.assertEqual(diction[key], set())

    def test_add(self):
        # x = torch.Tensor([[0.23, 0.23]]),
        #                   [0.23, 0.23]])
        final_schemata = self.base_schemata + self.new_schemata
        skeleton_dicts = [{
                         "ON": {0, 1},
                         "NL": {0, 1},
                         "PR": {0, 1},
                         "QY": {0, 1}},
                         {"ON": {0},
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


if __name__ == "__main__":
    unittest.main()
