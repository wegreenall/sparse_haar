import unittest
import torch
# from sparse_haar.schemata import Schema, Schemata, SchemataConstructor,\
#                SchemaConstructor, IndexConstructor

from sparse_haar.schemata import SchemataConstructor,\
    IndexConstructor, SchemaConstructor


class TestSchemaMethods(unittest.TestCase):
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
        x2 = torch.Tensor([[0.23, 0.23],
                           [0.23, 0.23]])

        js1, ks1, vs1 = self.index_constructor.get_jkvs(x1)

        d = 0
        schema_1 = self.schema_constructor.get_schema(js1,
                                                      ks1[:, :, d],
                                                      vs1[:, :, d])

        schema_2 = self.schema_constructor.get_schema(js1,
                                                      ks1[:, :, d],
                                                      vs1[:, :, d])
        schema_1.shift_inputs(1)
        for key in schema_2:
            schema_2[key] = set([1])

        self.assertEqual(schema_1, schema_2)

    def test_shift_multiple(self):
        d = 0
        # multiple_inputs
        x2 = torch.Tensor([[0.23, 0.23],
                           [0.23, 0.23]])

        js2, ks2, vs2 = self.index_constructor.get_jkvs(x2)
        schema_3 = self.schema_constructor.get_schema(js2,
                                                      ks2[:, :, d],
                                                      vs2[:, :, d])

        schema_4 = self.schema_constructor.get_schema(js2,
                                                      ks2[:, :, d],
                                                      vs2[:, :, d])
        for key in schema_4:
            schema_4[key] = set([1, 2])
        schema_3.shift_inputs(1)
        self.assertEqual(schema_3, schema_4)

    #@unittest.skip("Skipping Schema add")
    def test_add(self):
        # individual parts
        x1 = torch.Tensor([[0.23, 0.23]])
        x2 = torch.Tensor([[0.23, 0.29]])

        js1, ks1, vs1 = self.index_constructor.get_jkvs(x1)
        js2, ks2, vs2 = self.index_constructor.get_jkvs(x2)

        # combined parts
        x = torch.Tensor([[0.23, 0.23],
                          [0.23, 0.29]])

        js, ks, vs = self.index_constructor.get_jkvs(x)

        d = 0
        schema_1 = self.schema_constructor.get_schema(js1,
                                                      ks1[:, :, d],
                                                      vs1[:, :, d])

        schema_2 = self.schema_constructor.get_schema(js2,
                                                      ks2[:, :, d],
                                                      vs2[:, :, d])
        schema_2.shift_inputs(1)
        test_schema = schema_1 + schema_2
        schema = self.schema_constructor.get_schema(js,
                                                    ks[:, :, d],
                                                    vs[:, :, d])

        self.assertEqual(schema, test_schema)


class TestSchemataMethods(unittest.TestCase):
    def setUp(self):
        # parameters
        dim = 2
        max_j = 20
        max_k = 20

        # data
        x1 = torch.Tensor([[0.23, 0.23]])
        x2 = torch.Tensor([[0.23, 0.29]])

        self.schemata_constructor = SchemataConstructor(dim, max_j, max_k)
        self.new_schemata = self.schemata_constructor.get_schemata(x2)

        # constructors and objects
        self.base_schemata = self.schemata_constructor.get_schemata(x1)

    def test_combine(self):
        x2 = torch.Tensor([[0.23, 0.26]])
        new_schemata = self.schemata_constructor.get_schemata(x2)
        pass

    def test_add(self):
        x = torch.Tensor([[0.23, 0.23],
                          [0.23, 0.29]])
        final_schemata = self.base_schemata + self.new_schemata
        test_schemata = self.schemata_constructor.get_schemata(x)
        # breakpoint()
        self.assertEqual(final_schemata, test_schemata)


if __name__ == "__main__":
    unittest.main()
