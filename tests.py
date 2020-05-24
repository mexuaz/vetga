import unittest

import numpy as np
import torch
from kcd.kcd import KCD
from graphio.graphio import GraphIO
from graphutils.graphutils import GraphUtils

from glob import glob
import os.path
import re


class TestMutliArange(unittest.TestCase):
    def test_numpy(self):
        print("Testing Multi-arange Function in Numpy ...")
        s = np.array([2, 6, 0, 6, 10, 13, 0, 2, 10, 13, 2, 6, 13, 2, 6, 10], dtype=np.uint32)
        c = np.array([4, 4, 2, 4, 3, 3, 2, 4, 3, 3, 4, 4, 3, 4, 4, 3], dtype=np.uint32)
        e = np.array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 6, 7, 8, 9, 10, 11, 12,
                      13, 14, 15, 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 2, 3,
                      4, 5, 6, 7, 8, 9, 13, 14, 15, 2, 3, 4, 5, 6, 7, 8, 9,
                      10, 11, 12], dtype=np.uint32)

        m = GraphUtils.multi_arange_numpy(s, c)
        np.testing.assert_array_equal(m, e)

        st = torch.from_numpy(s.astype(np.int64))
        ct = torch.from_numpy(c.astype(np.int64))
        et = torch.from_numpy(e.astype(np.int64))
        mt = GraphUtils.multi_arange_torch_ex(st, ct, 'cpu')  # mt.data.numpy()
        self.assertTrue(torch.all(torch.eq(mt, et)))


class TestIO(unittest.TestCase):
    def test_read_txt(self):
        print("Testing IO Functions ...")
        for ig in glob("./datasets/sample_input_graph_*.txt"):
            kcd = KCD(ig)
            g = GraphIO.reconstruct_original_edge_list(kcd.N, kcd.D, kcd.I)

            # Load text file and do pre-processing
            arr_txt = np.loadtxt(ig, dtype=kcd.gio.dtype, comments="#")
            src, dst = GraphIO.remove_self_loops(arr_txt[:, 0], arr_txt[:, 1])
            src, dst = GraphIO.directed_to_undirected(src, dst)
            sort_perm = np.lexsort((dst, src))
            src = src[sort_perm]
            dst = dst[sort_perm]
            el = GraphIO.merge(src, dst)

            self.assertEqual(g.dtype, el.dtype)
            np.testing.assert_array_equal(g, el)

    def test_serialize(self):
        print("Testing Serialization ...")
        for ig in glob("./datasets/sample_input_graph_*.txt"):
            base_path, ext = os.path.splitext(ig)

            kcd_txt = KCD(ig,
                          skip_prepare=False,
                          mode="numpy",
                          serialize="h5",
                          serialize_prepared=False)
            kcd_h5 = KCD(base_path + ".h5",
                         skip_prepare=False,
                         mode="numpy",
                         serialize="mtx",
                         serialize_prepared=False)
            kcd_mtx = KCD(base_path + ".mtx",
                          skip_prepare=False,
                          mode="numpy",
                          serialize="npz",
                          serialize_prepared=False)
            kcd_npz = KCD(base_path + ".npz",
                          skip_prepare=False,
                          mode="numpy",
                          serialize="none",
                          serialize_prepared=False)

            g_txt = GraphIO.reconstruct_original_edge_list(kcd_txt.N, kcd_txt.D, kcd_txt.I)
            g_h5 = GraphIO.reconstruct_original_edge_list(kcd_h5.N, kcd_h5.D, kcd_h5.I)
            g_mtx = GraphIO.reconstruct_original_edge_list(kcd_mtx.N, kcd_mtx.D, kcd_mtx.I)
            g_npz = GraphIO.reconstruct_original_edge_list(kcd_npz.N, kcd_npz.D, kcd_npz.I)

            np.testing.assert_array_equal(g_txt, g_h5)
            np.testing.assert_array_equal(g_txt, g_mtx)
            np.testing.assert_array_equal(g_txt, g_npz)


class TestKCDAlgorithms(unittest.TestCase):

    def test_original_numpy_algorithms(self):
        print("Testing KCD on Numpy Platform ...")
        for ig in glob("./datasets/sample_input_graph_*.txt"):
            kcd = KCD(ig, mode="numpy")
            kcd.run()

            # Load output
            og = re.sub("sample_input", "expected_output_kcore", ig)
            arr_txt = np.loadtxt(og, dtype=kcd.gio.dtype, comments="#")

            self.assertEqual(kcd.K.dtype, arr_txt.dtype)
            np.testing.assert_array_equal(kcd.K, arr_txt[:, 1])

    def test_original_torch_algorithms(self):
        print("Testing KCD on Torch platform ...")
        for ig in glob("./datasets/sample_input_graph_*.txt"):
            kcd = KCD(ig, mode="torch-cpu")
            kcd.run()

            # Load output
            og = re.sub("sample_input", "expected_output_kcore", ig)
            arr_txt = np.loadtxt(og, dtype=kcd.gio.dtype, comments="#")

            self.assertEqual(kcd.K.dtype, arr_txt.dtype)
            np.testing.assert_array_equal(kcd.K, arr_txt[:, 1])


if __name__ == '__main__':
    unittest.main()
