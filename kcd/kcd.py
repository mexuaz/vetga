"""
Created on May 20, 2019

@author: A. Mehrafsa
"""

from io import StringIO

from os.path import splitext

# for profiling
from timeit import default_timer as timer  # For profiling the algorithm
from datetime import datetime  # For getting timestamp
from datetime import timedelta

import pstats
import cProfile

# main libraries
import numpy as np  # performing operations on CPU
import torch  # performing operations on GPU

# visualizing progress
from tqdm import tqdm as tq

from graphio.graphio import GraphIO
from graphutils.graphutils import GraphUtils


class KCD(object):

    def __init__(self,
                 filename,
                 skip_prepare=False,
                 mode="numpy",
                 serialize="none",
                 serialize_prepared=False,
                 comment="Default",
                 verbose=2):

        self.version = 1.1
        self.verbose = verbose
        self.mode = mode

        if self.verbose > 1:
            print("V"*30)
            print(f"Core decomposition version {self.version} started in {self.mode} mode ")

        self.N = np.array([])
        self.D = np.array([])
        self.I = np.array([])
        self.II = np.array([])
        self.d_max = np.array([])

        if self.mode == "numpy":
            self.multi_arrange = GraphUtils.multi_arange_numpy
            self.unique = GraphUtils.unique_numpy_bin_count
        elif self.mode == "torch-gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available in this platform")
            self.multi_arrange = GraphUtils.multi_arange_torch_gpu
            self.unique = GraphUtils.unique_torch_bin_count
        elif self.mode == "torch-cpu":
            self.multi_arrange = GraphUtils.multi_arange_torch_cpu
            self.unique = GraphUtils.unique_torch_bin_count
        else:
            raise RuntimeError(f"Unsupported mode {self.mode}!")

        self.profiler = dict()  # profiler dictionary

        self.profiler["version"] = self.version
        self.profiler["mode"] = mode
        self.profiler["comment"] = comment
        self.profiler["data-time"] = datetime.now()
        self.profiler["input-file"] = filename
        self.profiler["cpu-processor"], self.profiler["cpu-cores"] = GraphIO.linux_cpu_info()
        if torch.cuda.is_available():
            self.profiler["gpu-processor"] = torch.cuda.get_device_name(torch.cuda.current_device())
        else:
            self.profiler["gpu-processor"] = "NA"
        self.profiler['analytics'] = -1
        self.profiler['stats'] = ""

        self.c_profiler = cProfile.Profile()
        self.c_profiler.disable()

        self.gio = GraphIO(verbose=self.verbose)

        if skip_prepare and serialize_prepared:
            raise RuntimeError("Skip-prepared and serialize prepared could not be used with together.")

        if serialize_prepared:
            src, dst, self.N = self.gio.read(filename, True)
            if serialize != "none":
                self.gio.write(GraphIO.serialized_filename(filename, serialize), src, dst)
        else:
            src, dst, n = self.gio.read(filename, False)
            if serialize != "none":
                self.gio.write(GraphIO.serialized_filename(filename, serialize), src, dst)
            if not skip_prepare:
                src, dst, self.N = GraphIO.prepare_input(src, dst, self.verbose)

        assert (src.dtype == self.gio.dtype)
        assert (dst.dtype == self.gio.dtype)
        assert (self.N.dtype == self.gio.dtype)
        self.initialize_essentials(src, dst)

        # used by teardown algorithm and build neighbors histogram
        self.d_max = self.D.max()

        # For storing the results
        self.K = np.zeros_like(self.D)

    def initialize_essentials(self, edge_list_src, edge_list_dst):

        if self.verbose > 0:
            print("Pre-processing the edge-list ...")
            tr = timer()

        # Evaluating degree of nodes
        # Get new D for new edge list we are going to need this for re-ordering
        _, self.D = np.unique(edge_list_src, return_index=False, return_counts=True)

        # Apply a stable first based on the degree of nodes in edge_list_dst array and then on edge_list_src
        sort_permutation = np.lexsort((self.D[edge_list_dst], edge_list_src))
        edge_list_src = edge_list_src[sort_permutation]
        edge_list_dst = edge_list_dst[sort_permutation]

        # II: Array containing the start index of neighbors block in edge_list_dst array
        # update D after applying stable sort ordering
        _, self.II, self.D = np.unique(edge_list_src, return_index=True, return_counts=True)
        self.D = self.D.astype(edge_list_src.dtype)
        self.II = self.II.astype(edge_list_src.dtype)

        self.I = edge_list_dst

        # make sure everything is type of IO data type
        assert (self.D.dtype == self.gio.dtype)
        assert (self.I.dtype == self.gio.dtype)
        assert (self.II.dtype == self.gio.dtype)

        if self.verbose > 0:
            print(f"Pre-processing finished in {str(timedelta(seconds=timer() - tr))}.")

    def run(self):
        if self.verbose:
            self.run_original_verbose()
        else:
            self.run_original()

    def prepare_arrays(self, histogram_algorithm=False):
        """
        Prepare arrays such as converting them to proper types
         and uploading them into device (GPU ...) memory
         in order to start the algorithm (run_ methods)
        :param histogram_algorithm:
        :return:
        """
        if self.mode == "numpy":
            # Generating local editable arrays
            D = self.D.copy()  # Copy of degree array
            K = np.zeros_like(self.K)
            I = self.I.copy()

            II = self.II.copy()
            CD = self.D.copy()  # copy of D to remain unchanged

            C = np.arange(len(self.N), dtype=self.gio.dtype)  # Indices of nodes that are relevant
        elif self.mode == "torch-cpu":
            torch.set_grad_enabled(False)

            D = torch.LongTensor(self.D.astype(np.int64))  # Copy of degree array
            K = torch.LongTensor(len(self.D)).fill_(0)
            I = torch.LongTensor(self.I.astype(np.int64))

            II = torch.LongTensor(self.II.astype(np.int64))
            CD = torch.LongTensor(self.D.astype(np.int64))  # Copy of D to remain unchanged

            C = torch.arange(len(self.N), dtype=torch.int64)  # Indices of nodes that are relevant
        elif self.mode == "torch-gpu":
            torch.set_grad_enabled(False)

            # Uploading the arrays to GPU
            D = torch.cuda.LongTensor(self.D.astype(np.int64))  # Copy of degree array
            K = torch.cuda.LongTensor(len(self.D)).fill_(0)
            I = torch.cuda.LongTensor(self.I.astype(np.int64))

            II = torch.cuda.LongTensor(self.II.astype(np.int64))
            CD = torch.cuda.LongTensor(self.D.astype(np.int64))  # copy of D to remain unchanged

            C = torch.arange(len(self.N), dtype=torch.int64, device='cuda')  # Indices of nodes that are relevant
        else:
            raise RuntimeError(f"Unsupported mode {self.mode}!")

        return D, K, I, II, CD, C

    def collect_results(self, K):
        if self.mode == "numpy":
            self.K = K
        elif self.mode == "torch-gpu":
            # Downloading the results to cpu
            self.K = K.cpu().data.numpy().astype(self.gio.dtype)
        elif self.mode == "torch-cpu":
            # Downloading the results to cpu
            self.K = K.data.numpy().astype(self.gio.dtype)
        else:
            raise RuntimeError(f"Unsupported mode {self.mode}!")

        print(f"Max K {np.max(self.K)}")
        print(f"AVG K {np.mean(self.K):.2f}")

    # main algorithm without histogram
    def run_original(self):
        # We actually do not need bins here
        D, K, I, II, CD, C = self.prepare_arrays()

        k = 1

        self.c_profiler.clear()
        self.c_profiler.enable()

        B = C[D[C] <= k]  # indices of nodes that will be deleted

        while C.shape[0] > 1:
            while B.shape[0] > 0:
                D[B] = 0
                K[B] = k

                # subtracting from neighbors
                m = self.multi_arrange(II[B], CD[B])
                J = I[m]
                H = J[D[J] > 0]

                H, cnt = self.unique(H)
                D[H] -= cnt
                B = H[D[H] <= k]  # indices of nodes that will be deleted

                # np.subtract.at(D, H, 1)
                # B = N[((D <= k) & (D > 0))]

                # End of inner loop

            k = k + 1
            C = C[D[C] >= k]  # indices of nodes that are relevant
            B = C[D[C] == k]  # indices of nodes that will be deleted


        # End of outer loop
        self.c_profiler.disable()
        ps_io = StringIO()
        ps = pstats.Stats(self.c_profiler, stream=ps_io)
        self.profiler['analytics'] = ps.total_tt
        ps.strip_dirs().sort_stats('cumtime').print_stats()  # will print into profiler['stats']
        # self.profiler['stats'] = ps_io.getvalue().strip()

        self.collect_results(K)

        return
    # main algorithm without histogram

    def run_original_verbose(self):
        # We actually do not need bins here
        D, K, I, II, CD, C = self.prepare_arrays()

        k = 1

        if self.verbose > 0:
            print(f"Analytics start ...")
            if self.verbose > 1:
                pbar = tq(total=len(K), miniters=1, desc=f"Coreness {k + 1}")
                nz = 0

        self.c_profiler.clear()
        self.c_profiler.enable()

        B = C[D[C] <= k]  # indices of nodes that will be deleted

        while C.shape[0] > 1:
            while B.shape[0] > 0:
                D[B] = 0
                K[B] = k

                # subtracting from neighbors
                m = self.multi_arrange(II[B], CD[B])
                J = I[m]
                H = J[D[J] > 0]

                H, cnt = self.unique(H)
                D[H] -= cnt
                B = H[D[H] <= k]  # indices of nodes that will be deleted

                # np.subtract.at(D, H, 1)
                # B = N[((D <= k) & (D > 0))]

                # End of inner loop

            k = k + 1
            C = C[D[C] >= k]  # indices of nodes that are relevant
            B = C[D[C] == k]  # indices of nodes that will be deleted

            if self.verbose > 1:
                if self.mode == "numpy":
                    nz_t = np.count_nonzero(K)
                elif self.mode == "torch-cpu" or self.mode == "torch-gpu":
                    nz_t = torch.sum(K != 0).item()
                else:
                    raise RuntimeError(f"Unknown mode: {self.mode}")
                pbar.update(nz_t - nz)
                nz = nz_t
                pbar.set_description(desc=f"Coreness {k + 1}", refresh=True)

        # End of outer loop
        self.c_profiler.disable()
        ps_io = StringIO()
        ps = pstats.Stats(self.c_profiler, stream=ps_io)
        self.profiler['analytics'] = ps.total_tt
        ps.strip_dirs().sort_stats('cumtime').print_stats()  # will print into profiler['stats']
        # self.profiler['stats'] = ps_io.getvalue().strip()

        if self.verbose > 0:
            print(f"Analytics finished in {str(timedelta(seconds=self.profiler['analytics']))}.")
            if self.verbose > 1:
                pbar.close()

        self.collect_results(K)

        return

    def write_results(self, filename):

        if len(self.N) != len(self.K):
            raise RuntimeError(f"The cores results size {len(self.K)} does not match nodes size {len(self.N)}")

        _, ext = splitext(filename)

        if ext == ".txt":
            self.gio.write_txt(filename, self.N, self.K, mtx_header=False)
        elif ext == ".mtx":
            self.gio.write_txt(filename, self.N, self.K, mtx_header=True)
        elif ext == ".npz":
            self.gio.write_npz(filename, N=self.N, K=self.K)
        elif ext == ".h5" or ext == ".hdf5" or ext == ".hdf":
            self.gio.write_hdf5(filename, N=self.N, K=self.K)
        else:
            raise RuntimeError((f"Output file of type {ext} is not supported.\n"
                                "Only .txt|.npz|.h5|.mtx are supported."))

    def write_profile(self, filename, append=True):
        self.gio.write_csv(filename, self.profiler, append)
