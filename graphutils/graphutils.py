"""
Created on May 20, 2019

@author: A. Mehrafsa
"""

# main libraries
import numpy as np  # performing operations on CPU
import torch  # performing operations on GPU


class GraphUtils(object):

    def __init__(self, library="numpy"):
        self.version = 1.1
        self.multi_arange = None
        self.multi_arange_torch = None
        self.join = None
        self.unique = None

        if library == "numpy":
            self.multi_arange = GraphUtils.multi_arange_numpy
            self.join = GraphUtils.join_numpy
            self.unique = GraphUtils.unique_numpy_bin_count
        elif library == "torch-cpu":
            self.multi_arange_torch = GraphUtils.multi_arange_torch_cpu
            self.multi_arange = self.multi_arange_torch
            self.join = self.join_torch
            self.unique = GraphUtils.unique_torch_bin_count
        elif library == "torch-gpu" or library == "torch-cuda":
            self.multi_arange_torch = GraphUtils.multi_arange_torch_gpu
            self.multi_arange = self.multi_arange_torch
            self.join = self.join_torch
            self.unique = GraphUtils.unique_torch_bin_count
        else:
            raise RuntimeError("Unknown library passed to GraphUtils module!")

    @staticmethod
    def join_numpy(src, dst, deg, nei):
        """
        This will return all two-hop passes from src to des vectors
        :param src: source nodes vector in edge list
        :param dst: destination nodes vector in edge list
        :param deg: degree of nodes
        :param nei: Index array for where each block of neighbors start
        :return: list of (st, dn) pairs connected by any intermediate nodes
        """
        st = np.repeat(src, deg[dst])  # source nodes repeat
        m = GraphUtils.multi_arange_numpy(nei[dst], deg[dst])
        dn = dst[m]  # destination neighbors
        return st, dn

    def join_torch(self, src, dst, deg, nei):
        st = torch.repeat_interleave(src, deg[dst])  # source nodes repeat
        m = self.multi_arange_torch(nei[dst], deg[dst])
        dn = dst[m]  # destination neighbors
        return st, dn

    @staticmethod
    def multi_arange_nei(start, count):

        nnz = np.nonzero(count)
        start = start[nnz]
        count = count[nnz]

        arr_len = np.sum(count)

        # building reset indices
        nei = np.zeros(count.shape[0]+1, dtype=count.dtype)
        nei[1:] = np.cumsum(count)
        ri = (nei[:-1]).copy()

        # building incremental indices
        # This vector require to be long to handle negative numbers properly
        incr = np.ones(arr_len.item(), dtype=np.long)
        incr[ri] = start

        # correcting start indices for initial values
        # np.add.at(incr, ri[1:], 1 - (start[:-1] + count[:-1]))
        incr[ri[1:]] += 1 - (start[:-1] + count[:-1]).astype(np.long)

        # typecast to normal data type
        return np.cumsum(incr).astype(start.dtype), nei

    @staticmethod
    def multi_arange_numpy(start, count):

        nnz = np.nonzero(count)
        start = start[nnz]
        count = count[nnz]

        arr_len = np.sum(count)

        # building reset indices
        ri = np.zeros(count.shape[0], dtype=count.dtype)
        ri[1:] = np.cumsum(count)[:-1]

        # building incremental indices
        # This vector require to be long to handle negative numbers properly
        incr = np.ones(arr_len.item(), dtype=np.long)
        incr[ri] = start

        # correcting start indices for initial values
        # np.add.at(incr, ri[1:], 1 - (start[:-1] + count[:-1]))
        incr[ri[1:]] += 1 - (start[:-1] + count[:-1]).astype(np.long)

        # typecast to normal data type
        return np.cumsum(incr).astype(start.dtype)

    @staticmethod
    def multi_arange_torch_ex(start, count, device):

        nnz = torch.nonzero(count).reshape(-1)
        start = start[nnz]
        count = count[nnz]

        arr_len = torch.sum(count)

        # building reset indices
        ri = torch.zeros(count.shape[0], dtype=count.dtype, device=device, requires_grad=False)
        ri[1:] = torch.cumsum(count, dim=0)[:-1]

        # building incremental indices
        # This vector require to be long to handle negative numbers properly
        incr = torch.ones(arr_len.item(), dtype=torch.int64, device=device, requires_grad=False)
        incr[ri] = start

        # correcting start indices for initial values
        # torch.add.at(incr, ri[1:], 1 - (start[:-1] + count[:-1]))
        incr[ri[1:]] += 1 - (start[:-1] + count[:-1]).long()

        return torch.cumsum(incr, dim=0).type(start.dtype)

    @staticmethod
    def multi_arange_torch_cpu(start, count):
        return GraphUtils.multi_arange_torch_ex(start, count, 'cpu')

    @staticmethod
    def multi_arange_torch_gpu(start, count):
        return GraphUtils.multi_arange_torch_ex(start, count, 'cuda')

    @staticmethod
    def unique_numpy_bin_count(vec):
        q = np.bincount(vec).astype(vec.dtype)
        u = np.nonzero(q)[0].astype(vec.dtype)  # nonzero return tuple of arrays
        return u, q[u]

    @staticmethod
    def unique_torch_bin_count(vec):
        q = torch.bincount(vec)
        u = torch.nonzero(q).reshape(-1)  # return 2d array
        return u, q[u]

    @staticmethod
    def cantor_pairing(s, d):
        """
        Cantor pairing function:

        $\mathbb{N}\times\mathbb{N}\rightarrow\mathbb{N}$

        $\pi(a,b) = \frac{1}{2}(a+b)(a+b+1) + b$

        :param s: first vector
        :param d: second vector
        :return: pairing values
        """
        return (1 / 2 * (s + d) * (s + d + 1)).astype(s.dtype) + d

    @staticmethod
    def associative_pairing(s, d):
        """
        Producing unique number given two interchangable integers (associative pairing)

        $f(a,b) = \dfrac{\max(a,b)(\max(a,b)+1)}{2}+\min(a,b)$

        This satisfies $f(a,b)=f(b,a)$ and grows quadratically with $max(a,b)$.

        Growth pattern:
        * $f(0,0) = 0$
        * $f(1,0) = 1, f(1,1) = 2$
        * $f(2,0) = 3, f(2,1) = 4, f(2,2) = 5$
        * $f(3,0) = 6, f(3,1) = 7, f(3,2) = 8, f(3,3) = 9$

        :param s: first vector
        :param d: second vector
        :return: pairing values
        """
        m = np.maximum(s, d)
        n = np.minimum(s, d)
        return (m * (m + 1) / 2).astype(m.dtype) + n


