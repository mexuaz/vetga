"""
Created on May 20, 2019

@author: A. Mehrafsa
"""

# prepare file
import os.path
import gzip
import shutil
import zipfile
import h5py

#  used by download_file_from_google_drive
import requests
import re
from urllib.parse import urlparse

# for profiling
from timeit import default_timer as timer  # For profiling the algorithm
from datetime import timedelta

import csv  # For storing profiler values

# main libraries
import numpy as np  # performing operations on CPU

# visualizing progress
from tqdm import tqdm as tq


class GraphIO(object):

    def __init__(self, dtype=np.uint32, verbose=2):

        self.version = 1.1
        self.dtype = dtype
        self.verbose = verbose

        if self.verbose > 1:
            print(f"Graph IO version {self.version}.")

    @staticmethod
    def merge(*args):
        """
        Merge src and dst vector to unified graph of two columns
        :param args:
        :return:
        """
        return np.hstack(tuple(arg.reshape(-1, 1) for arg in args))

    @staticmethod
    def split(graph):
        """
        Break multi column graph to list of multiple arrays
        :param graph:
        :return:
        """
        return [x.reshape(-1) for x in np.hsplit(graph, graph.shape[1])]

    @staticmethod
    def download_file_by_google_shared_link(shared_url, buffer_sz=32768):
        """
        Download a file from Google drive by shared link
        :param shared_url: sample url https://drive.google.com/file/d/17L0twEBlaDjyDpR5dK-6_PJPIZP9EVwC/view?usp=sharing
        :param buffer_sz: Download buffer
        :return: local file
        """

        def get_file_id(url, retrieve_name=False):
            parts = urlparse(url).path.split("/")

            if not retrieve_name:
                return parts[parts.index('d') + 1]

            # return file name if requested
            res = requests.get(url, allow_redirects=True)
            pattern = re.compile(r'<title.*?>(.+?) - Google Drive</title>')
            titles = re.findall(pattern, res.content.decode('utf-8'))
            return parts[parts.index('d') + 1], titles[0] if titles else None

        def get_confirm_token(res):
            for key, value in res.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        def save_response_content(res):

            file_size = int(res.headers.get("Content-Length", -1))

            # Get filename from the header
            filename = (res.headers['Content-Disposition'].split(";")[1].split("=")[1]).replace('"', '')

            with open(filename, "wb") as f:
                for chunk in tq(res.iter_content(buffer_sz), total=file_size,
                                unit='B', unit_scale=True, desc=f"downloading {filename}"):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            return filename

        url_e = "https://docs.google.com/uc?export=download"

        fid = get_file_id(shared_url)  # parse file id from URL

        session = requests.Session()

        response = session.get(url_e, params={'id': fid}, stream=True)

        token = get_confirm_token(response)

        if token:
            params = {'id': fid, 'confirm': token}
            response = session.get(url_e, params=params, stream=True)

        return save_response_content(response)

    @staticmethod
    def prepare_file(filename):
        """
        Download the file if hosted in google drive and extract if it is compressed
        ZIP files: The compressed archive should contain only one file
        :param filename: A remote file hosted on google drive with shared link or local compressed file (.zip or .gz)
        :return: Uncompressed local file
        """

        if filename.startswith("https://drive.google.com/file"):
            filename = GraphIO.download_file_by_google_shared_link(filename)
            if not filename:
                raise RuntimeError(f"Downloading {filename} filed!")

        # Check if the file is a compressed zip and extract the file
        basename = os.path.basename(filename)
        stem, ext = os.path.splitext(basename)
        if ext == ".zip":
            with zipfile.ZipFile(filename, 'r') as zipObj:
                zlist = zipObj.namelist()
                if len(zlist) != 1:
                    raise RuntimeError("Ambiguous input file: .zip file contains more than one file.")
                print(f"Extracting the file {zlist[0]} ...")
                # extract to the current directory
                # Will raise an Error in case of exception
                zipObj.extract(member=zlist[0], path='.')
                os.remove(filename)  # remove the zip file if successfully extracted
                filename = zlist[0]  # replace filename with extracted one
        elif ext == ".gz":
            with gzip.open(filename, 'rb') as f_in:
                with open(stem, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        return filename

    def read_txt(self, filename, mtx_header=False):

        if mtx_header:
            # First row is file type description: %%MatrixMarket matrix coordinate pattern general integer
            # Second row is: first col maximum, second col maximum and total edges
            arr = np.loadtxt(filename, dtype=self.dtype, skiprows=2)
        else:
            arr = np.loadtxt(filename, dtype=self.dtype, comments='#')

        return GraphIO.split(arr)

    def read_npz(self, filename):

        ka = np.load(filename)

        # change the data type if necessary
        data = [v.astype(self.dtype) for v in ka.values()]

        return data

    def read_hdf5(self, filename):

        with h5py.File(filename, 'r') as hf:
            data = [hf.get(k)[:].astype(self.dtype) for k in hf.keys()]

        return data

    @staticmethod
    def remove_self_loops(src, dst, verbose=0):
        """
        Remove self loops from provided edge list of graph
        :param src: first column of edge list (source nodes)
        :param dst: second column of edge list (destination nodes)
        :param verbose: verbosity of procedure
        :return: src, dst with removed self loops
        """
        if verbose > 0:
            print("Removing self loops ...")

        self_loops = (dst - src) != 0
        src = src[self_loops]
        dst = dst[self_loops]

        if verbose > 0:
            print(f"Removed {np.sum(np.logical_not(self_loops))} self loops.")

        return src, dst

    @staticmethod
    def map_to_consequent_integers(n, src, dst, verbose=0):
        """
        This solves the missing node ids problem (All ids will start from zero with no missing subsequent)
        Find the look up table (index_lookup) for re-indexing the nodes in edge_list[:,1]
        later we will use N to get the original indices
        :param n: list of nodes IDs
        :param src: first column of edge list (source nodes)
        :param dst: second column of edge list (destination nodes)
        :param verbose: verbosity of procedure
        :return: src, dst with consequent integer nodes id
        """

        if verbose > 0:
            print("Correcting the missing nodes ID and base index ...")

        index_lookup = np.full(shape=np.max(n) + np.uint32(1), fill_value=-1,
                               dtype=src.dtype)
        index_lookup[n] = np.arange(len(n), dtype=src.dtype)

        # Regenerate edge list with new ids in order and no missing values
        src = index_lookup[src]
        dst = index_lookup[dst]

        if verbose > 0:
            print("Finished mapping nodes ID to consequent integer numbers.")

        return src, dst

    @staticmethod
    def directed_to_undirected(src, dst, verbose=0):
        """
        Convert a given edge list of a directed graph to undirected graph by
        adding reverse edges where there is none
        This function also remove duplicated edges from edge list
        :param src: first column of edge list (source nodes)
        :param dst: second column of edge list (destination nodes)
        :param verbose: verbosity of procedure
        :return: src, dst with consequent integer nodes id
        """
        if verbose > 0:
            print("Converting directed edges to undirected ...")

        # Remove duplicates pairs from src and dst
        g = {(x, dst[i]): 1 for i, x in np.ndenumerate(src)}
        src_n = np.array([], dtype=src.dtype)
        dst_n = np.array([], dtype=dst.dtype)
        for k in g.keys():
            if (k[1], k[0]) not in g:
                src_n = np.append(src_n, k[1])
                dst_n = np.append(dst_n, k[0])

        if verbose > 0:
            print(f"Finished converting directed to undirected by inserting {len(src_n)} new edges.")

        # Reproducing src and dst from key pairs
        src, dst = GraphIO.split(np.array(list(g.keys())))

        return np.append(src, src_n), np.append(dst, dst_n)

    @staticmethod
    def single_directed(src, dst, verbose=0):
        """
        Convert a given edge list of a directed/undirected graph to directed graph without
        reverse edge
        This function also remove duplicated edges from edge list
        :param src: first column of edge list (source nodes)
        :param dst: second column of edge list (destination nodes)
        :param verbose: verbosity of procedure
        :return: src, dst with consequent integer nodes id
        """
        if verbose > 0:
            print("Converting directed/undirected edges to single-directed ...")

        # Remove duplicates pairs from src and dst
        g = {(x, dst[i]): 1 for i, x in np.ndenumerate(src)}
        n = 0
        for k in g.keys():
            if (k[1], k[0]) in g:
                del g[(k[1], k[0])]
                n = n + 1

        if verbose > 0:
            print(f"Finished converting directed/undirected to single-directed by removing {n} edges.")

        # Reproducing src and dst from key pairs
        return GraphIO.split(np.array(list(g.keys())))

    @staticmethod
    def prepare_input(src, dst, verbose=0):

        src, dst = GraphIO.remove_self_loops(src, dst, verbose)
        src, dst = GraphIO.directed_to_undirected(src, dst, verbose)

        n = np.unique(src, return_index=False, return_counts=False)
        src, dst = GraphIO.map_to_consequent_integers(n, src, dst, verbose)

        # Apply a stable sort based on the destination nodes of edge list and then source nodes
        sort_permutation = np.lexsort((dst, src))
        src = src[sort_permutation]
        dst = dst[sort_permutation]

        return src, dst, n

    def read(self, filename, prepare=True):
        """
        Read the graph file
        :param filename: edge list graph file
        :param prepare: Will remove self loops and duplicated edges, change directed edges
        to undirected edges by adding a reverse edge, replace non-continues node ids with
        continues nodes id
        :return: src, dst (edge list) and n (original nodes id return edges)
        """
        filename = GraphIO.prepare_file(filename)
        _, ext = os.path.splitext(filename)

        if self.verbose > 0:
            print(f"Reading edge list arrays from {filename} ...")
            tr = timer()

        if ext == ".txt":
            el_src, el_dst = self.read_txt(filename, mtx_header=False)
        elif ext == ".mtx":
            el_src, el_dst = self.read_txt(filename, mtx_header=True)
        elif ext == ".npz":
            el_src, el_dst = self.read_npz(filename)
        elif ext == ".h5" or ext == ".hdf" or ext == ".hdf5":
            el_src, el_dst = self.read_hdf5(filename)
        else:
            raise RuntimeError((f"Input file of type {ext} is not supported.\n"
                                "Only .txt and .npz are supported."))

        if self.verbose > 0:
            print(f"Reading finished in {str(timedelta(seconds=timer() - tr))}.")

        if prepare:
            return self.prepare_input(el_src, el_dst)
        else:
            return el_src, el_dst, np.unique(el_src, return_index=False, return_counts=False)

    def write_txt(self, filename, *args, mtx_header=False):

        if len(args) == 0:
            return

        v_len = len(args[0])
        for arg in args:
            if len(arg) != v_len:
                raise RuntimeError(f"Provided vectors length for writing should be equal: {len(arg)} != {v_len}.")

        if self.verbose > 0:
            print(f"Writing vectors to {filename} ...")
            tr = timer()

        with open(filename, 'w') as f:
            if mtx_header:
                # header section
                f.write("%%MatrixMarket matrix coordinate pattern general integer\n")
                for arg in args:
                    f.write(f"{np.max(arg)}\t")
                f.write(f"{v_len}\n")

            # body section
            for i in range(v_len):
                for arg in args:
                    f.write(f"{arg[i]}\t")
                f.write("\n")

        if self.verbose > 0:
            print(f"Writing finished in {str(timedelta(seconds=timer() - tr))}.")

        return

    def write_npz(self, filename, **kwargs):

        if self.verbose > 0:
            print(f"Writing zip NUMPY arrays to {filename} ...")
            tr = timer()

        np.savez(filename, **kwargs)

        if self.verbose > 0:
            print(f"Writing finished in {str(timedelta(seconds=timer() - tr))}.")

        return

    def write_hdf5(self, filename, **kwargs):

        if self.verbose > 0:
            print(f"Writing h5 NUMPY arrays to {filename} ...")
            tr = timer()

        with h5py.File(filename, 'w') as hf:
            for key, val in kwargs.items():
                hf.create_dataset(f"/{key}", data=val)

        if self.verbose > 0:
            print(f"Writing finished in {str(timedelta(seconds=timer() - tr))}.")

        return

    @staticmethod
    def serialized_filename(filename, serialize):

        base_path, _ = os.path.splitext(filename)

        if serialize == "txt":
            return base_path + ".txt"
        elif serialize == "mtx":
            return base_path + ".mtx"
        elif serialize == "npz":
            return base_path + ".npz"
        elif serialize == "h5" or serialize == "hdf" or serialize == "hdf5":
            return base_path + ".h5"

    def write(self, filename, src, dst):

        _, ext = os.path.splitext(filename)

        if ext == ".txt":
            self.write_txt(filename, src, dst, mtx_header=False)
        elif ext == ".mtx":
            self.write_txt(filename, src, dst, mtx_header=True)
        elif ext == ".npz":
            self.write_npz(filename, src=src, dst=dst)
        elif ext == ".h5" or ext == ".hdf" or ext == ".hdf5":
            self.write_hdf5(filename, src=src, dst=dst)

    @staticmethod
    def reconstruct_original_edge_list(n, d, i):
        """
        For testing purposes reconstruct the whole sorted edge list
        :param n: nodes id vector
        :param d: nodes degree vector
        :param i: nodes neighbor list vector
        :return: sorted edge list
        """
        el_src = np.repeat(np.arange(len(d)).astype(d.dtype), d)  # edge list source

        # map numbers to original ones
        el_src = n[el_src]
        el_dst = n[i]

        # sort according to nodes ID
        sort_perm = np.lexsort((el_dst, el_src))
        el_src = el_src[sort_perm]
        el_dst = el_dst[sort_perm]

        return GraphIO.merge(el_src, el_dst)

    @staticmethod
    def write_csv(filename, record, append=True):
        """
        append CSV record
        :param filename: The given csv filename
        :param record: A dictionary of key and values
        :param append: Append or create new
        """
        if append and os.path.isfile(filename):
            with open(filename, 'r+') as f:
                header = next(csv.reader(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC))
                dict_writer = csv.DictWriter(f, fieldnames=header, restval=-1,
                                             delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                dict_writer.writerow(record)
        else:
            with open(filename, 'w') as f:
                dict_writer = csv.DictWriter(f, fieldnames=list(record.keys()), restval=-1,
                                             delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                dict_writer.writeheader()
                dict_writer.writerow(record)

    @staticmethod
    def linux_cpu_info():
        if os.path.isfile("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r") as f:
                cpu_info = {f[0]: (f[1] if len(f) > 1 else None) for f in
                            [[' '.join(r.strip().split()) for r in ln.split(":")] for ln in f.readlines()]}
            return cpu_info['model name'], int(cpu_info['cpu cores'])
        return "NA", -1
