# VETGA: Vectorized Toolkit for Graph Analytics

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)


Python implementation of vectorized [K-core decompositio](https://en.wikipedia.org/wiki/Degeneracy_(graph_theory)) using [Numpy](https://github.com/numpy/numpy) and [pyTorch](https://github.com/pytorch/pytorch) libraries.


## Required softwares
Main requirements are:
* Python 3.8.3 or later
* Numpy 1.18.2 or later

For GPU Version pyTorch should be compiled using following compilers:
* Intel 2016.4
* Torch 20171030
* Cuda 8.0.44 (For GPU Version)
* Python 3.7.0
* Numpy 1.17

Other 3rd party libraries used in utility class are:
* networkx
* matplotlib
* tqdm
* requests
* h5py

## Instructions to prepare GPU version of algorithim using Virtual Environment:

* `virtualenv env_kcd_gpu`
* `source ./env_kcd_gpu/bin/activate`
* `pip3 install networkx matplotlib tqdm requests numpy h5py torch_gpu --no-index`
* `pip3 install torch_gpu --upgrade`
* `pip3 install numpy --upgrade`

## Dataset files

Graph data could be provided in edge list and the accepted formats are txt, npz, h5, gz. The algorithm is capable of reading files from a local disk or Google drive.

Text files are the most common used format but very slow to read. It is recommended to use HDF5 as they are the best way to store and read datasets fast and there is no limitation for size (There is a 2GB limitation for npz files).

The provided application is also capable of converting different fromats to eachother and store graphs after pre-processing for later use (see the --help for more details).

Some sample graphs could be found here:
<http://webgraph.di.unimi.it/>

## Running the algorithim

* Run the algorithm
	* Run the algorithm using a hdf5 file as an input, will print average K, max K and time spend.
		* `python3 kcdapp.py ./LiveJournal.h5`
	* Run the algorithm using a hdf5 file as an input, will also create a file that has coreness of nodes and a detail of profling the algorithm in a csv file.
		* `python3 kcdapp.py ./amazon.h5 -o=amazon-cores.h5 -v=2 -m=numpy -p=stat.csv`
	* Run the algorithim using a text input file
	* `python3 kcdapp.py ./LiveJournal.txt`
	* Run the progrma with npz file as an input
		* `python3 kcdapp.py "./data9_soc-LiveJournal.npz" "./data9_soc-LiveJournal-cores.txt" --mode numpy`
		* `python3 kcdapp.py data9_soc-LiveJournal.npz data9_soc-LiveJournal-cores.txt --mode torch-gpu`

## Testing the algorithm correctness

Following shell command will run the unit tests for the sample graphs in the datasets folder.

```shell
python3 tests.py
```

## Contributing

Contributions are welcome by submiting pull requests.

## Program usage guide
```
usage: kcdapp.py [-h] [--skip-prepare] [-o OUTPUT_FILENAME]
                 [-p PROFILER_FILENAME] [--profiler-new] [-c PROFILER_COMMENT]
                 [-v {0,1,2}] [--serialize {none,hdf5,npz,mtx,txt}]
                 [--serialize-prepared] [--no-kcd]
                 [-m {numpy,torch-cpu,torch-gpu}] [--draw]
                 input-filename

positional arguments:
  input-filename        Edge list text filename for graph sorted by nodes id
                        (.txt) or pre-processed Numpy arrays for graphs(.npz).
                        Shared links from google drive could also be used. If
                        a .zip file with one archive file provided the
                        extracted file will be used.

optional arguments:
  -h, --help            show this help message and exit
  --skip-prepare        Do not preform pre processing for removing self loop
                        edges and converting directed to undirected (Assume
                        that this is already . done for graph.
  -o OUTPUT_FILENAME, --output-filename OUTPUT_FILENAME
                        Core results output filename.
  -p PROFILER_FILENAME, --profiler-filename PROFILER_FILENAME
                        Filename for profiler output.
  --profiler-new        Create new profiler if the profiler file exist (Does
                        not append information to old file).
  -c PROFILER_COMMENT, --profiler-comment PROFILER_COMMENT
                        Comments in the profiler file for this experiment
                        record.
  -v {0,1,2}, --verbose {0,1,2}
                        Program in verbosity level.
  --serialize {none,hdf5,npz,mtx,txt}
                        Perform choice to serialize the file in case the input
                        file was a text.
  --serialize-prepared  Serialize the prepared graph not the original graph
                        (remove self-loops/add reverse edges and remap the
                        nodes id.
  --no-kcd              Do not run k-core decomposition (e.g. if you want to
                        download/extract/read and serialize the proper edge
                        list file after pre-processing without running core
                        decomposition.
  -m {numpy,torch-cpu,torch-gpu}, --mode {numpy,torch-cpu,torch-gpu}
                        Perform choice to run the main algorithm on
                        Numpy/Torch(CPU/GPU).
  --draw                Draw the graph to a file in the same path as input
                        with extension .png.

```





