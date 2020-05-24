"""
Created on May 25, 2019

@author: A. Mehrafsa

Modules: matplotlib networkx
"""

from kcd.kcd import KCD
import argparse
import traceback

# for drawing the graph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from os.path import splitext
from sys import exit

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("input_filename", metavar="input-filename", type=str, help=("Edge list text filename for"
                                                                                    " graph sorted by nodes id (.txt)"
                                                                                    " or pre-processed Numpy arrays"
                                                                                    " for graphs(.npz).\n"
                                                                                    "Shared links from google drive"
                                                                                    " could also be used.\n"
                                                                                    "If a .zip file with one archive"
                                                                                    " file provided the"
                                                                                    " extracted file will be used."))

    parser.add_argument("--skip-prepare", action='store_true', help=("Do not preform pre processing for removing"
                                                                     " self loop edges and converting directed to "
                                                                     " undirected (Assume that this is already ."
                                                                     " done for graph."))

    parser.add_argument("-o", "--output-filename", dest="output_filename", type=str,
                        help="Core results output filename.")

    parser.add_argument("-p", "--profiler-filename", type=str, help="Filename for profiler output.")

    parser.add_argument("--profiler-new", action='store_true', help=("Create new profiler if the profiler file"
                                                                     " exist (Does not append information to"
                                                                     " old file)."))

    parser.add_argument("-c", "--profiler-comment", dest="profiler_comment", type=str,
                        help="Comments in the profiler file for this experiment record.")
    parser.set_defaults(profiler_comment="Default")

    parser.add_argument("-v", "--verbose", dest='verbose', type=int, choices=range(0, 3),
                        help="Program in verbosity level.")
    parser.set_defaults(verbose=2)

    parser.add_argument("--serialize", choices=['none', 'hdf5', 'npz', 'mtx', 'txt'],
                        type=str.lower,
                        help="Perform choice to serialize the file in case the input file was a text.")

    parser.set_defaults(serialize="none")

    parser.add_argument("--serialize-prepared", action='store_true', help=("Serialize the prepared graph not the"
                                                                           " original graph (remove self-loops/add "
                                                                           "reverse edges and remap the nodes id."))

    parser.add_argument("--no-kcd", action='store_true', help=("Do not run k-core decomposition (e.g. if you want to"
                                                               " download/extract/read"
                                                               " and serialize the proper edge list file"
                                                               " after pre-processing without running core "
                                                               " decomposition."))

    parser.add_argument("-m", "--mode", choices=['numpy', 'torch-cpu', 'torch-gpu'],
                        type=str.lower,
                        help="Perform choice to run the main algorithm on Numpy/Torch(CPU/GPU).")

    parser.set_defaults(mode="numpy")

    parser.add_argument('--draw', dest='draw', action="store_true", help=("Draw the graph to a file in the same "
                                                                          "path as input with extension .png."))

    try:

        args = parser.parse_args()

        kcd = KCD(filename=args.input_filename,
                  skip_prepare=args.skip_prepare,
                  mode=args.mode,
                  serialize=args.serialize,
                  serialize_prepared=args.serialize_prepared,
                  comment=args.profiler_comment,
                  verbose=args.verbose)

        if args.draw:
            print("Drawing the graph file ...")
            g = nx.MultiDiGraph()
            g.add_edges_from(np.loadtxt(args.input_filename, dtype=np.long, comments='#'))
            plt.figure(figsize=(8, 8))
            nx.draw(g, with_labels=True)
            base_path, _ = splitext(args.input_filename)
            plt.savefig(base_path)
            print("Drawing finished.")

        if args.no_kcd:
            print("The program is configured to finish without running the algorithm.")
            exit(0)

        kcd.run()

        if args.output_filename is not None:
            kcd.write_results(args.output_filename)

        if args.profiler_filename is not None:
            kcd.write_profile(args.profiler_filename, not args.profiler_new)

    except RuntimeError as e:
        print(f"Error: {str(e)}")
        print("_" * 20, "TRACEBACK", "_" * 20)
        print(traceback.format_exc())
