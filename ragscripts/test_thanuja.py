from pycmc import *
from create_single_mergetree import fuse
from subprocess import call, Popen, PIPE
from add_rf_feature import add_rf_feature
import glob
import sys
import os

def tee(cmd, log_file = None):

    if log_file is None:

        call(cmd)

    else:

        proc = Popen(cmd, stdout=PIPE, stderr=PIPE)

        with open(log_file, "w") as f:
            while proc.poll() is None:
                line = proc.stdout.readline()
                while line:
                    print line.strip()
                    f.write(line)
                    line = proc.stdout.readline()
                line = proc.stderr.readline()
                while line:
                    print line.strip()
                    f.write(line)
                    line = proc.stderr.readline()

def test():

    if not os.path.exists("tif"):
        os.mkdir("tif")
    if not os.path.exists("log"):
        os.mkdir("log")
    if not os.path.exists("hdf"):
        os.mkdir("hdf")

    ## create project
    tee([
        "cmc_create_project",
        "--forceParentCandidate=false",
        "--supervoxels=/home/thanuja/DATA/ISBI2012/test/fragments_rfc",
        "--mergeHistory=/home/thanuja/DATA/ISBI2012/test/mergetree_rfc",
        "--intensities=/home/thanuja/DATA/ISBI2012/test/raw",
        "--boundaries=/home/thanuja/DATA/ISBI2012/test/mem_inv_rfc",
        "--importTrainingResult=/home/vleite/PhD/research/thanuja-data/trainning/rfc.hdf",
        "--2dSupervoxels=true",
        "--resX=4",
        "--resY=4",
        "--resZ=40",
        "--cragType=empty",
        "--maxZLinkBoundingBoxDistance=200",
        "-p", "/home/thanuja/DATA/ISBI2012/test/rfc.hdf"
    ], "log/create_project_test.log")

    ## extract features
    tee([
        "cmc_extract_features",
        "--forceParentCandidate=false",
        "--noVolumeRays=true",
        "--noSkeletons=true",
        "--minMaxFromProject=true",
        "--normalize=true",
        "--boundariesFeatures=true",
        "--boundariesBoundaryFeatures=true",
        "--noCoordinatesStatistics=true",
        "-p", "/home/thanuja/DATA/ISBI2012/test/rfc.hdf"
    ], "log/extract_features_test.log")

    add_rf_feature("/home/vleite/PhD/research/thanuja-data/trainning/rfc.hdf", "/home/thanuja/DATA/ISBI2012/test/rfc.hdf", 0, 0)

    candidate_bias = 0
    merge_bias = 0

    # create solution
    tee([
        "cmc_solve",
        "-p", "/home/thanuja/DATA/ISBI2012/test/rfc.hdf",
        "--exportSolution=/home/thanuja/DATA/ISBI2012/test/tif/rfc",
        "--foregroundBias=" + str(candidate_bias),
        "--mergeBias=" + str(merge_bias),
    ], "log/solve_test.log")

    # evaluate against groundtruth
#    tee([
#            "./evaluate.sh"
#        ])


if __name__ == "__main__":

        test()
