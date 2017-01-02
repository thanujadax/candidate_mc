#!/usr/bin/python

import os
from subprocess import call, Popen, PIPE
from train_random_forest import train_rf
import glob

def tee(cmd, log_file = None):

    full_cmd = ""
    for i in cmd:
        full_cmd += i + " "
    print "running: " + full_cmd

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

if __name__ == "__main__":

    if not os.path.exists("tif"):
        os.mkdir("tif")
    if not os.path.exists("log"):
        os.mkdir("log")
    if not os.path.exists("hdf"):
        os.mkdir("hdf")

    # create project
    tee([
        "cmc_create_project",
        "--forceParentCandidate=false",
        "--supervoxels=/home/thanuja/Dropbox/data/multicut/train/fragments_rfc",
        "--mergeHistory=/home/thanuja/Dropbox/data/multicut/train/mergetree_rfc",
        "--groundTruth=/home/thanuja/Dropbox/data/multicut/train/groundTruthIdx",
        "--intensities=/home/thanuja/Dropbox/data/multicut/train/raw",
        "--boundaries=/home/thanuja/Dropbox/data/multicut/train/mem_inv_rfc",
        "--2dSupervoxels=true",
        "--resX=4",
        "--resY=4",
        "--resZ=40",
        "--cragType=empty",
        "--maxZLinkBoundingBoxDistance=200",
        "-p", "/home/thanuja/Dropbox/data/multicut/train/projectFiles/rfc.hdf"
    ], "log/create_project.log")

    # extract features
    tee([
        "cmc_extract_features",
        "--forceParentCandidate=false",
        "--noVolumeRays=true",
        "--noSkeletons=true",
        "--normalize=true",
        "--boundariesFeatures=true",
        "--boundariesBoundaryFeatures=true",
        "--noCoordinatesStatistics=true",
        "-p", "/home/thanuja/Dropbox/data/multicut/train/projectFiles/rfc.hdf"
    ], "log/extract_features.log")

    # create best-effort
    tee([
        "cmc_train",
        "--forceParentCandidate=false",
        "-p", "/home/thanuja/Dropbox/data/multicut/train/projectFiles/rfc.hdf",
        "--dryRun",
        "--exportBestEffort=/home/thanuja/Dropbox/data/multicut/train/tif/rfc"
    ], "log/extract_best-effort.log")

    # train random forest
    train_rf("/home/thanuja/Dropbox/data/multicut/train/projectFiles/rfc.hdf")


