import os
from subprocess import call, Popen, PIPE
from train_random_forest import train_rf
import sys
sys.path.append('..')
from add_rf_feature import add_rf_feature
import datetime

def tee(cmd, log_file = None):

    full_cmd = ""
    for i in cmd:
        full_cmd += i + " "
    print("running: " + full_cmd)

    if log_file is None:

        call(cmd)

    else:

        proc = Popen(cmd, stdout=PIPE, stderr=PIPE)

        with open(log_file, "w") as f:
            while proc.poll() is None:
                line = proc.stdout.readline()
                while line:
                    print (line.strip())
                    f.write(line)
                    line = proc.stdout.readline()
                line = proc.stderr.readline()
                while line:
                    print (line.strip())
                    f.write(line)
                    line = proc.stderr.readline()

if __name__ == "__main__":

    if not os.path.exists("tif"):
        os.mkdir("tif")
    if not os.path.exists("log"):
        os.mkdir("log")
    if not os.path.exists("hdf"):
        os.mkdir("hdf")

    sample = 'A'
    projectName = 'train_rf'

    ts = datetime.datetime.now()
     tee([
         "cmc_create_project",
         "--mergeHistoryWithScores",
         "--maxMerges=8",
         "--2dSupervoxels",
         "--resX=4",
         "--resY=4",
         "--resZ=40",
         "--cragType=empty",
         "--maxZLinkBoundingBoxDistance=400",
         "--supervoxels=/home/thanuja/Dropbox/data/multicut/train/fragments_rfc" + sample,
         "--mergeHistory=/home/thanuja/Dropbox/data/multicut/train/mergetree_rfc" + sample,
         "--groundTruth=/home/thanuja/Dropbox/data/multicut/train/groundTruthIdx" + sample,
         "--intensities=/home/thanuja/Dropbox/data/multicut/train/raw" + sample,
         "--boundaries=../01_data_preparation/train_rf/membrane" + sample,
         "--xAffinities=../01_data_preparation/train_rf/affinities" + sample + "/aff_x",
         "--yAffinities=../01_data_preparation/train_rf/affinities" + sample + "/aff_y",
         "--zAffinities=../01_data_preparation/train_rf/affinities" + sample + "/aff_z",
         "--project=hdf/" + projectName + ".hdf"
     ], "log/create_project_" + projectName + ".log")
  
    # extract features
    tee([
        "cmc_extract_features",
        "--log-level=debug",
        "--project=hdf/" + projectName + ".hdf",
        "--statisticsFeatures=true",
        "--topologicalFeatures=true",
        "--assignmentFeatures=true",
        "--normalize=true",
        "--dumpFeatureNames=" + projectName
    ], "log/extract_features_" + projectName + ".log")
  
    # create best-effort - without dryRun: train SSVM
    tee([
        "cmc_train",
        "--log-level=debug",
        "--project=hdf/" + projectName + ".hdf",
        "--assignmentSolver",
        "--dryRun",
        "--bestEffortLoss=newLoss",
        "--exportBestEffort=tif/best-effort" + projectName
    ], "log/best_effort_" + projectName + ".log")
 
    # train random forest
    train_rf( "hdf/" + projectName + ".hdf")

    tf = datetime.datetime.now()
    print ("time: ")
    print (tf-ts)
