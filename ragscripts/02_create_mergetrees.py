#!/usr/bin/python

from glob import glob
import numpy
import os
import joblib
from subprocess import call

def merge_tree(sp_file, membrane_file, region_size_exponent):

#    basename = os.path.basename(membrane_file).strip(".tiff")

    call([
        "merge_tree",
#        "--minRegionSizeExponent=" + str(region_size_exponent),
#        "--dontConsiderRegionSize",
#         "--mergeSmallRegionsFirst",
        "--initialSuperpixels=" + sp_file,
        "-s", membrane_file,
        "--mergeHistory=/home/thanuja/Dropbox/data/multicut/train/mergetree_rfc/0000.txt",
    ])

if __name__ == "__main__":

#    if not os.path.isdir("mergetrees"):
#        os.mkdir("mergetrees")

    #region_size_exponents = numpy.arange(0, 1.1, 0.1)
    region_size_exponents = [ 1 ]

    #using membrane_inv so, the border is bright (higher values)
#    membrane_files = glob("/home/vleite/PhD/research/scripts/candidate_mc_scripts/data/training/membrane/*")
#    membrane_files.sort()
#
    jobs = []
#    for membrane_file in membrane_files:
#
#        basename = os.path.basename(membrane_file).strip(".tiff")
    sp_file = "/home/thanuja/Dropbox/data/multicut/train/fragments_rfc/00000.tif"

#    for region_size_exponent in region_size_exponents:
    jobs.append(joblib.delayed(merge_tree)(sp_file, "/home/thanuja/Dropbox/data/multicut/train/mem_inv_rfc/000.tif", 1))

    joblib.Parallel(n_jobs=20)(jobs)
