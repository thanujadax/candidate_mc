#!/usr/bin/python

from scipy import ndimage as ndi
import mahotas as mh
import numpy as np
import glob
import os

# parameter taken from grid-search on sample_A
sigma_seeds = 5
sigma_watersheds = 1
ms = 11

files = glob.glob("../data/sample_A/crop/membrane_inv/training_small/*")
files.sort()

#labels = 0 
if not os.path.isdir("../data/sample_A/crop/fragment/small_w1ms3/"):
    os.mkdir("../data/sample_A/crop/fragment/small_w1ms3/")

for i in range(len(files)):
    print("Processing " + files[i])

    membrane = mh.imread(files[i])
#    membrane_seeds = mh.gaussian_filter(membrane, sigma_seeds)/255.0
    membrane_watersheds = mh.gaussian_filter(membrane, sigma_watersheds)/255.0
#    mh.imsave("fragment/smoothed_seeds.tif", membrane_seeds.astype(float))
#    mh.imsave("fragment/smoothed_watersheds.tif", membrane_watersheds.astype(float))

    maxima = mh.regmin(membrane_watersheds, np.ones((ms,ms)))

    (seeds, num_seeds) = ndi.label(maxima, structure=np.ones((3,3)))
#    mh.imsave("fragment/seeds.tif", seeds.astype(float))

    print("Found " + str(num_seeds) + " seeds")
    labels = mh.cwatershed(membrane_watersheds, seeds)
    #labels += label_offset
    #label_offset += num_seeds

    mh.imsave("../data/sample_A/crop/fragment/small_w1ms3/" + str(i).zfill(5) + ".tif", labels.astype(float))

    #mh.imsave("superpixels.tif", labels.astype(float))
