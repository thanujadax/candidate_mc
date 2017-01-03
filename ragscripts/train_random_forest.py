#!/usr/bin/python

from pycmc import *
from time import time
import sys

def train_rf(project_file):

    print ("Training vigra RF classifier")

    print ("Reading CRAG and features")
    crag = Crag()
    nodeFeatures = NodeFeatures(crag)
    edgeFeatures = EdgeFeatures(crag)
    bestEffort   = CragSolution(crag)

    cragStore = Hdf5CragStore(project_file)
    cragStore.retrieveCrag(crag)
    cragStore.retrieveNodeFeatures(crag, nodeFeatures)
    cragStore.retrieveEdgeFeatures(crag, edgeFeatures)
    cragStore.retrieveSolution(crag, bestEffort, "best-effort")

    sliceNodeRandomForest = RandomForest()
    assNodeRandomForest = RandomForest()
    edgeRandomForest = RandomForest()

    print ("Preparing training for:")
    print ("\t" + "slice nodes with " + str(nodeFeatures.dims(CragNodeType.SliceNode))    + " features")
    print ("\t" + "assignment nodes with " + str(nodeFeatures.dims(CragNodeType.AssignmentNode))    + " features")
    print ("\t" + "no-assignment edges with " + str(edgeFeatures.dims(CragEdgeType.NoAssignmentEdge)) + " features")

    num_slice_nodes = 0
    num_ass_nodes = 0
    num_noass_edges = 0
    for n in crag.nodes():
        if crag.type(n) == CragNodeType.SliceNode:
            num_slice_nodes += 1
        if crag.type(n) == CragNodeType.AssignmentNode:
            num_ass_nodes += 1
    for e in crag.edges():
        if crag.type(e) == CragEdgeType.NoAssignmentEdge:
            num_noass_edges += 1

    sliceNodeRandomForest.prepareTraining(num_slice_nodes, nodeFeatures.dims(CragNodeType.SliceNode))
    assNodeRandomForest.prepareTraining(num_ass_nodes, nodeFeatures.dims(CragNodeType.AssignmentNode))
    edgeRandomForest.prepareTraining(num_noass_edges, edgeFeatures.dims(CragEdgeType.NoAssignmentEdge))

    for n in crag.nodes():
        if crag.type(n) == CragNodeType.SliceNode:
            sliceNodeRandomForest.addSample(nodeFeatures[n], bestEffort.selected(n))
        if crag.type(n) == CragNodeType.AssignmentNode:
            assNodeRandomForest.addSample(nodeFeatures[n], bestEffort.selected(n))
    for e in crag.edges():
        if crag.type(e) == CragEdgeType.NoAssignmentEdge:
            edgeRandomForest.addSample(edgeFeatures[e], bestEffort.selected(e))


    # train with auto-selection of trees and features per node

    t = time()
    print ("Training slice node RF on " + str(num_slice_nodes) + " samples...")
    sliceNodeRandomForest.train(0, 0, False)
    oob = sliceNodeRandomForest.getOutOfBagError()
    print ("Finished with OOB = " + str(oob) + " in " + str(time() - t) + " seconds")

    t = time()
    print ("Training assignment node RF on " + str(num_ass_nodes) + " samples...")
    assNodeRandomForest.train(0, 0, False)
    oob = assNodeRandomForest.getOutOfBagError()
    print ("Finished with OOB = " + str(oob) + " in " + str(time() - t) + " seconds")

    t = time()
    print ("Training edge RF on " + str(num_noass_edges) + " samples...")
    edgeRandomForest.train(0, 0, False)
    oob = edgeRandomForest.getOutOfBagError()
    print ("Finished with OOB = " + str(oob) + " in " + str(time() - t) + " seconds")

    print ("Training finished, storing RFs")

    sliceNodeRandomForest.write(project_file, "classifiers/slice_node_rf");
    assNodeRandomForest.write(project_file, "classifiers/assignment_node_rf");
    edgeRandomForest.write(project_file, "classifiers/no_assignment_edge_rf");

    print ("Done")

if __name__ == "__main__":
	#train random forest
	train_rf("hdf/new_AS_project.hdf")

