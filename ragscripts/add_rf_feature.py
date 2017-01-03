#!/usr/bin/python

# adds a RF "feature" to the given dataset (which is assumed to not have the 
# feature yet)

from pycmc import *
import math
import sys

def probToEnergy(prob):

    # Get the probability of being switched on (i.e., label 1). In a simple 
    # probabilistic model, this is
    #
    #   p(y_n==1) = 1/Z exp(-E_n(1))
    #
    # where Z = exp(-E_n(0)) + exp(-E_n(1)) and E_n is an energy.
    #
    #   exp(-E_n(1)) = p(y_n==1)*( exp(-E_n(0)) + exp(-E_n(1)) )
    #   exp(-E_n(1)) = p(y_n==1)*exp(-E_n(0)) + p(y_n==1)*exp(-E_n(1))
    #   exp(-E_n(1))*(1 - p(y_n==1)) = p(y_n==1)*exp(-E_n(0))
    #   exp(-E_n(1) + E_n(0))(1 - p(y_n==1)) = p(y_n==1)
    #   exp(-E_n(1) + E_n(0)) = p(y_n==1)/p(y_n==0)
    #   -E_n(1) + E_n(0) = log(p(y_n==1)/p(y_n==0))
    #   -E_n(1) + E_n(0) = log(p(y_n==1)) - log(p(y_n==0))
    #
    # without loss of generality, we can set E_n(0) = 0
    #
    #   E_n(1) = log(p(y_n==0)) - log(p(y_n==1))
    #
    # this energy is negative, if p(y_n==1) > 0.5

    # ensure numerical stability
    prob = max(0.001, min(0.999, prob))

    return math.log(1.0-prob) - math.log(prob);

def add_rf_feature(rf_filename, project_filename, higher_node_bias, higher_edge_bias):

    crag = Crag()
    nodeFeatures = NodeFeatures(crag)
    edgeFeatures = EdgeFeatures(crag)

    cragStore = Hdf5CragStore(project_filename)
    cragStore.retrieveCrag(crag)
    cragStore.retrieveNodeFeatures(crag, nodeFeatures)
    cragStore.retrieveEdgeFeatures(crag, edgeFeatures)

    print "Read slice node features of dim " + str(nodeFeatures.dims(CragNodeType.SliceNode))
    print "Read assignment node features of dim " + str(nodeFeatures.dims(CragNodeType.AssignmentNode))
    print "Read edge features of dim " + str(edgeFeatures.dims(CragEdgeType.NoAssignmentEdge))

    sliceNodeRandomForest = RandomForest()
    assNodeRandomForest = RandomForest()
    edgeRandomForest = RandomForest()

    sliceNodeRandomForest.read(rf_filename, "classifiers/slice_node_rf");
    assNodeRandomForest.read(rf_filename, "classifiers/assignment_node_rf");
    edgeRandomForest.read(rf_filename, "classifiers/no_assignment_edge_rf");

    print "Adding RF \"feature\"..."

    node_energies = {}
    edge_energies = {}

    # get the energies for each edge and node
    for n in crag.nodes():
        prob = 0
        if crag.type(n) == CragNodeType.SliceNode:
            prob = sliceNodeRandomForest.getProbabilities(nodeFeatures[n])[1]
        elif crag.type(n) == CragNodeType.AssignmentNode:
            prob = assNodeRandomForest.getProbabilities(nodeFeatures[n])[1]
        else:
            continue
        node_energies[crag.id(n)] = probToEnergy(prob)
    for e in crag.edges():
        if crag.type(e) == CragEdgeType.NoAssignmentEdge:
            # all samples in the training dataset belonged to one class
            #prob = edgeRandomForest.getProbabilities(edgeFeatures[e])[1]
            #edge_energies[crag.id(e)] = probToEnergy(prob)
            edge_energies[crag.id(e)] = 0

    # multiply the scores of the higher candidates with their "weight", i.e., the 
    # number of leaf decisions they imply

    # map from leaf nodes to their parents
    parents = {}
    for n in crag.nodes():
        if crag.isRootNode(n):
            for arc in crag.inArcs(n):
                child = arc.source()
                parents[crag.id(child)] = crag.id(n)

    # number of leaf nodes and edges under a higher node or edge
    node_num_leaf_nodes = {}
    node_num_leaf_edges = {}
    edge_num_leaf_edges = {}

    # count number of leaf nodes and edges under higher node
    for n in crag.nodes():
        node_num_leaf_nodes[crag.id(n)] = len(crag.leafNodes(n))
        node_num_leaf_edges[crag.id(n)] = len(crag.leafEdges(n))
    # count number of leaf edges under higher edge
    for e in crag.edges():
        edge_num_leaf_edges[crag.id(e)] = len(crag.leafEdges(e))

    for n in crag.nodes():
        if not crag.isLeafNode(n) and (crag.type(n) == CragNodeType.SliceNode or crag.type(n) == CragNodeType.AssignmentNode):
            nln = node_num_leaf_nodes[crag.id(n)]
            nle = node_num_leaf_edges[crag.id(n)]
            node_energies[crag.id(n)] *= nln + nle
            node_energies[crag.id(n)] += higher_node_bias
    for e in crag.edges():
        if not crag.isLeafEdge(e) and crag.type(e) == CragEdgeType.NoAssignmentEdge:
            edge_energies[crag.id(e)] *= edge_num_leaf_edges[crag.id(e)]
            edge_energies[crag.id(e)] += higher_edge_bias

    # DEBUG "FEATURES" ########################
    #for n in crag.nodes():
        #if not crag.isLeafNode(n):
            #nodeFeatures.append(n, node_num_leaf_nodes[crag.id(n)]);
            #nodeFeatures.append(n, node_num_leaf_edges[crag.id(n)]);
        #else:
            #nodeFeatures.append(n, 0);
            #nodeFeatures.append(n, 0);
    #for e in crag.edges():
        #if not crag.isLeafEdge(e):
            #edgeFeatures.append(e, edge_num_leaf_edges[crag.id(e)]);
        #else:
            #edgeFeatures.append(e, 0);
    ###########################################

    for n in crag.nodes():
        if crag.type(n) == CragNodeType.SliceNode or crag.type(n) == CragNodeType.AssignmentNode:
            nodeFeatures.append(n, node_energies[crag.id(n)]);
    for e in crag.edges():
        if crag.type(e) == CragEdgeType.NoAssignmentEdge:
            edgeFeatures.append(e, edge_energies[crag.id(e)]);

    cragStore.saveNodeFeatures(crag, nodeFeatures);
    cragStore.saveEdgeFeatures(crag, edgeFeatures);

    print "Updating feature weights"

    # set all feature weights to zero, except the RF feature
    weights = FeatureWeights(nodeFeatures, edgeFeatures, 0)
    weights[CragNodeType.SliceNode][-1] = 1
    weights[CragNodeType.AssignmentNode][-1] = 1
    weights[CragEdgeType.NoAssignmentEdge][-1] = 1
    cragStore.saveFeatureWeights(weights);

    cragStore = None
