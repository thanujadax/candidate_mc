/**
 * Reads a treemc project file containing features and solves the segmentation 
 * problem for a given set of feature weights.
 */

#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

#include <util/Logger.h>
#include <util/ProgramOptions.h>
#include <util/exceptions.h>
#include <util/helpers.hpp>
#include <util/timing.h>
#include <io/CragImport.h>
#include <io/Hdf5CragStore.h>
#include <io/vectors.h>
#include <io/volumes.h>
#include <crag/DownSampler.h>
#include <crag/PlanarAdjacencyAnnotator.h>
#include <features/FeatureExtractor.h>
#include <inference/MultiCut.h>

util::ProgramOption optionFeatureWeights(
		util::_long_name        = "featureWeights",
		util::_short_name       = "w",
		util::_description_text = "A file containing feature weights.",
		util::_default_value    = "feature_weights.txt");

util::ProgramOption optionForegroundBias(
		util::_long_name        = "foregroundBias",
		util::_short_name       = "f",
		util::_description_text = "A bias to be added to each node weight.",
		util::_default_value    = 0);

util::ProgramOption optionMergeBias(
		util::_long_name        = "mergeBias",
		util::_short_name       = "b",
		util::_description_text = "A bias to be added to each edge weight.",
		util::_default_value    = 0);

util::ProgramOption optionProjectFile(
		util::_long_name        = "projectFile",
		util::_short_name       = "p",
		util::_description_text = "The treemc project file.");

util::ProgramOption optionMergeTree(
		util::_long_name        = "mergeTree",
		util::_short_name       = "m",
		util::_description_text = "The merge-tree image. If this is a directory, one mergtree will be extracted "
		                          "per image in the directory and adjacencies introduced across subsequent images.",
		util::_default_value    = "merge_tree.tif");

util::ProgramOption optionIntensities(
		util::_long_name        = "intensities",
		util::_short_name       = "i",
		util::_description_text = "The raw intensity image or directory of images.",
		util::_default_value    = "raw.tif");

util::ProgramOption optionBoundaries(
		util::_long_name        = "boundaries",
		util::_short_name       = "b",
		util::_description_text = "The boundary prediciton image or directory of images.",
		util::_default_value    = "prob.tif");

util::ProgramOption optionResX(
		util::_long_name        = "resX",
		util::_description_text = "The x resolution of one pixel in the input images.",
		util::_default_value    = 1);

util::ProgramOption optionResY(
		util::_long_name        = "resY",
		util::_description_text = "The y resolution of one pixel in the input images.",
		util::_default_value    = 1);

util::ProgramOption optionResZ(
		util::_long_name        = "resZ",
		util::_description_text = "The z resolution of one pixel in the input images.",
		util::_default_value    = 1);

util::ProgramOption optionOffsetX(
		util::_long_name        = "offsetX",
		util::_description_text = "The x offset of the input images.",
		util::_default_value    = 0);

util::ProgramOption optionOffsetY(
		util::_long_name        = "offsetY",
		util::_description_text = "The y offset of the input images.",
		util::_default_value    = 0);

util::ProgramOption optionOffsetZ(
		util::_long_name        = "offsetZ",
		util::_description_text = "The z offset of the input images.",
		util::_default_value    = 0);

util::ProgramOption optionDownsampleCrag(
		util::_long_name        = "downSampleCrag",
		util::_description_text = "Reduce the number of candidates in the CRAG by removing candidates smaller than minCandidateSize, "
		                          "followed by contraction of single children with their parents.");

util::ProgramOption optionMinCandidateSize(
		util::_long_name        = "minCandidateSize",
		util::_description_text = "The minimal size for a candidate to keep it during downsampling (see downSampleCrag).",
		util::_default_value    = 100);

int main(int argc, char** argv) {

	try {

		util::ProgramOptions::init(argc, argv);
		logger::LogManager::init();

		Crag* crag = new Crag();
		CragVolumes* volumes = new CragVolumes(*crag);

		NodeFeatures nodeFeatures(*crag);
		EdgeFeatures edgeFeatures(*crag);

		if (optionProjectFile) {

			Hdf5CragStore cragStore(optionProjectFile.as<std::string>());
			cragStore.retrieveCrag(*crag);
			cragStore.retrieveVolumes(*volumes);

			cragStore.retrieveNodeFeatures(*crag, nodeFeatures);
			cragStore.retrieveEdgeFeatures(*crag, edgeFeatures);

		} else {

			util::point<float, 3> resolution(
					optionResX,
					optionResY,
					optionResZ);
			util::point<float, 3> offset(
					optionOffsetX,
					optionOffsetY,
					optionOffsetZ);

			std::string mergeTreePath = optionMergeTree;

			CragImport import;
			import.readCrag(mergeTreePath, *crag, *volumes, resolution, offset);

			if (optionDownsampleCrag) {

				UTIL_TIME_SCOPE("downsample CRAG");

				Crag* downSampled = new Crag();
				CragVolumes* downSampledVolumes = new CragVolumes(*downSampled);

				DownSampler downSampler(optionMinCandidateSize.as<int>());
				downSampler.process(*crag, *volumes, *downSampled, *downSampledVolumes);

				delete crag;
				delete volumes;
				crag = downSampled;
				volumes = downSampledVolumes;
			}

			{
				UTIL_TIME_SCOPE("find CRAG adjacencies");

				PlanarAdjacencyAnnotator annotator(PlanarAdjacencyAnnotator::Direct);
				annotator.annotate(*crag, *volumes);
			}

			ExplicitVolume<float> intensities = readVolume<float>(getImageFiles(optionIntensities));
			intensities.setResolution(resolution);
			intensities.setOffset(offset);
			intensities.normalize();

			ExplicitVolume<float> boundaries = readVolume<float>(getImageFiles(optionBoundaries));
			boundaries.setResolution(resolution);
			boundaries.setOffset(offset);
			boundaries.normalize();

			FeatureExtractor featureExtractor(*crag, *volumes, intensities, boundaries);
			featureExtractor.extract(nodeFeatures, edgeFeatures);
		}

		int numNodes = 0;
		int numRootNodes = 0;
		double sumSubsetDepth = 0;
		int maxSubsetDepth = 0;
		int minSubsetDepth = 1e6;

		for (Crag::NodeIt n(*crag); n != lemon::INVALID; ++n) {

			if (crag->isRootNode(n)) {

				int depth = crag->getLevel(n);

				sumSubsetDepth += depth;
				minSubsetDepth = std::min(minSubsetDepth, depth);
				maxSubsetDepth = std::max(maxSubsetDepth, depth);
				numRootNodes++;
			}

			numNodes++;
		}

		int numAdjEdges = 0;
		for (Crag::EdgeIt e(*crag); e != lemon::INVALID; ++e)
			numAdjEdges++;
		int numSubEdges = 0;
		for (Crag::SubsetArcIt e(*crag); e != lemon::INVALID; ++e)
			numSubEdges++;

		LOG_USER(logger::out) << "created CRAG" << std::endl;
		LOG_USER(logger::out) << "\t# nodes          : " << numNodes << std::endl;
		LOG_USER(logger::out) << "\t# root nodes     : " << numRootNodes << std::endl;
		LOG_USER(logger::out) << "\t# adjacencies    : " << numAdjEdges << std::endl;
		LOG_USER(logger::out) << "\t# subset edges   : " << numSubEdges << std::endl;
		LOG_USER(logger::out) << "\tmax subset depth : " << maxSubsetDepth << std::endl;
		LOG_USER(logger::out) << "\tmin subset depth : " << minSubsetDepth << std::endl;
		LOG_USER(logger::out) << "\tmean subset depth: " << sumSubsetDepth/numRootNodes << std::endl;

		std::vector<double> weights = retrieveVector<double>(optionFeatureWeights);

		Costs costs(*crag);

		float edgeBias = optionMergeBias;
		float nodeBias = optionForegroundBias;

		unsigned int numNodeFeatures = nodeFeatures.dims();
		unsigned int numEdgeFeatures = edgeFeatures.dims();

		for (Crag::NodeIt n(*crag); n != lemon::INVALID; ++n) {

			costs.node[n] = nodeBias;

			for (unsigned int i = 0; i < numNodeFeatures; i++)
				costs.node[n] += weights[i]*nodeFeatures[n][i];
		}

		for (Crag::EdgeIt e(*crag); e != lemon::INVALID; ++e) {

			costs.edge[e] = edgeBias;

			for (unsigned int i = 0; i < numEdgeFeatures; i++)
				costs.edge[e] += weights[i + numNodeFeatures]*edgeFeatures[e][i];
		}

		MultiCut multicut(*crag);

		multicut.setCosts(costs);
		{
			UTIL_TIME_SCOPE("solve candidate multi-cut");
			multicut.solve();
		}
		multicut.storeSolution(*volumes, "solution.tif");
		multicut.storeSolution(*volumes, "solution_boundary.tif", true);

		delete crag;
		delete volumes;

	} catch (Exception& e) {

		handleException(e, std::cerr);
	}
}



