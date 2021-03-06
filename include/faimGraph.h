//------------------------------------------------------------------------------
// filename: faimGraph.h
// Author : @SH
// Description: the system to manager memory and update 
// Date: 2019.5.8
//
//------------------------------------------------------------------------------
//

#pragma once

#include "Utility.h"
#include "EdgeUpdate.h"
#include "VertexUpdate.h"
#include "MemoryManager.h"
#include "ConfigurationParser.h"
#include "GraphParser.h"

#define MEMORYOVERALLOCATION 1.0f

// Forward declaration
class GraphParser;
template <typename DataType>
class CSR;

class faimGraph
{
public:
	faimGraph(std::shared_ptr<Config> configuration, std::unique_ptr<GraphParser>& graph_parser) :
		config{ std::move(configuration) },
		memory_manager{ std::make_unique<MemoryManager>(static_cast<uint64_t>(GIGABYTE * config->device_mem_size_), config, graph_parser) },
		edge_update_manager{ std::make_unique<EdgeUpdateManager>() },
		vertex_update_manager{ std::make_unique<VertexUpdateManager<VertexData, VertexUpdate>>() } {}

	// Setup
	void initializeMemory(std::unique_ptr<GraphParser>& graph_parser);
	void initializeMemory(vertex_t* d_offset, vertex_t* d_adjacency, int number_vertices);

	// Reinitialize
	CSR<float> reinitializeFaimGraph(float overallocation_factor);

	// Updates
	void edgeInsertion();
	void edgeDeletion();
	void vertexInsertion(VertexMapper<index_t, index_t>& mapper);
	void vertexDeletion(VertexMapper<index_t, index_t>& mapper);

	// Verification
	std::unique_ptr<aimGraphCSR> verifyGraphStructure(std::unique_ptr<MemoryManager>& memory_manager);
	bool compareGraphs(std::unique_ptr<GraphParser>& graph_parser, std::unique_ptr<aimGraphCSR>& verify_graph, bool duplicate_check);

public:
	std::shared_ptr<Config> config;
	std::unique_ptr<MemoryManager> memory_manager;
	std::unique_ptr<EdgeUpdateManager> edge_update_manager;
	std::unique_ptr<VertexUpdateManager<VertexData, VertexUpdate>> vertex_update_manager;
};
