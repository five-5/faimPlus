//------------------------------------------------------------------------------
// filename: GraphParser.h
// Author : @SH
// Description: parser the input graph file to host memory in format in offset and adjacency
// Date：2019.5.8
//
//------------------------------------------------------------------------------
//

#pragma once

#include <string>
#include <memory>

#include "Definitions.h"

class GraphParser
{
public:
	enum class GraphFormat
	{
		DIMACS,
		SNAP,
		MM,
		RMAT,
		UNKNOWN
	};
public:
	explicit GraphParser(const std::string& filename) :
		filename_{ filename }, format_{ GraphFormat::UNKNOWN } {}
	~GraphParser() {}

	// Parses graph from file
	bool parseGraph(bool generateGraph = false);
	void getFreshGraph();
	bool parseDIMACSGraph();
	bool checkGraphFormat();
	bool generateGraphSynthetical();

	// Verification
	void printAdjacencyAtIndex(index_t index);

	// Getter & Setter
	vertex_t getNumberOfVertices() const { return number_vertices; }
	vertex_t getNumberOfEdges() const { return number_edges; }
	AdjacencyList_t& getAdjacency() { return adjacency_modifiable_; }
	OffsetList_t& getOffset() { return offset_modifiable_; }
	std::vector<index_t>& getIndexQueue() { return index_queue; }

private:
	std::string filename_;
	AdjacencyList_t adjacency_;
	OffsetList_t offset_;
	AdjacencyList_t adjacency_modifiable_;
	OffsetList_t offset_modifiable_;
	vertex_t number_vertices;
	vertex_t number_edges;
	vertex_t highest_edge;
	GraphFormat format_;

	std::vector<index_t> index_queue;
};
