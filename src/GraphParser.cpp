//------------------------------------------------------------------------------
// filename: GraphParser.cpp
// Author : @SH
// Description: parser the input graph file to host memory in format in offset and adjacency
// Date：2019.5.8
//
//------------------------------------------------------------------------------
//

#include <fstream>
#include <sstream>
#include <iostream>

#include "GraphParser.h"


//------------------------------------------------------------------------------
bool GraphParser::parseGraph(bool generateGraph)
{
	std::cout << "parseGraph called with " + filename_ << std::endl;

	if (generateGraph)
		return generateGraphSynthetical();

	// First check for valid graph format
	if (!checkGraphFormat())
		return false;

	if (format_ == GraphFormat::DIMACS)
		return parseDIMACSGraph();
}

//------------------------------------------------------------------------------
//
bool GraphParser::generateGraphSynthetical()
{
	srand(100);
	number_vertices = 80 * 1000 * 1000;
	vertex_t max_number_edges_per_adjacency = 10;
	vertex_t number_edges_per_adjacency;
	vertex_t vertex_index = 0;
	for (unsigned int i = 0; i < number_vertices; ++i)
	{
		offset_.push_back(vertex_index);
		number_edges_per_adjacency = (rand() % max_number_edges_per_adjacency) + 1;
		for (unsigned int j = 0; j < number_edges_per_adjacency; ++j)
		{
			adjacency_.push_back(rand() % number_vertices);
			++vertex_index;
		}
	}
	offset_.push_back(vertex_index);
	number_edges = adjacency_.size();
	std::cout << "#v: " << number_vertices << " and #e: " << number_edges << std::endl;
	getFreshGraph();
	return true;
}

//------------------------------------------------------------------------------
//
bool GraphParser::parseDIMACSGraph()
{
	// Open file and iterate over it line by line
	std::ifstream graph_file(filename_);
	std::string line;
	vertex_t index;
	vertex_t vertex_index = 0;
	highest_edge = 0;

	if (!graph_file.is_open())
	{
		std::cout << "File does not exist" << std::endl;
		return false;
	}

	/* Graph starts with #vertices #edges
	*  after that always the adjacency list of each vertex
	*/

	// Overstep comments and parse #v and #e 
	while (std::getline(graph_file, line))
	{
		std::istringstream istream(line);
		if (istream >> number_vertices)
		{
			// found first non-comment, we got #v and #e
			istream >> number_edges;
			break;
		}
	}

	vertex_t degree = 0, max_degree = 0;
	vertex_t line_num = 0, index_num = 0;
	// Parse adjacency list
	while (std::getline(graph_file, line))
	{
		offset_.push_back(vertex_index);
		degree = 0;
		std::istringstream istream(line);
		while (istream >> index)
		{
			// Graph format uses 1-n, we would like to have 0 - (n-1)
			adjacency_.push_back(index - 1);
			++vertex_index;
			++degree;
			if (index > highest_edge)
			{
				highest_edge = index;
			}
		}
		if (max_degree < degree) {
			index_num = line_num;
			max_degree = degree;
		}
		if (line != "") {
			++line_num;
		}
	}
	// Also include the offset for the #v+1 element (needed for a calculation later)
	offset_.push_back(vertex_index);
	number_edges = adjacency_.size();
	std::cout << "#v: " << number_vertices << " and #e: " << number_edges << " and highest edge: " << highest_edge << std::endl;
	printf("max_degree[%d] = %d\n", max_degree, index_num);
	std::cout << "End parsing Graph!" << std::endl;
	
	getFreshGraph();
	return true;
}


//------------------------------------------------------------------------------
//
void GraphParser::getFreshGraph()
{
	adjacency_modifiable_ = adjacency_;
	offset_modifiable_ = offset_;
}

//------------------------------------------------------------------------------
//
bool GraphParser::checkGraphFormat()
{
	if (filename_.find(".graph") != std::string::npos)
	{
		format_ = GraphFormat::DIMACS;
		//std::cout << "Graph format is DIMACS!" << std::endl;
		return true;
	}
	else if (filename_.find(".txt") != std::string::npos)
	{
		format_ = GraphFormat::SNAP;
		std::cout << "Graph format SNAP is currently not supported!" << std::endl;
		return false;
	}
	else if (filename_.find(".mtx") != std::string::npos)
	{
		format_ = GraphFormat::MM;
		/*std::cout << "Graph format MM is currently not supported!" << std::endl;*/
		return true;
	}
	else if (filename_.find(".kron") != std::string::npos)
	{
		format_ = GraphFormat::RMAT;
		std::cout << "Graph format RMAT is currently not supported!" << std::endl;
		return false;
	}
	else
	{
		format_ = GraphFormat::UNKNOWN;
		std::cout << "Invalid Format" << std::endl;
		return false;
	}
}

//------------------------------------------------------------------------------
//
void GraphParser::printAdjacencyAtIndex(index_t index)
{
	index_t start_index = offset_.at(index);
	index_t end_index = offset_.at(index + 1);
	std::cout << "Print adjacency for index " << index << std::endl;
	for (size_t i = start_index; i < end_index; ++i)
	{
		std::cout << adjacency_.at(i) << " | ";
	}
	std::cout << std::endl;
}