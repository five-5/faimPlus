//------------------------------------------------------------------------------
// filename: EdgeUpdate.cpp
// Author : @SH
// Date：2019.5.13
//
//------------------------------------------------------------------------------
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <time.h>

#include "EdgeUpdate.h"
#include "GraphParser.h"
#include "MemoryManager.h"

//------------------------------------------------------------------------------
//
std::unique_ptr<EdgeUpdateBatch> EdgeUpdateManager::generateEdgeUpdates(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range, unsigned int offset)
{
	std::unique_ptr<EdgeUpdateBatch> edge_update(std::make_unique<EdgeUpdateBatch>());
	// Generate random edge updates
	srand(seed + 1);
	for (vertex_t i = 0; i < batch_size / 2; ++i)
	{
		EdgeUpdate edge_update_data;
		vertex_t intermediate = rand() % ((range && (range < number_vertices)) ? range : number_vertices);
		vertex_t source;
		if (offset + intermediate < number_vertices)
			source = offset + intermediate;
		else
			source = intermediate;
		edge_update_data.source = source;
		edge_update_data.update.destination = rand() % number_vertices;
		edge_update->edge_update.push_back(edge_update_data);
	}

	for (vertex_t i = batch_size / 2; i < batch_size; ++i)
	{
		EdgeUpdate edge_update_data;
		vertex_t intermediate = rand() % (number_vertices);
		vertex_t source;
		if (offset + intermediate < number_vertices)
			source = offset + intermediate;
		else
			source = intermediate;
		edge_update_data.source = source;
		edge_update_data.update.destination = rand() % number_vertices;
		edge_update->edge_update.push_back(edge_update_data);
	}

	/*for (auto const& update : edge_update->edge_update)
	{
	if (update.source == 218)
	printf("Generate Update %u | %u\n", update.source, update.update.destination);
	}*/

	// Write data to file to verify
	static int counter = 0;
#ifdef DEBUG_VERBOSE_OUPUT
	std::string filename = std::string("../tests/Verification/VerifyInsert");
	filename += std::to_string(counter) + std::string(".txt");
	std::ofstream file(filename);
	if (file.is_open())
	{
		for (int i = 0; i < batch_size; ++i)
		{
			file << edge_update->edge_update.at(i).source << " ";
			file << edge_update->edge_update.at(i).update.destination << "\n";
		}
	}
#endif
	++counter;

	return std::move(edge_update);
}


//------------------------------------------------------------------------------
//
void EdgeUpdateManager::receiveEdgeUpdates(std::unique_ptr<EdgeUpdateBatch> edge_updates, EdgeUpdateVersion type)
{
	if (type == EdgeUpdateVersion::GENERAL)
	{
		updates = std::move(edge_updates);
	}
	else if (type == EdgeUpdateVersion::INSERTION)
	{
		updates_insertion = std::move(edge_updates);
	}
	else if (type == EdgeUpdateVersion::DELETION)
	{
		updates_deletion = std::move(edge_updates);
	}
}

//------------------------------------------------------------------------------
//

void EdgeUpdateManager::hostCudaAllocConcurrentUpdates()
{
	HANDLE_ERROR(cudaHostAlloc((void **)&(updates_insertion->raw_edge_update), updates_insertion->edge_update.size() * sizeof(EdgeUpdate), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void **)&(updates_deletion->raw_edge_update), updates_deletion->edge_update.size() * sizeof(EdgeUpdate), cudaHostAllocDefault));
}


//------------------------------------------------------------------------------
//

void EdgeUpdateManager::hostCudaFreeConcurrentUpdates()
{
	HANDLE_ERROR(cudaFreeHost(updates_insertion->raw_edge_update));
	HANDLE_ERROR(cudaFreeHost(updates_deletion->raw_edge_update));
}

//------------------------------------------------------------------------------
//
void EdgeUpdateManager::writeEdgeUpdatesToFile(vertex_t number_vertices, vertex_t batch_size, const std::string& filename)
{
	std::ofstream file(filename);
	srand(static_cast<unsigned int>(time(NULL)));

	if (file.is_open())
	{
		for (vertex_t i = 0; i < batch_size; ++i)
		{
			vertex_t edge_src = rand() % number_vertices;
			vertex_t edge_dst = rand() % number_vertices;
			file << edge_src << " " << edge_dst << std::endl;
		}
	}
	file.close();
	return;
}

//------------------------------------------------------------------------------
//

void EdgeUpdateManager::writeEdgeUpdatesToFile(const std::unique_ptr<EdgeUpdateBatch>& edges, vertex_t batch_size, const std::string& filename)
{
	std::ofstream file(filename);

	if (file.is_open())
	{
		for (vertex_t i = 0; i < batch_size; ++i)
		{
			vertex_t edge_src = edges->edge_update.at(i).source;
			vertex_t edge_dst = edges->edge_update.at(i).update.destination;
			file << "|" << edge_src << " " << edge_dst << std::endl;
		}
	}
	return;
}

//------------------------------------------------------------------------------
//

std::unique_ptr<EdgeUpdateBatch> EdgeUpdateManager::readEdgeUpdatesFromFile(const std::string& filename)
{
	std::unique_ptr<EdgeUpdateBatch> edge_update(new EdgeUpdateBatch());
	std::ifstream graph_file(filename);
	std::string line;

	while (std::getline(graph_file, line))
	{
		EdgeUpdate edge_update_data;
		std::istringstream istream(line);

		istream >> edge_update_data.source;
		istream >> edge_update_data.update.destination;
		edge_update->edge_update.push_back(edge_update_data);
	}

	return std::move(edge_update);
}

//------------------------------------------------------------------------------
//

void EdgeUpdateManager::hostEdgeInsertion(const std::unique_ptr<GraphParser>& parser)
{
	auto& adjacency = parser->getAdjacency();
	auto& offset = parser->getOffset();
	auto number_vertices = parser->getNumberOfVertices();
	int number_updates;
	if (updates)
	{
		number_updates = updates->edge_update.size();
	}
	else
	{
		number_updates = updates_insertion->edge_update.size();
	}

	// Go over all updates
	for (int i = 0; i < number_updates; ++i)
	{
		// Set iterator to begin()
		auto iter = adjacency.begin();
		vertex_t edge_src, edge_dst;

		if (updates)
		{
			edge_src = updates->edge_update.at(i).source;
			edge_dst = updates->edge_update.at(i).update.destination;
		}
		else
		{
			edge_src = updates_insertion->edge_update.at(i).source;
			edge_dst = updates_insertion->edge_update.at(i).update.destination;
		}

		//------------------------------------------------------------------------------
		// TODO: Currently no support for adding new vertices!!!!!
		//------------------------------------------------------------------------------
		//
		if (edge_src >= number_vertices || edge_dst >= number_vertices)
		{
			continue;
		}

		// Calculate iterator positions
		auto begin_iter = iter + offset.at(edge_src);
		auto end_iter = iter + offset.at(edge_src + 1);

		// Search item
		auto pos = std::find(begin_iter, end_iter, edge_dst);
		if (pos != end_iter)
		{
			// Edge already present     
			continue;
		}
		else
		{
			// Insert edge
			adjacency.insert(pos, edge_dst);

			// Update offset list (on the host this is number_vertices + 1 in size)
			for (auto i = edge_src + 1; i < (number_vertices + 1); ++i)
			{
				offset[i] += 1;
			}
		}

	}
	return;
}

//------------------------------------------------------------------------------
//
void EdgeUpdateManager::hostEdgeDeletion(const std::unique_ptr<GraphParser>& parser)
{
	auto& adjacency = parser->getAdjacency();
	auto& offset = parser->getOffset();
	auto number_vertices = parser->getNumberOfVertices();
	int number_updates;
	if (updates)
	{
		number_updates = updates->edge_update.size();
	}
	else
	{
		number_updates = updates_insertion->edge_update.size();
	}

	// Go over all updates
	for (int i = 0; i < number_updates; ++i)
	{
		// Set iterator to begin()
		auto iter = adjacency.begin();
		vertex_t edge_src, edge_dst;

		if (updates)
		{
			edge_src = updates->edge_update.at(i).source;
			edge_dst = updates->edge_update.at(i).update.destination;
		}
		else
		{
			edge_src = updates_insertion->edge_update.at(i).source;
			edge_dst = updates_insertion->edge_update.at(i).update.destination;
		}

		// Check if valid vertices are given
		if (edge_src >= number_vertices || edge_dst >= number_vertices)
		{
			continue;
		}

		// Calculate iterator positions
		auto begin_iter = iter + offset.at(edge_src);
		auto end_iter = iter + offset.at(edge_src + 1);

		// Search item
		auto pos = std::find(begin_iter, end_iter, edge_dst);
		if (pos != end_iter)
		{
			// Found edge, will be deleted now
			adjacency.erase(pos);

			// Update offset list (on the host this is number_vertices + 1 in size)
			for (auto i = edge_src + 1; i < (number_vertices + 1); ++i)
			{
				offset[i] -= 1;
			}
		}
		else
		{
			// Edge not present
			continue;
		}
	}
	return;
}

//------------------------------------------------------------------------------
//
void EdgeUpdateManager::writeGraphsToFile(const std::unique_ptr<aimGraphCSR>& verify_graph,
	const std::unique_ptr<GraphParser>& graph_parser,
	const std::string& filename)
{
	static int counter = 0;
	std::string prover_filename = filename;// + "_prover" + std::to_string(counter) + ".txt";
										   //std::string verifier_filename = filename + "_verifier" + std::to_string(counter) + ".txt";
	int number_vertices = graph_parser->getNumberOfVertices();
	++counter;

	// Start with prover graph
	std::ofstream prover_file(prover_filename);

	// Write number of vertices and edges
	prover_file << verify_graph->number_vertices << " " << verify_graph->number_edges << std::endl;

	for (index_t i = 0; i < verify_graph->number_vertices; ++i)
	{
		int offset = verify_graph->h_offset[i];
		int neighbours = ((i == (verify_graph->number_vertices - 1))
			? (verify_graph->number_edges)
			: (verify_graph->h_offset[i + 1]))
			- offset;
		for (int j = 0; j < neighbours; ++j)
		{
			// Graph format uses 1-n, we have 0 - (n-1), hence now add 1
			prover_file << verify_graph->h_adjacency[offset + j] + 1;
			if (j < (neighbours - 1))
				prover_file << " ";
		}
		prover_file << "\n";
	}

	prover_file.close();

	// End with verifier graph
	// std::ofstream verifier_file(verifier_filename);

	// // Write number of vertices and edges
	// verifier_file << number_vertices << " " << graph_parser->getAdjacency().size()  << std::endl;

	// for(int i = 0; i < number_vertices; ++i)
	// {
	//   int offset = graph_parser->getOffset().at(i);
	//   int neighbours = graph_parser->getOffset().at(i+1) - offset;
	//   for(int j = 0; j < neighbours; ++j)
	//   {
	//     // Graph format uses 1-n, we have 0 - (n-1), hence now add 1
	//     verifier_file << graph_parser->getAdjacency().at(offset + j) + 1;
	//     if(j < (neighbours - 1))
	//       verifier_file << " ";
	//   }
	//   verifier_file << "\n";
	// }

	// verifier_file.close();

	// std::cout << "Writing files is done" << std::endl;

	return;
}