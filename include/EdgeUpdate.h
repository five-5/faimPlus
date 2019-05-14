//------------------------------------------------------------------------------
// filename: EdgeUpdate.h
// Author : @SH
// Date：2019.5.8
//
//------------------------------------------------------------------------------
//

#pragma once

#include "Utility.h"
#include "MemoryManager.h"

#define QUEUING
#define CLEAN_PAGE

enum class QueryKernelConfig
{
	STANDARD,
	WARPSIZED,
	VERTEXCENTRIC
};

// Forward declaration
class GraphParser;
class Config;
class faimGraph;

class EdgeBlock
{
	EdgeData edgeblock[15];
};

/*! \class EdgeUpdateBatch
\brief Templatised class to hold a batch of edge updates
*/
// √
class EdgeUpdateBatch
{
public:
	// Host side
	std::vector<EdgeUpdate> edge_update;
	EdgeUpdate* raw_edge_update;

	// Device side
	EdgeUpdate* d_edge_update;
};



/// Modify:@SH change vertex_neighbour to workitem
//typedef struct vertex_neighbour workitem;
struct workitem {
	index_t index;
	index_t neighbour;
	/// Add:@SH add page_num attribute
	index_t page_num;
	index_t off;
	__host__ __device__ workitem() { index = DELETIONMARKER; neighbour = 0; page_num = 0; off = 0; }
	__host__ __device__ workitem(int x) { page_num = x; }
	__host__ __device__ workitem(int x, int y, int t, int off) :index(x), neighbour(y), page_num(t), off(off) {}
	__host__ __device__ workitem(const workitem& w) { index = w.index; neighbour = w.neighbour; page_num = w.page_num; off = w.off; }

	/*friend __host__ __device__ bool operator<(const vertex_neighbour &lhs, const vertex_neighbour &rhs)

	{
	return ((lhs.neighbour < rhs.neighbour));
	}*/
	// Modify:@SH change the process logic 
	friend __host__ __device__ workitem operator+(const workitem &lhs, const workitem &rhs)
	{
		return workitem(rhs.index, rhs.neighbour, rhs.page_num + lhs.page_num, rhs.off);
	}
};

/*! \class EdgeUpdatePreProcessing
\brief Templatised class used for preprocessing of edge updates
*/
class EdgeUpdatePreProcessing
{
public:

	EdgeUpdatePreProcessing(vertex_t number_vertices, vertex_t batch_size, std::unique_ptr<MemoryManager>& memory_manager, size_t sizeofVertexData)
	{
		TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), number_vertices, sizeofVertexData);
		temp_memory_dispenser.getTemporaryMemory<EdgeUpdate>(batch_size); // Move after update data

																			  // Now let's set the member pointers
		d_edge_src_counter = temp_memory_dispenser.getTemporaryMemory<index_t>(number_vertices + 1);
		d_update_src_offsets = temp_memory_dispenser.getTemporaryMemory<index_t>(number_vertices + 1);
		d_different_vertexes = temp_memory_dispenser.getTemporaryMemory<workitem>(number_vertices + 1);
	}

	index_t* d_edge_src_counter;
	index_t* d_update_src_offsets;
	workitem* d_different_vertexes;
};

enum class EdgeUpdateVersion
{
	GENERAL,
	INSERTION,
	DELETION
};

enum class EdgeUpdateMechanism
{
	SEQUENTIAL,
	CONCURRENT
};

//------------------------------------------------------------------------------
//
class EdgeUpdateManager
{
public:
	EdgeUpdateManager() : update_type{ EdgeUpdateMechanism::SEQUENTIAL } {}

	// Sequential Update Functionality on device
	void deviceEdgeInsertion(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);
	void deviceEdgeDeletion(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);

	// Concurrent Update Functionality on device
	void deviceEdgeUpdateConcurrentStream(cudaStream_t& insertion_stream, cudaStream_t& deletion_stream, std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);
	void deviceEdgeUpdateConcurrent(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);

	// Sequential Update Functionality on host
	void hostEdgeInsertion(const std::unique_ptr<GraphParser>& parser);
	void hostEdgeDeletion(const std::unique_ptr<GraphParser>& parser);

	// Generate Edge Update Data
	std::unique_ptr<EdgeUpdateBatch> generateEdgeUpdates(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range = 0, unsigned int offset = 0);
	std::unique_ptr<EdgeUpdateBatch> generateEdgeUpdates(const std::unique_ptr<MemoryManager>& memory_manager, vertex_t batch_size, unsigned int seed, unsigned int range = 0, unsigned int offset = 0);
	template <typename VertexUpdateType>
	std::unique_ptr<EdgeUpdateBatch<EdgeUpdateType>> generateEdgeUpdatesConcurrent(std::unique_ptr<faimGraph>& faimGraph, const std::unique_ptr<MemoryManager>& memory_manager, vertex_t batch_size, unsigned int seed, unsigned int range = 0, unsigned int offset = 0);

	void receiveEdgeUpdates(std::unique_ptr<EdgeUpdateBatch> updates, EdgeUpdateVersion type);
	void hostCudaAllocConcurrentUpdates();
	void hostCudaFreeConcurrentUpdates();

	// Edge Update Processing
	std::unique_ptr<EdgeUpdatePreProcessing> edgeUpdatePreprocessing(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);
	void edgeUpdateDuplicateChecking(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config, const std::unique_ptr<EdgeUpdatePreProcessing>& preprocessed);
	void edgeUpdateDuplicateCheckingByBlocksize(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config, const std::unique_ptr<EdgeUpdatePreProcessing>& preprocessed, index_t* d_pageindex, index_t* thm, index_t* ths, index_t* th0, index_t* deltehelp);
	void updateWorkItem(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config, const std::unique_ptr<EdgeUpdatePreProcessing>& preprocessed, index_t* scanhelper, index_t* thm, index_t* ths, index_t* th0);
	//void setupPhrase(std::unique_ptr<MemoryManager>& memory_manager, EdgeUpdateType* edge_update_data, int batch_size, int grid_size, int block_size);
	void setupPhrase(std::unique_ptr<MemoryManager>& memory_manager, int batch_size, int grid_size, int block_size);
	//void d_setupneighbor(EdgeUpdateType* edge_update_data, int batch_size);

	void setupscanhelper(std::unique_ptr<MemoryManager>& memory_manager, int batch_size, index_t* scanhelper);
	// Write/Read Update to/from file
	void writeEdgeUpdatesToFile(vertex_t number_vertices, vertex_t batch_size, const std::string& filename);
	void writeEdgeUpdatesToFile(const std::unique_ptr<EdgeUpdateBatch>& edges, vertex_t batch_size, const std::string& filename);
	std::unique_ptr<EdgeUpdateBatch> readEdgeUpdatesFromFile(const std::string& filename);
	void writeGraphsToFile(const std::unique_ptr<aimGraphCSR>& verify_graph, const std::unique_ptr<GraphParser>& graph_parser, const std::string& filename);

	// General
	void setUpdateType(EdgeUpdateMechanism type) { update_type = type; }
	EdgeUpdateMechanism getUpdateType() { return update_type; }

private:
	// Interface for calling update kernel explicitely
	void w_edgeInsertion(cudaStream_t& stream, const std::unique_ptr<EdgeUpdateBatch>& updates_insertion, std::unique_ptr<MemoryManager>& memory_manager, int batch_size, int block_size, int grid_size);
	void w_edgeDeletion(cudaStream_t& stream, const std::unique_ptr<EdgeUpdateBatch>& updates_deletion, std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config, int batch_size, int block_size, int grid_size);

	std::unique_ptr<EdgeUpdateBatch> updates;
	std::unique_ptr<EdgeUpdateBatch> updates_insertion;
	std::unique_ptr<EdgeUpdateBatch> updates_deletion;
	EdgeUpdateMechanism update_type;
};

//------------------------------------------------------------------------------
//

class EdgeQueryManager
{
public:
	void deviceQuery(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);
	void generateQueries(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range = 0, unsigned int offset = 0);
	void receiveQueries(std::unique_ptr<EdgeUpdateBatch> adjacency_queries)
	{
		queries = std::move(adjacency_queries);
	}
	std::unique_ptr<EdgeUpdatePreProcessing> edgeQueryPreprocessing(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& configuration);

private:
	std::unique_ptr<EdgeUpdateBatch> queries;
	std::unique_ptr<bool[]> query_results;
	bool* d_query_results;
	QueryKernelConfig config{ QueryKernelConfig::VERTEXCENTRIC };
};