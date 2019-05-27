//------------------------------------------------------------------------------
// filename: faimGraph.cu
// Author : @SH
// Date：2019.5.8
//
//------------------------------------------------------------------------------
//

#pragma once

#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <thrust/device_vector.h>

#include "faimGraph.h"
#include "MemoryManager.h"
#include "GraphParser.h"
#include "ConfigurationParser.h"
#include "CSR.h"


namespace faimGraphGeneral
{
	//#############################################################################
	// Device funtionality
	//#############################################################################

	//------------------------------------------------------------------------------
	// calculate the pages every vertex adjacency need
	__global__ void d_calculateMemoryRequirements(MemoryManager* memory_manager,
		vertex_t* offset,
		vertex_t* neighbours,
		vertex_t* block_requirements,
		vertex_t* pageindex_requirements,
		int number_vertices,
		int page_size)
	{
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if (tid >= number_vertices)
		{
			if (tid == number_vertices) {
				block_requirements[tid] = 0;
				pageindex_requirements[tid] = 0;
			}
			return;
		}


		// Calculate memory requirements
		vertex_t vertices_in_adjacency = offset[tid + 1] - offset[tid];
		// We need space for all the vertices in EdgeData and also need an edgeblock index per block
		vertex_t number_blocks = (vertex_t)ceil((float)(vertices_in_adjacency * MEMORYOVERALLOCATION) / (float)(memory_manager->edges_per_page));
		// If we have no edges initially, we still want an empty block
		if (number_blocks == 0)
		{
			number_blocks = 1;
		}

		neighbours[tid] = vertices_in_adjacency;
		block_requirements[tid] = number_blocks;
		pageindex_requirements[tid] = (vertex_t)ceil((float)(number_blocks * MEMORYOVERALLOCATION) / (float)(memory_manager->pageindexes_per_page));

		return;
	}

	//------------------------------------------------------------------------------
	//
	__global__ void d_setupFaimGraph(MemoryManager* memory_manager,
		memory_t* memory,
		vertex_t* adjacency,
		vertex_t* offset,
		vertex_t* neighbours,
		vertex_t* mem_offsets,
		vertex_t* pageindex_offsets,
		int number_vertices,
		int page_size,
		ConfigurationParameters::PageLinkage page_linkage)
	{
		int tid = threadIdx.x + blockIdx.x*blockDim.x;

		// Initialise queuing approach
		memory_manager->d_page_queue.init();
		memory_manager->d_vertex_queue.init();

		if (tid >= number_vertices)
			return;

		// Setup memory
		VertexData* vertices = (VertexData*)memory;
		VertexData vertex;

		// TODO: @SH modify to record the start of page index 
		vertex.mem_index = pageindex_offsets[tid];  
		vertex.locking = UNLOCK;
		vertex_t edges_per_page = memory_manager->edges_per_page;
		vertex_t adj_page_index = mem_offsets[tid];
		vertex.neighbours = neighbours[tid];
		vertex.capacity = ((mem_offsets[tid + 1] - adj_page_index) * memory_manager->edges_per_page);
		vertex.host_identifier = tid;

		// Set vertex management data in memory
		vertices[tid] = vertex;

		AdjacencyIterator adjacency_iterator(pageAccess<EdgeData>(memory, adj_page_index, page_size, memory_manager->start_index));   // @SH
		PageIndexIterator pageindex_iterator(pageAccess<index_t>(memory, vertex.mem_index, page_size, memory_manager->start_page_index));  // @SH

		// Setup next free block
		if (tid == (number_vertices - 1))
		{
			// The last vertex sets the next free block
			memory_manager->next_free_page = mem_offsets[number_vertices];
			memory_manager->next_free_pageindex_page = pageindex_offsets[number_vertices];         // @SH
			                     
			// Decrease free memory at initialization
			memory_manager->free_memory -= (page_size * (mem_offsets[number_vertices]));
			memory_manager->free_memory -= (page_size * (pageindex_offsets[number_vertices]));     // @SH
		}

		// Write EdgeData
		int offset_index = offset[tid];
		int pages = 0; //@SH
					   //@SH
		pageindex_iterator.pageindexSetup(pages, adj_page_index, edges_per_page, memory, page_size, memory_manager->start_page_index, vertex.mem_index);  
		for (int i = 0; i < vertex.neighbours; ++i)
		{
			if (page_linkage == ConfigurationParameters::PageLinkage::SINGLE) {
				adjacency_iterator.adjacencySetup(i, edges_per_page, memory, page_size, memory_manager->start_index, adjacency, offset_index, adj_page_index, pages, memory_manager->pageindexes_per_page, memory_manager->start_page_index, pageindex_iterator, vertex.mem_index);
			}
		}

		// Set the rest adjacency page to deletionmarker
		for (int i = vertex.neighbours; i < vertex.capacity; ++i)
		{
			setDeletionMarker(adjacency_iterator.getIterator(), edges_per_page);
			++adjacency_iterator;
		} 

		// @SH
		// Set the rest pageindex page to deletionmarker
		int pageindexes_per_page = memory_manager->pageindexes_per_page;
		for (int i = pages % pageindexes_per_page; i < pageindexes_per_page; ++i)
		{
			pageindex_iterator[i] = DELETIONMARKER;
		}

		return;
	}

	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType, typename EdgeDataType>
	__global__ void d_faimGraphToCSR(MemoryManager* memory_manager,
		memory_t* memory,
		vertex_t* adjacency,
		vertex_t* offset,
		int page_size)
	{
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if (tid >= memory_manager->next_free_vertex_index)
			return;

		VertexDataType* vertices = (VertexDataType*)memory;

		// Deleted vertices
		if (vertices[tid].host_identifier == DELETIONMARKER)
		{
			return;
		}

		// Retrieve data
		vertex_t edge_data_index = vertices[tid].mem_index;
		vertex_t number_neighbours = vertices[tid].neighbours;
		vertex_t edges_per_page = memory_manager->edges_per_page;
		AdjacencyIterator adjacency_iterator(pageAccess<EdgeDataType>(memory, edge_data_index, page_size, memory_manager->start_index));


		// Write EdgeData
		int offset_index = offset[tid];
		for (int i = 0, j = 0; j < number_neighbours; ++i)
		{
			// Normal case
			vertex_t adj_dest = adjacency_iterator.getDestination();
			if (adj_dest != DELETIONMARKER)
			{
				adjacency[offset_index + j] = adj_dest;
				j += 1;
			}
			adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
		}

		return;
	}


	//------------------------------------------------------------------------------
	//
	__global__ void d_CompareGraphs(vertex_t* adjacency_prover,
		vertex_t* adjacency_verifier,
		vertex_t* offset_prover,
		vertex_t* offset_verifier,
		int number_vertices,
		int* equal)
	{
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if (tid >= number_vertices)
			return;

		// Compare offset
		if (offset_prover[tid] != offset_verifier[tid])
		{
			printf("Offset-Round: %d | Prover: %d | Verifier: %d\n", tid, offset_prover[tid], offset_verifier[tid]);
			*equal = 0;
			return;
		}

		// Compare adjacency
		int offset = offset_verifier[tid];
		int neighbours = offset_verifier[tid + 1] - offset;
		for (int j = 0; j < neighbours; ++j)
		{
			int found_match = 0;
			for (int k = 0; k < neighbours; ++k)
			{
				if (adjacency_prover[offset + k] == adjacency_verifier[offset + j])
				{
					found_match = 1;
					break;
				}
			}
			if (found_match == 0)
			{
				printf("Vertex-Index: %d\n", tid);
				if (tid != number_vertices - 1)
				{
					printf("[DEVICE] Neighbours: %d\n", offset_prover[tid + 1] - offset_prover[tid]);
				}
				printf("[DEVICE]Prover-List:\n");
				for (int l = 0; l < neighbours; ++l)
				{
					printf(" %d", adjacency_prover[offset + l]);
				}
				if (tid != number_vertices - 1)
				{
					printf("\n[HOST] Neighbours: %d\n", offset_verifier[tid + 1] - offset_verifier[tid]);
				}
				printf("[HOST]Verifier-List:\n");
				for (int l = 0; l < neighbours; ++l)
				{
					printf(" %d", adjacency_verifier[offset + l]);
				}
				*equal = 0;
				return;
			}
		}
		return;
	}
}


//############################################################################################################################################################
// Host funtionality
//############################################################################################################################################################

//------------------------------------------------------------------------------
// @SH modified
void faimGraph::initializeMemory(std::unique_ptr<GraphParser>& graph_parser)
{
	// First setup memory manager
	memory_manager->initialize(config);

	// Set data in memory manager
	int number_of_vertices = graph_parser->getNumberOfVertices();

	// Setup csr data and calculate launch params
	std::unique_ptr<CSRData> csr_data(new CSRData(graph_parser, memory_manager));
	int block_size = config->testruns_.at(config->testrun_index_)->params->init_launch_block_size_;
	int grid_size = (number_of_vertices / block_size) + 1;

	// Push memory manager information to device
	updateMemoryManagerDevice(memory_manager);

	// Calculate memory requirements
	faimGraphGeneral::d_calculateMemoryRequirements << < grid_size, block_size >> >((MemoryManager*)memory_manager->d_memory,
		csr_data->d_offset,
		csr_data->d_neighbours,
		csr_data->d_block_requirements,		// 计算被赋值
		csr_data->d_pageindex_requirements, // @SH 计算被赋值
		number_of_vertices,
		memory_manager->page_size);


	// Prefix scan on d_block_requirements to get correct memory offsets
	thrust::device_ptr<vertex_t> th_block_requirements(csr_data->d_block_requirements);
	thrust::exclusive_scan(th_block_requirements, th_block_requirements + number_of_vertices + 1, th_block_requirements);

	// @SH Prefix scan on d_pageindex_requirements to get correct memory offsets
	thrust::device_ptr<vertex_t> th_pageindex_requirements(csr_data->d_pageindex_requirements);
	thrust::exclusive_scan(th_pageindex_requirements, th_pageindex_requirements + number_of_vertices + 1, th_pageindex_requirements);

	// Setup GPU Streaming memory
	faimGraphGeneral::d_setupFaimGraph << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
		memory_manager->d_data,
		csr_data->d_adjacency,
		csr_data->d_offset,
		csr_data->d_neighbours,
		csr_data->d_block_requirements,
		csr_data->d_pageindex_requirements,
		number_of_vertices,
		memory_manager->page_size,
		memory_manager->page_linkage);

	// Push memory manager information back to host
	size_t mem_before_device_update = memory_manager->free_memory;
	updateMemoryManagerHost(memory_manager);

	// Vertex management data allocated
	memory_manager->decreaseAvailableMemory(sizeof(VertexData) * number_of_vertices);

	return;
}

// @SH modified
void faimGraph::initializeMemory(vertex_t* d_offset, vertex_t* d_adjacency, int number_vertices)
{
	// First setup memory manager
	memory_manager->initialize(config);

	// Setup csr data and calculate launch params
	std::unique_ptr<CSRData> csr_data(new CSRData(d_offset, d_adjacency, memory_manager, memory_manager->number_vertices, memory_manager->number_edges));
	int block_size = config->testruns_.at(config->testrun_index_)->params->init_launch_block_size_;
	int grid_size = (number_vertices / block_size) + 1;

	// Push memory manager information to device
	updateMemoryManagerDevice(memory_manager);

	// Calculate memory requirements
	faimGraphGeneral::d_calculateMemoryRequirements << < grid_size, block_size >> >((MemoryManager*)memory_manager->d_memory,
		csr_data->d_offset,
		csr_data->d_neighbours,
		csr_data->d_block_requirements,
		csr_data->d_pageindex_requirements, // @SH 计算被赋值
		number_vertices,
		memory_manager->page_size);

	// Prefix scan on d_block_requirements to get correct memory offsets
	thrust::device_ptr<vertex_t> th_block_requirements(csr_data->d_block_requirements);
	thrust::exclusive_scan(th_block_requirements, th_block_requirements + number_vertices + 1, th_block_requirements);

	// @SH Prefix scan on d_pageindex_requirements to get correct memory offsets
	thrust::device_ptr<vertex_t> th_pageindex_requirements(csr_data->d_pageindex_requirements);
	thrust::exclusive_scan(th_pageindex_requirements, th_pageindex_requirements + number_vertices + 1, th_pageindex_requirements);

	// Setup GPU Streaming memory
	faimGraphGeneral::d_setupFaimGraph << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
		memory_manager->d_data,
		csr_data->d_adjacency,
		csr_data->d_offset,
		csr_data->d_neighbours,
		csr_data->d_block_requirements,
		csr_data->d_pageindex_requirements,
		number_vertices,
		memory_manager->page_size,
		memory_manager->page_linkage);

	// Push memory manager information back to host
	size_t mem_before_device_update = memory_manager->free_memory;
	updateMemoryManagerHost(memory_manager);

	// Vertex management data allocated
	memory_manager->decreaseAvailableMemory(sizeof(VertexData) * number_vertices);
}

// ------------------------------------------------------------------------------
//

CSR<float> faimGraph::reinitializeFaimGraph(float overallocation_factor)
{
	/*
	* Three cases:
	* 1.) Both old and new faimGraph fit in memory at the same time, initialize new memory directly from old memory
	* 2.) New does not fit in memory while old is in memory, retire to CSR -> init anew
	* 3.) Size increase not possible any longer
	*	a.) Retire to CSR and back to Host
	*	b.) If not possible, retire faimGraph to Host entirely and retire to CSR on host
	*/

	/*
	TODO:
	* Number Edges does not yet reflect correct count!
	*
	*/

	memory_t* device_memory{ nullptr }, *csr_helper{ nullptr };
	int memory_manager_offset = MEMMANOFFSET * static_cast<int>(ceil(static_cast<float>(sizeof(MemoryManager)) / static_cast<float>(MEMMANOFFSET)));
	bool allocationSuccessful{ true }, csrSuccessful{ true };
	CSR<float> host_csr;
	uint64_t new_size = memory_manager->total_memory * overallocation_factor;

	// Attempt allocation
	if (cudaSuccess != cudaMalloc((void**)&device_memory, new_size))
	{
		allocationSuccessful = false;
		printf("New Allocation is not possible with size %zd\n", new_size);
	}

	if (allocationSuccessful)
	{
		/*printf("Initialize new faimGraph from old faimGraph\n");*/
		//  Both old and new faimGraph fit in memory at the same time, initialize new memory directly from old memory

		/*
		###########################################################################################################
		mem_man | vertices | ++++++++++++++++ space ++++++++++++++++++++| pages | stack | vertex_queue | page_queue
		###########################################################################################################
		*/

		// Copy over mem_man + vertices
		HANDLE_ERROR(cudaMemcpy(device_memory, memory_manager->d_memory, memory_manager_offset + (sizeof(VertexData) * memory_manager->next_free_vertex_index), cudaMemcpyDeviceToDevice));

		// Copy over pages
		memory_t* ptr_pages_old_faimGraph = pageAccess<memory_t>(memory_manager->d_data, memory_manager->next_free_page - 1, memory_manager->page_size, memory_manager->start_index);
		uint64_t new_start_index = static_cast<uint64_t>((new_size - (memory_manager->d_page_queue.size_ * sizeof(index_t) * 2) - config->testruns_.at(config->testrun_index_)->params->stacksize_ - memory_manager_offset) / memory_manager->page_size) - 1;
		memory_t* ptr_pages_new_faimGraph = pageAccess<memory_t>(device_memory + memory_manager_offset, memory_manager->next_free_page - 1, memory_manager->page_size, new_start_index);
		HANDLE_ERROR(cudaMemcpy(ptr_pages_new_faimGraph, ptr_pages_old_faimGraph, memory_manager->page_size * (memory_manager->next_free_page), cudaMemcpyDeviceToDevice));

		// Copy over queues
		memory_t* old_queue_ptr = reinterpret_cast<memory_t*>(memory_manager->d_vertex_queue.queue_);
		memory_manager->d_page_queue.queue_ = reinterpret_cast<index_t*>(device_memory + new_size);
		memory_manager->d_page_queue.queue_ -= memory_manager->d_page_queue.size_;
		memory_manager->d_vertex_queue.queue_ = memory_manager->d_page_queue.queue_ - memory_manager->d_vertex_queue.size_;
		memory_manager->d_stack_pointer = reinterpret_cast<memory_t*>(memory_manager->d_vertex_queue.queue_);
		HANDLE_ERROR(cudaMemcpy(memory_manager->d_vertex_queue.queue_, old_queue_ptr, memory_manager->d_page_queue.size_ * sizeof(index_t) * 2, cudaMemcpyDeviceToDevice));

		// Free old memory
		HANDLE_ERROR(cudaFree(memory_manager->d_memory));

		// Set new pointers
		memory_manager->d_memory = device_memory;
		memory_manager->d_data = memory_manager->d_memory + memory_manager_offset;

		// Set appropriate parameters anew
		memory_manager->total_memory = new_size;
		memory_manager->free_memory = new_size - (memory_manager->page_size * (memory_manager->next_free_page - 1) + memory_manager_offset + (sizeof(VertexData) * memory_manager->next_free_vertex_index) + memory_manager->d_page_queue.size_ * sizeof(index_t) * 2);
		memory_manager->start_index = new_start_index;
		updateMemoryManagerDevice(memory_manager);
	}
	else
	{
		uint64_t csr_size = ((memory_manager->next_free_vertex_index + 1) * 3 * sizeof(vertex_t)) + (memory_manager->number_edges * sizeof(index_t));
		if (cudaSuccess != cudaMalloc((void**)&csr_helper, csr_size))
		{
			csrSuccessful = false;
			printf("CSR allocation is not possible with size %zd\n", csr_size);
		}

		if (csrSuccessful)
		{
			/*printf("Initialize new faimGraph from CSR\n");*/
			// One of 2 choices
			//   New does not fit in memory while old is in memory, retire to CSR -> init anew
			//   Retire to CSR and back to Host

			int block_size = 256;
			int grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;
			size_t number_adjacency{ 0 };

			vertex_t* offset = reinterpret_cast<vertex_t*>(csr_helper);
			vertex_t* adjacency = offset + (memory_manager->next_free_vertex_index + 1);

			number_adjacency = memory_manager->numberEdgesInMemory(offset, true);

			if (number_adjacency != memory_manager->number_edges)
			{
				printf("Number Adjacency %lu != Number Edges %u\n", number_adjacency, memory_manager->number_edges);
				exit(-1);
			}

			faimGraphGeneral::d_faimGraphToCSR<VertexData, EdgeData> << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
				memory_manager->d_data,
				adjacency,
				offset,
				memory_manager->page_size);

			// Release old faimGraph
			HANDLE_ERROR(cudaFree(memory_manager->d_memory));

			// Start new setup
			memory_manager->total_memory = new_size;
			memory_manager->free_memory = new_size;
			memory_manager->initialized = false;
			if (cudaSuccess == cudaMalloc((void **)&(memory_manager->d_memory), new_size))
			{
				initializeMemory(offset, adjacency, memory_manager->next_free_vertex_index);
			}
			else
			{
				printf("Retire to host with a CSR\n");
				host_csr.alloc(memory_manager->next_free_vertex_index, memory_manager->next_free_vertex_index, memory_manager->number_edges, false);
				// Retire CSR back to host
				HANDLE_ERROR(cudaMemcpy(host_csr.row_offsets.get(), offset, sizeof(vertex_t) * (memory_manager->next_free_vertex_index + 1), cudaMemcpyDeviceToHost));
				HANDLE_ERROR(cudaMemcpy(host_csr.col_ids.get(), adjacency, sizeof(vertex_t) * memory_manager->number_edges, cudaMemcpyDeviceToHost));
			}
		}
		else
		{
			printf("Retire to device\n");
			// Couldn't even allocate a csr structure to hold the graph, retire back to the host
			// This is obviously the worst case and very slow

			std::unique_ptr<memory_t[]> host_faimGraph(std::make_unique<memory_t[]>(memory_manager->total_memory));
			HANDLE_ERROR(cudaMemcpy(host_faimGraph.get(), memory_manager->d_memory + memory_manager_offset, memory_manager->total_memory - memory_manager_offset, cudaMemcpyDeviceToHost));
			host_csr.alloc(memory_manager->number_vertices, memory_manager->number_vertices, memory_manager->number_edges, false);

			// Transfer host faimGraph to CSR
			VertexData* vertices = reinterpret_cast<VertexData*>(host_faimGraph.get());
			vertex_t accumulated_offset{ 0 };
			for (vertex_t i = 0; i < memory_manager->number_vertices; ++i)
			{
				VertexData& vertex = vertices[i];
				host_csr.row_offsets[i] = accumulated_offset;
				EdgeData* edges = pageAccess<EdgeData>(host_faimGraph.get(), vertex.mem_index, memory_manager->page_size, memory_manager->start_index);
				for (vertex_t j = 0; j < vertex.neighbours; ++j)
				{
					host_csr.col_ids[accumulated_offset + j] = edges[j % memory_manager->edges_per_page].destination;
					if (((j) % (memory_manager->edges_per_page)) == (memory_manager->edges_per_page - 1))
					{
						// Pointer Handling traversal
						pointerHandlingTraverse(edges, host_faimGraph.get(), memory_manager->page_size, memory_manager->edges_per_page, memory_manager->start_index);
					}
				}
				accumulated_offset += vertex.neighbours;
			}
			host_csr.row_offsets[memory_manager->number_vertices] = accumulated_offset;
		}
	}

	return std::move(host_csr);
}


//------------------------------------------------------------------------------
//
std::unique_ptr<aimGraphCSR> faimGraph::verifyGraphStructure(std::unique_ptr<MemoryManager>& memory_manager)
{
	int block_size = KERNEL_LAUNCH_BLOCK_SIZE;
	int grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;
	std::unique_ptr<aimGraphCSR> verifyGraph(new aimGraphCSR(memory_manager));
	// d_mem_requirement赋值neighbours
	size_t num_edges = memory_manager->numberEdgesInMemory(verifyGraph->d_mem_requirement, true);
	void* allocation{ nullptr };
	HANDLE_ERROR(cudaMalloc(&allocation, num_edges * sizeof(vertex_t)));
	verifyGraph->d_adjacency = reinterpret_cast<vertex_t*>(allocation);

	// Prefix scan on d_mem_requirement to get correct memory offsets
	HANDLE_ERROR(cudaMemcpy(verifyGraph->d_offset, verifyGraph->d_mem_requirement, sizeof(vertex_t) * (memory_manager->next_free_vertex_index + 1), cudaMemcpyDeviceToDevice));

	// Copy offsets to host
	HANDLE_ERROR(cudaMemcpy(verifyGraph->h_offset,
		verifyGraph->d_offset,
		sizeof(vertex_t) * (memory_manager->next_free_vertex_index + 1),
		cudaMemcpyDeviceToHost));

	verifyGraph->number_edges = num_edges;

	// Allocate memory for adjacency
	verifyGraph->scoped_mem_access_counter.alterSize(sizeof(vertex_t) * num_edges);

	verifyGraph->h_adjacency = (vertex_t*)malloc(sizeof(vertex_t) * num_edges);

	faimGraphGeneral::d_faimGraphToCSR<VertexData, EdgeData> << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
		memory_manager->d_data,
		verifyGraph->d_adjacency,
		verifyGraph->d_offset,
		memory_manager->page_size);

	// Copy adjacency to host
	HANDLE_ERROR(cudaMemcpy(verifyGraph->h_adjacency,
		verifyGraph->d_adjacency,
		sizeof(vertex_t) * num_edges,
		cudaMemcpyDeviceToHost));


	// Print adjacency at certain index
	//index_t index = 888;
	//index_t start_index = verifyGraph->h_offset[index];
	//index_t end_index = verifyGraph->h_offset[index + 1];
	//std::cout << "\n############################################################\n";
	//std::cout << "Print adjacency for index " << index << " with " << (end_index - start_index) << " neighbours" << std::endl;
	//for(size_t i = start_index, j = 0; i < end_index; ++i, ++j)
	//{
	//  std::cout << verifyGraph->h_adjacency[i] << " | ";
	//  if(((j) % (memory_manager->edges_per_page)) == (memory_manager->edges_per_page - 1))
	//  {
	//    std::cout << std::endl;
	//  }
	//}
	//std::cout << std::endl;
	//std::cout << "############################################################\n\n";

	return std::move(verifyGraph);
}

//------------------------------------------------------------------------------
//
bool h_CompareGraphs(vertex_t* adjacency_prover,
	vertex_t* adjacency_verifier,
	vertex_t* offset_prover,
	vertex_t* offset_verifier,
	int number_vertices,
	int number_edges,
	bool duplicate_check)
{
	for (int tid = 0; tid < number_vertices; ++tid)
	{
		// Compare offset
		int offset_prover_tid, offset_verifier_tid;
		if (tid != number_vertices - 1)
		{
			offset_prover_tid = offset_prover[tid + 1];
			offset_verifier_tid = offset_verifier[tid + 1];
		}
		else
		{
			offset_prover_tid = number_edges;
			offset_verifier_tid = offset_verifier[tid + 1];
		}

		if (offset_prover_tid != offset_verifier_tid)
		{
			std::cout << "[Offset] Src-Vertex: " << tid << " | [Device]faimGraph: " << offset_prover_tid << " | [Host]Host-Graph: " << offset_verifier_tid << std::endl;
		}

		int offset;
		int neighbours;

		if (offset_verifier_tid >= offset_prover_tid)
		{
			offset = offset_verifier[tid];
			neighbours = offset_verifier_tid - offset;
			for (int j = 0; j < neighbours; ++j)
			{
				bool found_match = false;
				vertex_t search_item = adjacency_verifier[offset + j];
				for (int k = 0; k < neighbours; ++k)
				{
					if (adjacency_prover[offset + k] == search_item)
					{
						found_match = true;
						break;
					}
				}
				if (found_match == false)
				{
					std::cout << "Host-Graph >= faimGraph" << std::endl;
					std::cout << "[Adjacency] Src-Vertex: " << tid << " and search_item: " << search_item << std::endl;
					if (tid != number_vertices - 1)
					{
						std::cout << "\n[DEVICE] Neighbours: " << offset_prover_tid - offset_prover[tid] << std::endl;
					}

					std::cout << "[DEVICE]faimGraph-List:\n";
					for (int l = 0; l < neighbours; ++l)
					{
						std::cout << adjacency_prover[offset + l] << " | ";
					}
					if (tid != number_vertices - 1)
					{
						std::cout << "\n[HOST] Neighbours: " << offset_verifier_tid - offset_verifier[tid] << std::endl;
					}

					std::cout << "[HOST]Host-Graph-List:\n";
					for (int l = 0; l < neighbours; ++l)
					{
						std::cout << adjacency_verifier[offset + l] << " | ";
					}
					std::cout << std::endl;
					return false;
				}
			}
			if (duplicate_check)
			{
				// Duplicate check
				offset = offset_prover[tid];
				neighbours = offset_prover_tid - offset;
				for (int j = 0; j < neighbours - 1; ++j)
				{
					for (int k = j + 1; k < neighbours; ++k)
					{
						if (adjacency_prover[offset + j] == adjacency_prover[offset + k])
						{
							std::cout << "DUPLICATE: " << adjacency_prover[offset + j] << std::endl;
							return false;
						}
					}
				}
			}
		}
		else
		{
			offset = offset_prover[tid];
			neighbours = offset_prover_tid - offset;
			for (int j = 0; j < neighbours; ++j)
			{
				bool found_match = false;
				vertex_t search_item = adjacency_prover[offset + j];
				for (int k = 0; k < neighbours; ++k)
				{
					if (adjacency_verifier[offset + k] == search_item)
					{
						found_match = true;
						break;
					}
				}
				if (found_match == false)
				{
					std::cout << "faimGraph > Host-Graph" << std::endl;
					std::cout << "[Adjacency] Src-Vertex: " << tid << " and search_item: " << search_item << std::endl;

					std::cout << "\n[DEVICE] Neighbours: " << offset_prover_tid - offset_prover[tid] << std::endl;

					std::cout << "[DEVICE]faimGraph-List:\n";
					for (int l = 0; l < neighbours; ++l)
					{
						std::cout << adjacency_prover[offset + l] << " | ";
					}

					std::cout << "\n[HOST] Neighbours: " << offset_verifier_tid - offset_verifier[tid] << std::endl;

					std::cout << "[HOST]Host-Graph-List:\n";
					for (int l = 0; l < neighbours; ++l)
					{
						std::cout << adjacency_verifier[offset + l] << " | ";
					}
					std::cout << std::endl;
					return false;
				}
			}
			if (duplicate_check)
			{
				// Duplicate check
				offset = offset_prover[tid];
				neighbours = offset_prover_tid - offset;
				for (int j = 0; j < neighbours - 1; ++j)
				{
					for (int k = j + 1; k < neighbours; ++k)
					{
						if (adjacency_prover[offset + j] == adjacency_prover[offset + k])
						{
							std::cout << "DUPLICATE: " << adjacency_prover[offset + j] << std::endl;
							return false;
						}
					}
				}
			}
		}
	}
	return true;
}

//------------------------------------------------------------------------------
//
bool h_CompareGraphs(vertex_t* adjacency_prover,
	vertex_t* adjacency_verifier,
	vertex_t* offset_prover,
	vertex_t* offset_verifier,
	matrix_t* matrix_values_prover,
	matrix_t* matrix_values_verifier,
	int number_vertices,
	int number_edges,
	bool duplicate_check)
{
	for (int tid = 0; tid < number_vertices; ++tid)
	{
		// Compare offset
		int offset_prover_tid, offset_verifier_tid;
		if (tid != number_vertices - 1)
		{
			offset_prover_tid = offset_prover[tid + 1];
			offset_verifier_tid = offset_verifier[tid + 1];
		}
		else
		{
			offset_prover_tid = number_edges;
			offset_verifier_tid = offset_verifier[tid + 1];
		}

		if (offset_prover_tid != offset_verifier_tid)
		{
			std::cout << "[Offset] Src-Vertex: " << tid << " | [Device]faimGraph: " << offset_prover_tid << " | [Host]Host-Graph: " << offset_verifier_tid << std::endl;
		}

		int offset;
		int neighbours;

		if (offset_verifier_tid >= offset_prover_tid)
		{
			offset = offset_verifier[tid];
			neighbours = offset_verifier_tid - offset;
			for (int j = 0; j < neighbours; ++j)
			{
				bool found_match = false;
				vertex_t search_item = adjacency_verifier[offset + j];
				matrix_t search_value = matrix_values_verifier[offset + j];
				for (int k = 0; k < neighbours; ++k)
				{
					if (adjacency_prover[offset + k] == search_item && matrix_values_prover[offset + k] == search_value)
					{
						found_match = true;
						break;
					}
				}
				if (found_match == false)
				{
					std::cout << "Host-Graph >= faimGraph" << std::endl;
					std::cout << "[Adjacency] Src-Vertex: " << tid << " and search_item: " << search_item << " and search_value: " << search_value << std::endl;
					if (tid != number_vertices - 1)
					{
						std::cout << "\n[DEVICE] Neighbours: " << offset_prover_tid - offset_prover[tid] << std::endl;
					}

					std::cout << "[DEVICE]faimGraph-List:\n";
					for (int l = 0; l < neighbours; ++l)
					{
						std::cout << "(" << adjacency_prover[offset + l] << " | " << matrix_values_prover[offset + l] << ")" << " | ";
					}
					if (tid != number_vertices - 1)
					{
						std::cout << "\n[HOST] Neighbours: " << offset_verifier_tid - offset_verifier[tid] << std::endl;
					}

					std::cout << "[HOST]Host-Graph-List:\n";
					for (int l = 0; l < neighbours; ++l)
					{
						std::cout << "(" << adjacency_verifier[offset + l] << " | " << matrix_values_verifier[offset + l] << ")" << " | ";
					}
					std::cout << std::endl;
					return false;
				}
			}
		}
		else
		{
			offset = offset_prover[tid];
			neighbours = offset_prover_tid - offset;
			for (int j = 0; j < neighbours; ++j)
			{
				bool found_match = false;
				vertex_t search_item = adjacency_prover[offset + j];
				matrix_t search_value = matrix_values_prover[offset + j];
				for (int k = 0; k < neighbours; ++k)
				{
					if (adjacency_verifier[offset + k] == search_item && matrix_values_verifier[offset + k] == search_value)
					{
						found_match = true;
						break;
					}
				}
				if (found_match == false)
				{
					std::cout << "faimGraph > Host-Graph" << std::endl;
					std::cout << "[Adjacency] Src-Vertex: " << tid << " and search_item: " << search_item << std::endl;

					std::cout << "\n[DEVICE] Neighbours: " << offset_prover_tid - offset_prover[tid] << std::endl;

					std::cout << "[DEVICE]faimGraph-List:\n";
					for (int l = 0; l < neighbours; ++l)
					{
						std::cout << "(" << adjacency_prover[offset + l] << " | " << matrix_values_prover[offset + l] << ")" << " | ";
					}

					std::cout << "\n[HOST] Neighbours: " << offset_verifier_tid - offset_verifier[tid] << std::endl;

					std::cout << "[HOST]Host-Graph-List:\n";
					for (int l = 0; l < neighbours; ++l)
					{
						std::cout << "(" << adjacency_verifier[offset + l] << " | " << matrix_values_verifier[offset + l] << ")" << " | ";
					}
					std::cout << std::endl;
					return false;
				}
			}
		}
	}
	return true;
}

//------------------------------------------------------------------------------
//
bool faimGraph::compareGraphs(std::unique_ptr<GraphParser>& graph_parser,
	std::unique_ptr<aimGraphCSR>& verify_graph,
	bool duplicate_check)
{
	int number_of_vertices = memory_manager->next_free_vertex_index;
	const auto prover_offset = verify_graph->h_offset;
	const auto prover_adjacency = verify_graph->h_adjacency;
	const auto verifier_offset = graph_parser->getOffset().data();
	const auto verifier_adjacency = graph_parser->getAdjacency().data();

	return h_CompareGraphs(prover_adjacency, verifier_adjacency, prover_offset, verifier_offset, number_of_vertices, verify_graph->number_edges, duplicate_check);
}

