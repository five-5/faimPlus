//------------------------------------------------------------------------------
// filename: EdgeInsertion.cuh
// Author : @SH
// Description: the main body of faim edgeinsertion
// Date：2019.5.13
//
//------------------------------------------------------------------------------
//

#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cstddef>
#include <stdio.h>
#include <cuda_runtime.h>

#include "EdgeUpdate.h"
#include "faimGraph.h"
#include "MemoryManager.h"
#include "ConfigurationParser.h"

namespace faimGraphEdgeInsertion
{
	//------------------------------------------------------------------------------
	// set pageindex_off value
	__global__ void d_get_pageindex_off (memory_t* memory,
		int valid_size,
		vertex_t pageindexes_per_page,
		index_t* pageindex_off) 
	{
		
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if (tid >= valid_size) {
			return;	
		}
	
		VertexData* vertices = (VertexData*)memory;	
		vertex_t neighbours = vertices->neighbours;								
		pageindex_off[tid] = ( neighbours + pageindexes_per_page - 1) / pageindexes_per_page; 
		
	}

	//------------------------------------------------------------------------------
	// set pageindex value
	__global__ void d_fillIndexVertexCentric(MemoryManager* memory_manager,
											memory_t* memory,
											int valid_size,
											vertex_t pageindexes_per_page,
											index_t* pageindex,
											index_t* pageindex_off)
	{

		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if (tid >= valid_size) {
			return;	
		}

		VertexData* vertices = (VertexData*)memory;	
		int begin = pageindex_off[tid];								
		int offset = pageindex_off[tid + 1] - begin;
		index_t cur_pageindex = vertices[tid].mem_index;
		for (int i = 0; i < offset; ++i) {
			PageIndexIterator pageindex_iterator(pageAccess<index_t>(memory, cur_pageindex, memory_manager->page_size, memory_manager->start_page_index));
			pageindex[begin + i] = cur_pageindex;
			cur_pageindex = pageindex_iterator.getPageIndexNext(pageindexes_per_page);
		}

	}

	//------------------------------------------------------------------------------
	// d_pageAllocate
	__forceinline__ __device__ void d_pageAllocate(MemoryManager* memory_manager, index_t* edge_block_index_ptr, PAGETYPE type)
	{
		index_t edge_block_index;
#ifdef QUEUING			/// define in EdgeUpdate.h
				if (memory_manager->d_page_queue.dequeue(edge_block_index))
				{
					// We got something from the queue
					*edge_block_index_ptr = edge_block_index;
				}
				else
				{
#endif
					// Queue is currently empty
					if (type == PAGETYPE::ADJACENCY) {
						*edge_block_index_ptr = atomicAdd(&(memory_manager->next_free_page), 1);						
					} else if (type == PAGETYPE::PAGEINDEX) {
						*edge_block_index_ptr = atomicAdd(&(memory_manager->next_free_pageindex_page), 1);		
					}
#ifdef ACCESS_METRICS
					atomicAdd(&(memory_manager->access_counter), 1);
#endif

#ifdef QUEUING
				}
#endif
	}

	//------------------------------------------------------------------------------
	// d_calcPageindexCapacity
	__forceinline__ __device__ index_t d_calcPageindexCapacity(vertex_t capacity, vertex_t edges_per_page, vertex_t pageindexes_per_page)
	{
		return ( capacity / edges_per_page + pageindexes_per_page - 1 ) / pageindexes_per_page * pageindexes_per_page;
	}

	//------------------------------------------------------------------------------
	// edgeInsertionVertexCentric
	__global__ void d_edgeInsertionVertexCentric(MemoryManager* memory_manager,
		memory_t* memory,
		int page_size,
		EdgeUpdate* edge_update_data,
		int valid_size,
		workitem* work)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= valid_size)
		return;

		index_t index_offset = work[tid].off;
		index_t number_updates = work[tid + 1].off - index_offset;

		if (number_updates == 0)  
		return;

		// Now just threads that actually work on updates should be left, tid corresponds to the src vertex that is being modified
		// Gather pointer
		index_t src = work[tid].index;
		vertex_t edges_per_page = memory_manager->edges_per_page;
		vertex_t pageindexes_per_page = memory_manager->pageindexes_per_page;
		VertexData* vertices = (VertexData*)memory;
		int neighbours = vertices[src].neighbours;
		int capacity = vertices[src].capacity;
		int pageindex_numbers = neighbours / edges_per_page;
		int pageindex_capacity = d_calcPageindexCapacity(capacity, edges_per_page, pageindexes_per_page);

		
		PageIndexIterator pageindex_iterator(pageAccess<index_t>(memory, vertices[src].mem_index, page_size, memory_manager->start_page_index));
		index_t fresh_pageindex_position = pageindex_numbers;
		index_t fresh_position = neighbours - (pageindex_numbers - 1) * edges_per_page;
		
		/// TODO: to check the index whether need to sub one
		/// 1.1 pageindex iterator at  
		index_t adjacency_pageindex = pageindex_iterator.at(fresh_pageindex_position - 1, memory, page_size, memory_manager->start_page_index, pageindexes_per_page);

		/// 1.2 set adjacency_iterator to the last item in last page
		AdjacencyIterator adjacency_iterator(pageAccess<EdgeData>(memory, adjacency_pageindex, page_size, memory_manager->start_index));
		adjacency_iterator.advanceIteratorToIndexAbsolute(edges_per_page, memory, page_size, memory_manager->start_index, fresh_position, fresh_position, edges_per_page);
		
		while (true)
		{
			/// full of the page and stop go next
			while (neighbours < capacity && number_updates > 0)
			{
				/// first check DELETION
				if(edge_update_data[index_offset + (number_updates - 1)].update.destination == DELETIONMARKER)
				{
					--number_updates;
					continue;
				}

				updateAdjacency(adjacency_iterator.getIterator(), edge_update_data[index_offset + (number_updates - 1)], edges_per_page);
				++adjacency_iterator;
				--number_updates;
				++neighbours;
			}

			/// while stop when updates number is zero
			if (number_updates == 0)
			{
				// Then we are done
				vertices[src].neighbours = neighbours;
				vertices[src].capacity = capacity;
#ifdef CLEAN_PAGE      /// define in EdgeUpdate.h
				int number_pageindexes =  neighbours / edges_per_page;
				while (neighbours < capacity)
				{
					// Set the rest of the new block to DELETIONMARKERS
					setDeletionMarker(adjacency_iterator.getIterator(), edges_per_page);
					++adjacency_iterator;
					++neighbours;              
				}
				
				/// set the rest of the pageindex to DELETIONMARKERS
				while (number_pageindexes < d_calcPageindexCapacity(capacity, edges_per_page, pageindexes_per_page))
				{
					setPageIndexDeletionMarker(pageindex_iterator.getIterator(), pageindexes_per_page);
					++pageindex_iterator;
					++number_pageindexes;
				}
#endif
				break;
			}
			else
			{
				// We need to get a new page and start all over again
				// Set index to next block and then reset adjacency list
				
				/// return (index_t*)it
				index_t* edge_block_index_ptr = adjacency_iterator.getPageIndexPtr(edges_per_page);
				d_pageAllocate(memory_manager, edge_block_index_ptr, PAGETYPE::ADJACENCY);
				adjacency_iterator.setIterator(pageAccess<EdgeData>(memory, *edge_block_index_ptr, page_size, memory_manager->start_index));
				capacity += edges_per_page;

				if (pageindex_iterator.pageIsFull(memory_manager->pageindexes_per_page, memory_manager->start_page_index)) {
					
					index_t* pageindex_block_index_ptr = pageindex_iterator.getPageIndexPtr(pageindexes_per_page);
					d_pageAllocate(memory_manager, edge_block_index_ptr, PAGETYPE::PAGEINDEX);
					pageindex_iterator.setIterator(pageAccess<index_t>(memory, *pageindex_block_index_ptr, page_size, memory_manager->start_page_index));
				} else {
					updatePageIndex(pageindex_iterator.getIterator(), *edge_block_index_ptr);
					++pageindex_iterator;
				}
			}
		}

		return;
	}
}  // end of namespace faimGraphEdgeInsertion


//------------------------------------------------------------------------------
// Host funtionality
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
//  deviceEdgeInsertion
void EdgeUpdateManager::deviceEdgeInsertion(std::unique_ptr<MemoryManager>& memory_manager,
	const std::shared_ptr<Config>& config)
{
	// define record time related variables 
	float time_setup = 0;
	float time_preprocess = 0;
	float time_scanhelper = 0;
	float time_update_workitem = 0;
	float time_pageindex_off = 0;
	float time_pageindex = 0;
	float time_check = 0;
	float time_insert = 0;
	float time_diff = 0;
	cudaEvent_t in_start, in_stop;
	
	int batch_size = updates->edge_update.size();
	int block_size = config->testruns_.at(config->testrun_index_)->params->insert_launch_block_size_;
	int grid_size = (batch_size / block_size) + 1;

	/// allocate batch_size in memorymanager [decrease free memory]
	ScopedMemoryAccessHelper scoped_mem_access_counter(memory_manager.get(), sizeof(EdgeUpdate) *  batch_size);

	// Copy updates to device
	TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), memory_manager->next_free_vertex_index, sizeof(VertexData));
	updates->d_edge_update = temp_memory_dispenser.getTemporaryMemory<EdgeUpdate>(batch_size);  /// return the location next_free_vertex with batch_size  

	HANDLE_ERROR(cudaMemcpy(updates->d_edge_update,
		updates->edge_update.data(),
		sizeof(EdgeUpdate) * batch_size,
		cudaMemcpyHostToDevice));
	
	/// allocate deletehelp	
	index_t* deletehelp = temp_memory_dispenser.getTemporaryMemory<index_t>(batch_size);	

	ConfigurationParameters::DeletionVariant deletion_variant = config->testruns_.at(config->testrun_index_)->params->deletion_variant_;

	// Insert Edges using the vertex centric approach ( 1 thread / 1 vertex)
	if (config->testruns_.at(config->testrun_index_)->params->update_variant_ == ConfigurationParameters::UpdateVariant::VERTEXCENTRIC)
	{
		/// grid_size * block_size = vertex_size -> ( 1 thread / 1 vertex)
		grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;

		/// 1. perfect the update data
		/// setupPhrase: set neighbor + sort update data main by neighbor
		printf("setupphrase\n");
		start_clock(in_start, in_stop);
			setupPhrase(memory_manager, batch_size, grid_size, block_size);   /// include thrust::sort
		cudaDeviceSynchronize();
		time_diff = end_clock(in_start, in_stop);
		time_setup += time_diff;

		/// 2. construct and set update data related data
		/// 2.1 edgeUpdatePreprocessing: allocate workitem 
		printf("Preprocessing\n");
		start_clock(in_start, in_stop);
			auto preprocessed = std::make_unique<EdgeUpdatePreProcessing>(static_cast<uint32_t>(memory_manager->next_free_vertex_index),
			static_cast<vertex_t>(updates->edge_update.size()),
			memory_manager,
			static_cast<size_t>(sizeof(VertexData)));  
		time_diff = end_clock(in_start, in_stop);
		time_preprocess += time_diff;

		/// 2.2 set workitem
		printf("setWorkItem\n");
		
		/// @SH feel the old's logic have some problems so I fix it as follows
		/// allocate and set workitem related data
		TemporaryMemoryAccessHeap temp_memory_dispenser_for_insertion(memory_manager.get(), memory_manager->next_free_vertex_index + 1 + batch_size, sizeof(VertexData));
		
		index_t* scanhelper = temp_memory_dispenser_for_insertion.getTemporaryMemory<index_t>(batch_size + 1);
		index_t* thm = temp_memory_dispenser_for_insertion.getTemporaryMemory<index_t>(1);
		index_t* ths = temp_memory_dispenser_for_insertion.getTemporaryMemory<index_t>(1);
		index_t* th0 = temp_memory_dispenser_for_insertion.getTemporaryMemory<index_t>(1);
	
		HANDLE_ERROR(cudaMemset(thm, 0, sizeof(index_t)));
		HANDLE_ERROR(cudaMemset(ths, 0, sizeof(index_t)));
		HANDLE_ERROR(cudaMemset(th0, 0, sizeof(index_t)));

		/// setupscanhelper: set scanhelper with 1 in every first different update source corresponding position
		printf("|-setupscanhelper\n");
		start_clock(in_start, in_stop);
			setupscanhelper(memory_manager, batch_size, scanhelper);
			thrust::device_ptr<index_t> th_scanhelper(scanhelper);
			thrust::exclusive_scan(th_scanhelper, th_scanhelper + batch_size + 1, th_scanhelper);
			cudaDeviceSynchronize();
		time_diff = end_clock(in_start, in_stop);
		time_scanhelper += time_diff;

		/// updateWorkItem: set workitem and thl_n thm_n th0
		printf("|-updateWorkItem\n");
		start_clock(in_start, in_stop);
			updateWorkItem(memory_manager, config, preprocessed, scanhelper, thm, ths, th0);
		time_diff = end_clock(in_start, in_stop);
		time_update_workitem += time_diff;

		/// get work_valid_lenth
		index_t work_valid_lenth = 0;
		HANDLE_ERROR(cudaMemcpy(&work_valid_lenth, scanhelper + batch_size, sizeof(index_t), cudaMemcpyDeviceToHost));
			
#ifdef DEBUG_EDGEINSERTION
		/// print scanhelper 
		// copy from GPU to CPU
		index_t* h_scanhelper = (index_t*)malloc(sizeof(index_t) * (batch_size + 1));
		memset(h_scanhelper, 0, sizeof(index_t) * (batch_size + 1));
		for (int i = 0; i < batch_size + 1; i++) {
			h_scanhelper[i] = 0;
		}
		HANDLE_ERROR(cudaMemcpy(h_scanhelper, scanhelper, (batch_size + 1) * sizeof(index_t),
			cudaMemcpyDeviceToHost));
		// print some data to see copy whether valid
		for (int i = 0; i < 10 && i < batch_size+1; ++i) {
			printf("scanhelper[%d] = %d \n", i, h_scanhelper[i]);
		}
		free(h_scanhelper);

		std::cout << "work_valid_lenth = " << work_valid_lenth << std::endl;
		// copy workitem from device to host
		workitem* h_different_vertexes = (workitem*)malloc(sizeof(workitem) * (batch_size + 2));
		for (int i = 0; i < batch_size + 1; i++) {
			(h_different_vertexes + i)->index = DELETIONMARKER;
			(h_different_vertexes + i)->neighbour = 0;
			(h_different_vertexes + i)->page_num = 0;
		}
		HANDLE_ERROR(cudaMemcpy(h_different_vertexes, preprocessed->d_different_vertexes, (batch_size + 2) * sizeof(workitem),
			cudaMemcpyDeviceToHost));
		for (int i = 0; i < batch_size + 1; ++i) {
			printf("h_different_vertexes[%d].source = %d  neigh = %d  page_num = %d\n",
			i,
			(h_different_vertexes + i )-> index,
			(h_different_vertexes + i )-> neighbour,
			(h_different_vertexes + i )-> page_num);
		}
		free(h_different_vertexes);
#endif

		/// 3. compute pageindex
		/// 3.1 get d_pageindex_off
		grid_size = (work_valid_lenth / block_size) + 1;
		printf("get pageindex off\n");
		index_t *d_pageindex_off = temp_memory_dispenser_for_insertion.getTemporaryMemory<index_t>(work_valid_lenth + 1);
		HANDLE_ERROR(cudaMemset(d_pageindex_off, 0, sizeof(index_t) * (work_valid_lenth + 1)));
		start_clock(in_start, in_stop);
			faimGraphEdgeInsertion::d_get_pageindex_off<< < grid_size, block_size >> > (memory_manager->d_data,
			work_valid_lenth,
			memory_manager->pageindexes_per_page,
			d_pageindex_off);
			thrust::device_ptr<index_t> th_pageindex_off(d_pageindex_off);
			thrust::exclusive_scan(th_pageindex_off, th_pageindex_off + work_valid_lenth + 1, th_pageindex_off);
		cudaDeviceSynchronize();
		time_diff = end_clock(in_start, in_stop);
		time_pageindex_off += time_diff;

		/// get pageindex_nums
		index_t pageindex_nums = 0;
		HANDLE_ERROR(cudaMemcpy(&pageindex_nums, d_pageindex_off + work_valid_lenth, sizeof(index_t), cudaMemcpyDeviceToHost));
	
		/// 3.2 get d_pageindex
		printf("set pageindex\n");
		index_t * d_pageindex = temp_memory_dispenser_for_insertion.getTemporaryMemory<index_t>(pageindex_nums);
		start_clock(in_start, in_stop);
			faimGraphEdgeInsertion::d_fillIndexVertexCentric << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
				memory_manager->d_data,
				work_valid_lenth,
				memory_manager->pageindexes_per_page,
				d_pageindex,
				d_pageindex_off);
		cudaDeviceSynchronize();
		printf("fillindex done\n");
		time_diff = end_clock(in_start, in_stop);
		time_pageindex += time_diff;

		/// 4. check duplicate
		int pageNum = 2;
		block_size = pageNum * 32;
		grid_size = (batch_size / block_size) + 1;
		cudaDeviceSynchronize();
		printf("duplicateCheckingByBlockSize\n");
		start_clock(in_start, in_stop);
		/// edgeUpdateDuplicateCheckingByBlocksize: check duplicate in batch and adjacency
			edgeUpdateDuplicateCheckingByBlocksize(memory_manager, config, preprocessed, d_pageindex, d_pageindex_off, thm, ths, th0, deletehelp);
		time_diff = end_clock(in_start, in_stop);
		time_check += time_diff;
		
		/// 5. edge insertion
		printf("edgeInsertionVertexCentric\n");
		start_clock(in_start, in_stop);
		block_size = config->testruns_.at(config->testrun_index_)->params->insert_launch_block_size_;
		grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;
			faimGraphEdgeInsertion::d_edgeInsertionVertexCentric << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
				memory_manager->d_data,
				memory_manager->page_size,
				updates->d_edge_update,
				work_valid_lenth + 1,
				preprocessed->d_different_vertexes);
		time_diff = end_clock(in_start, in_stop);
		time_insert += time_diff;
	
		printf("vc:time_setup: %f |time_preprocess: %f |time_scanhelper: %f | time_update_workitem: %f |time_pageindex_off: %f |time_pageindex: %f |time_check: %f | time_insert: %f\n", time_setup, time_preprocess, time_scanhelper, time_update_workitem, time_pageindex_off, time_pageindex, time_check, time_insert);
	}
	
	updateMemoryManagerHost(memory_manager);

	return;
}


//------------------------------------------------------------------------------
// √
void faimGraph::edgeInsertion()
{
	edge_update_manager->deviceEdgeInsertion(memory_manager, config);
}