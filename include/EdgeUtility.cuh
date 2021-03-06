//------------------------------------------------------------------------------
// filename: EdgeUtility.cuh
// Author : @SH
// Description: 
// Date：2019.5.14
//
//------------------------------------------------------------------------------
//
#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <thrust/device_vector.h>
#include <cstddef>
#include <cstdio>
#include <climits>
#include <algorithm>
#include "faimGraph.h"
#include "EdgeUpdate.h"
#include "MemoryManager.h"
#include "ConfigurationParser.h"
#include "Definitions.h"
#define SHAREDSIZE 100

struct sortf{
  __host__ __device__
  bool operator() (const EdgeUpdate& lhs, const EdgeUpdate& rhs){
           if (lhs.neighbor == rhs.neighbor){
               if (lhs.source == rhs.source ){return lhs.update.destination < rhs.update.destination;}
           return lhs.source < rhs.source;
          }
          return lhs.neighbor < rhs.neighbor;
   
      }
  };

//------------------------------------------------------------------------------
// Device funtionality
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
// @SH set up edge update data neighbor
__global__ void d_setupneighbor(memory_t* memory,EdgeUpdate* edge_update_data,
  int batch_size) 
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid > batch_size) {
    return;
  }
  
  VertexData* vertices = (VertexData*)memory;
  
  edge_update_data[tid].neighbor = vertices[edge_update_data[tid].source].neighbours;
  //printf("edge_update_data[%d].source = %d\tneighbor = %d\n", tid, edge_update_data[tid].source, edge_update_data[tid].neighbor);

}

//------------------------------------------------------------------------------
// @SH set 1 to every first different update source corresponding position
__global__ void d_setupscanhelper(EdgeUpdate* edge_update_data,
  int batch_size, 
  index_t* scanhelper) 
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid > batch_size) return;
  //printf("edge_update_data[%d] = %d\n", tid, edge_update_data[tid].neighbor);
  scanhelper[tid] = 0;
  if (tid == 0) {
    scanhelper[tid] = 1;
    return;
  }
  if (edge_update_data[tid].source != edge_update_data[tid - 1].source) {
    scanhelper[tid] = 1;
  }
}

//------------------------------------------------------------------------------
// @SH set workitem and thl_n thm_n th0
__global__ void d_updateWorkItem(index_t edges_per_page, 
  EdgeUpdate* edge_update_data,
  int batch_size, 
  memory_t* memory, 
  workitem* work, 
  index_t* scanhelper, 
  index_t thl, 
  index_t thm, 
  index_t* thl_n, 
  index_t* thm_n, 
  index_t *th0) 
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;;
  if (tid > batch_size)
    return;

  EdgeUpdate edge_update = edge_update_data[tid];
  VertexData* vertices = (VertexData*)memory;
  index_t index = edge_update.source;
  VertexData vertex = vertices[index];
  
  int pos = scanhelper[tid];

  if (tid == 0) {
    work[pos].index = index;
    work[pos].neighbour = vertex.neighbours;
    work[pos].off = tid;
    *th0 = scanhelper[batch_size];  
    work[scanhelper[batch_size]].index = DELETIONMARKER;
    work[scanhelper[batch_size]].off = batch_size;

    return;
  }

  int s1 = edge_update_data[tid - 1].source;
  VertexData v2 = vertices[s1];
  
  if (index != s1)
  {
    work[pos].index = index;
    work[pos].neighbour = vertex.neighbours;
    work[pos].off = tid ;
    
    if (vertex.neighbours < thl && v2.neighbours >= thl) {
      *thl_n = pos;
    }
    else if (vertex.neighbours < thm && v2.neighbours >= thm) {
      *thm_n = pos;
    }
    return ;
  }
  
  return ;
}

#define MULTIPLICATOR 4
//------------------------------------------------------------------------------
// @SH check duplicate in update batch
__global__ void d_duplicateCheckingInSortedBatch(EdgeUpdate* edge_update_data,
  int batch_size)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= batch_size)
    return;

  EdgeUpdate edge_update = edge_update_data[tid];
  EdgeUpdate compare_element;
  while (tid < batch_size)
  {
    compare_element = edge_update_data[++tid];
    if ((edge_update.source == compare_element.source) && (edge_update.update.destination == compare_element.update.destination))
    {
      atomicExch(&(edge_update_data[tid].update.destination), DELETIONMARKER);
    }
    else
      break;
  }
  return;
}

//------------------------------------------------------------------------------
// @SH check upate data duplicate in adjacency
__global__ void d_process_kernel(MemoryManager* memory_manager,
  EdgeUpdate* UpdateData,
  workitem* work,
  index_t* pageindex,
  index_t* pageindex_off,
  memory_t* memory,
  index_t* deletion_helper,
  int workoffset)
{

  int bid = blockIdx.x; 
  bid += workoffset;      /// compute process workitem index

  workitem temp = work[bid];
  if (temp.index == DELETIONMARKER) return;

  /// compute pageindex related data 
  int tid = threadIdx.x;      
  int page_start = pageindex_off[tid];
  int page_num = pageindex_off[tid + 1] - page_start;
  int edges_per_page = memory_manager->edges_per_page ;

 
  int st = temp.off;                      /// get update data index
  int offset = work[bid + 1].off - st;

  int warp_id = threadIdx.x / 32;         /// process page serial number
  int t_step = (blockDim.x + 31) / 32;

  __shared__ AdjacencyIterator adjacency_iterator[1];


  while (warp_id < page_num) {
    tid = threadIdx.x % 32;
    adjacency_iterator[0].setIterator(pageAccess<EdgeData>(memory, pageindex[page_start + warp_id], memory_manager->page_size, memory_manager->start_index));

    while (tid < edges_per_page) {
        if (tid < temp.neighbour){
        int dest = adjacency_iterator[0].getDestinationAt(tid);
        d_binarySearch(UpdateData, dest, st , offset, deletion_helper);  /// TODO: 待解决搜索范围内有DELETION标识时二分范围变化问题
      }
     
      tid += 32;
    }
    warp_id += t_step;
  }
}


//------------------------------------------------------------------------------
// Host funtionality
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
// @SH setupPhrase: set neighbor + sort update data main by neighbor
void EdgeUpdateManager::setupPhrase(std::unique_ptr<MemoryManager>& memory_manager, int batch_size, int grid_size, int block_size)
{
  d_setupneighbor << <grid_size, block_size >> > (memory_manager->d_data, 
    updates->d_edge_update, 
    batch_size);

  if (batch_size <= 100000) {
    std::sort(updates->edge_update.begin(),updates->edge_update.end());
    HANDLE_ERROR(cudaMemcpy(updates->d_edge_update,
      updates->edge_update.data(),
      sizeof(EdgeUpdate) * batch_size,
      cudaMemcpyHostToDevice));
  } else {
    // thrust sort
    thrust::device_ptr<EdgeUpdate> th_edge_updates(updates->d_edge_update);
    thrust::sort(th_edge_updates, th_edge_updates + batch_size, sortf());   /// neighbor decreasing order source decreasing order destination increasing order
  }
}

//------------------------------------------------------------------------------
// @SH setupscanhelper: set scanhelper with 1 in every first different update source corresponding position
void EdgeUpdateManager::setupscanhelper(std::unique_ptr<MemoryManager>& memory_manager, int batch_size, index_t* scanhelper) 
{
  int block_size = 256;
  int grid_size = batch_size / block_size + 1;
  d_setupscanhelper << <grid_size, block_size >> > (updates->d_edge_update, 
    batch_size, 
    scanhelper);
}

//------------------------------------------------------------------------------
// @SH updateWorkItem: set workitem and thl_n thm_n th0
void EdgeUpdateManager::updateWorkItem(std::unique_ptr<MemoryManager>& memory_manager,
  const std::shared_ptr<Config>& config,
  const std::unique_ptr<EdgeUpdatePreProcessing>& preprocessed,
  index_t* scanhelper,
  index_t* thm,
  index_t* ths,
  index_t* th0)
{
  int batch_size = updates->edge_update.size();
  int block_size = 256; // config->testruns_.at(config->testrun_index_)->params->init_launch_block_size_;
  int grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;

  // Duplicate Checking in Graph (sorted updates)
  block_size = 256;
  grid_size = (batch_size / block_size) + 1;
  d_updateWorkItem << < grid_size, block_size >> > (memory_manager->edges_per_page,
    updates->d_edge_update,
    batch_size, 
    memory_manager->d_data, 
    preprocessed->d_different_vertexes, 
    scanhelper, 
    10 * memory_manager->edges_per_page,  /// thl
    1 * memory_manager->edges_per_page,   /// thm
    thm, 
    ths, 
    th0);

  cudaDeviceSynchronize();
  // printf("updateworkitem is done\n");
  return;
}

//------------------------------------------------------------------------------
// @SH edgeUpdateDuplicateCheckingByBlocksize: check duplicate in batch and adjacency
void EdgeUpdateManager::edgeUpdateDuplicateCheckingByBlocksize(std::unique_ptr<MemoryManager>& memory_manager,
  const std::shared_ptr<Config>& config,
  const std::unique_ptr<EdgeUpdatePreProcessing>& preprocessed, 
  index_t* d_pageindex, 
  index_t* d_pageindex_off,
  index_t* thm, 
  index_t* ths, 
  index_t* th0, 
  index_t* deletehelp) 
{
  float time_diff;
  float time_check_arry = 0;
  cudaEvent_t in_start, in_stop;
  
  int batch_size = updates->edge_update.size();

  /// 1. Duplicate Checking in Batch (sorted updates)
  int block_size = 256;
  int grid_size = (batch_size / block_size) + 1;
  printf("start check duplicate in batch\n");
  start_clock(in_start, in_stop);
    d_duplicateCheckingInSortedBatch << < grid_size, block_size >> > (updates->d_edge_update, batch_size);
  time_diff = end_clock(in_start, in_stop);
  time_check_arry += time_diff;
  printf("array check duplication in adj is : %f\n",time_check_arry);

  /// 2. Duplicate Checking in Adj by Different Block Size
  cudaStream_t stream[3];
  cudaEvent_t start[3], stop[3];
  float elapseTime[3] = { 0 };

  for (int i = 0; i < 3; ++i) {
    cudaStreamCreate(&stream[i]);
  }

  for (int i = 0; i < 3; ++i) {
    cudaEventCreate(&start[i]);
    cudaEventCreate(&stop[i]);
  }

  // s m l blocksize = pageNum * warpsize
  int pageNum[] = { 10, 2, 1};
  index_t h_th[4];
  cudaMemcpy(&h_th[1], thm, sizeof(index_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_th[2], ths, sizeof(index_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_th[3], th0, sizeof(index_t), cudaMemcpyDeviceToHost);
  h_th[0] = 0;

#ifdef DEBUG_EDGEINSERTION  
  for (int i = 0; i < 4; ++i) {
    printf("h_th[%d] = %d\n", i, h_th[i]); 
  }
#endif

  printf("start check duplicate in adjacency\n");
  for (int i = 0; i < 3; ++i) {
    if (h_th[i + 1] == 0) {        /// no data in this range no need to issue kernel
      continue;
    }
    block_size = pageNum[i] * 32;
    grid_size = h_th[i + 1] - h_th[i];
    cudaEventRecord(start[i], stream[i]);
      d_process_kernel<< < grid_size, block_size, 0, stream[i] >> >((MemoryManager*)memory_manager->d_memory,
            updates->d_edge_update,
            preprocessed->d_different_vertexes,
            d_pageindex,
            d_pageindex_off,
            memory_manager->d_data,
            deletehelp,
            h_th[i]);  
    cudaEventRecord(stop[i], stream[i]);
  }
  /// compute time every range record
  for (int i = 0; i < 3; ++i) {
    if(h_th[i+1]==0){           /// no issue kernel so no event record
      continue;
    }
    cudaEventSynchronize(stop[i]);
    cudaEventElapsedTime(&elapseTime[i], start[i], stop[i]);
    printf("@%d check duplication in adj is : %f\n", i, elapseTime[i]);
  }
  
  /// 3. release resources
  for (int i = 0; i < 3; ++i) {
    cudaStreamDestroy(stream[i]);
  }
  for (int i = 0; i < 3; ++i) {
    cudaEventDestroy(start[i]);
    cudaEventDestroy(stop[i]);
  }
  cudaDeviceSynchronize();
 
  return;
}

