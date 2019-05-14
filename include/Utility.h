//------------------------------------------------------------------------------
// filename: Utility.h
// Author : @SH
// Description: some tool functions
//				1. timekeeping function
//				2. cuda error handle and prop query
//				3. memory access function
// Date：2019.5.8
//
//------------------------------------------------------------------------------
//

#pragma once

#include "Definitions.h"
#include "MemoryLayout.h"

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void queryAndPrintDeviceProperties();
//------------------------------------------------------------------------------
void inline start_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&end));
	HANDLE_ERROR(cudaEventRecord(start, 0));
}
//------------------------------------------------------------------------------
float inline end_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	float time;
	HANDLE_ERROR(cudaEventRecord(end, 0));
	HANDLE_ERROR(cudaEventSynchronize(end));
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, end));
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(end));

	// Returns ms
	return time;
}

#ifdef __INTELLISENSE__
void __syncthreads();
void __syncwarp();
#endif

//------------------------------------------------------------------------------
//
template <typename IteratorDataType, typename IndexType, typename BlockType>
__forceinline__ __host__ __device__ IteratorDataType* pageAccess(memory_t* memory, IndexType page_index, BlockType page_size, uint64_t start_index)
{
	return (IteratorDataType*)&memory[(start_index - page_index) * page_size];
}

//------------------------------------------------------------------------------
// Set Adjacency
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
// 感觉好像不需要edges_per_page啊
__forceinline__ __device__ void setAdjacency(EdgeData* edge_data, vertex_t* adjacency, int index, vertex_t edges_per_page)
{
	edge_data->destination = adjacency[index];
}


//------------------------------------------------------------------------------
// Set DeletionMarker
//------------------------------------------------------------------------------
//
__forceinline__ __device__ void setDeletionMarker(EdgeData* edge_data, vertex_t edges_per_page)
{
	edge_data->destination = DELETIONMARKER;
}


//------------------------------------------------------------------------------
// Template specification for different EdgeDataTypes
//
// pointerHandlingSetup(iterator, memory, block_index, page_size, edges_per_page, start_index);
template <typename EdgeDataType>  
__forceinline__ __device__ void pointerHandlingSetup(EdgeDataType*& adjacency_list, memory_t* memory, vertex_t& block_index, int page_size, vertex_t edges_per_page, uint64_t start_index)
{
	++block_index;
	adjacency_list = pageAccess<EdgeDataType>(memory, block_index, page_size, start_index);
}

template <typename IteratorDataType>
__forceinline__ __device__ void PageindexPointerHandlingSetup(IteratorDataType*& pageindex_list, memory_t* memory, vertex_t& block_index, int page_size, vertex_t pageindexes_per_page, uint64_t start_index)
{
	*((index_t*)pageindex_list) = ++block_index;  // 设置下一页index编号
	pageindex_list = pageAccess<IteratorDataType>(memory, block_index, page_size, start_index);
}


//------------------------------------------------------------------------------
// Traversal
template <typename IteratorDataType>
__forceinline__ __device__ __host__ void pointerHandlingTraverse(IteratorDataType*& iterator_list, memory_t* memory, int page_size, vertex_t edges_per_page, uint64_t& start_index)
{
	iterator_list = pageAccess<IteratorDataType>(memory, *((index_t*)adjacency_list), page_size, start_index);
}

//------------------------------------------------------------------------------
// Traversal with given page index
//
template <typename IteratorDataType>
__forceinline__ __device__ void pointerHandlingTraverse(IteratorDataType*& iterator_list, memory_t* memory, int page_size, vertex_t edges_per_page, uint64_t& start_index, index_t& page_index)
{
	iterator_list = pageAccess<IteratorDataType>(memory, page_index, page_size, start_index);
}

//------------------------------------------------------------------------------
// Get PageIndex when traversal is finished
//

template <typename IteratorDataType>
__forceinline__ __device__ index_t* getBlockIndex(IteratorDataType* adjacency, int numbers_per_page)
{
	return (index_t*)adjacency;
}


//------------------------------------------------------------------------------
// Get PageIndex when pointer is on page start
//

template <typename IteratorDataType>
__forceinline__ __device__ index_t* getBlockIndexAbsolute(IteratorDataType* adjacency, int numbers_per_page)
{
	return (index_t*)(adjacency + edges_per_page);
}



//------------------------------------------------------------------------------
// Iterator class
//------------------------------------------------------------------------------
//
class PageIterator 
{
	
};


class AdjacencyIterator : public PageIterator
{
public:
	__device__ AdjacencyIterator(EdgeData* it) : iterator{ it } {}
	__device__ AdjacencyIterator(const AdjacencyIterator& it) : iterator{ it.iterator } {}
	__device__ AdjacencyIterator() {}

	__forceinline__ __device__ void setIterator(AdjacencyIterator& it) { iterator = it.iterator; }

	__forceinline__ __device__ void setIterator(EdgeData* it) { iterator = it; }

	__forceinline__ __device__ bool isValid() { return iterator != nullptr; }

	__forceinline__ __device__ bool isNotValid() { return iterator == nullptr; }

	__forceinline__ __device__ EdgeData*& getIterator() { return iterator; }

	__forceinline__ __device__ EdgeData* getIteratorAt(index_t index) { return iterator + index; }

	__forceinline__ __device__ vertex_t getDestination() { return iterator->destination; }

	__forceinline__ __device__ EdgeData getElement() { return *iterator; }

	__forceinline__ __device__ EdgeData getElementAt(index_t index) { return iterator[index]; }

	__forceinline__ __device__ EdgeData* getElementPtr() { return iterator; }

	__forceinline__ __device__ EdgeData* getElementPtrAt(index_t index) { return &iterator[index]; }

	__forceinline__ __device__ vertex_t* getDestinationPtr() { return &(iterator->destination); }

	__forceinline__ __device__ vertex_t getDestinationAt(index_t index) { return iterator[index].destination; }

	__forceinline__ __device__ vertex_t* getDestinationPtrAt(index_t index) { return &(iterator[index].destination); }

	__forceinline__ __device__ index_t* getPageIndexPtr(vertex_t& edges_per_page) { return getBlockIndex(iterator, edges_per_page); }

	__forceinline__ __device__ index_t getPageIndex(vertex_t& edges_per_page) { return *getBlockIndex(iterator, edges_per_page); }

	__forceinline__ __device__ index_t* getPageIndexPtrAbsolute(vertex_t& edges_per_page) { return getBlockIndexAbsolute(iterator, edges_per_page); }

	__forceinline__ __device__ index_t getPageIndexAbsolute(vertex_t& edges_per_page) { return *getBlockIndexAbsolute(iterator, edges_per_page); }

	__forceinline__ __device__ void setDestination(vertex_t value) { iterator->destination = value; }

	__forceinline__ __device__ void setDestinationAt(index_t index, vertex_t value) { iterator[index].destination = value; }

	__forceinline__ __device__ void setDestination(AdjacencyIterator& it) { iterator->destination = it.iterator->destination; }

	__forceinline__ __device__ AdjacencyIterator& operator++() { ++iterator; return *this; }

	__forceinline__ __device__ AdjacencyIterator operator++(int) { AdjacencyIterator result(*this); ++iterator; return result; }

	__forceinline__ __device__ AdjacencyIterator& operator+=(int edges_per_page) { iterator += edges_per_page; return *this; }

	__forceinline__ __device__ AdjacencyIterator& operator-=(int edges_per_page) { iterator -= edges_per_page; return *this; }

	__forceinline__ __device__ vertex_t operator[](int index)
	{
		return iterator[index].destination;
	}

	__forceinline__ __device__ vertex_t at(int index, memory_t*& memory, int page_size, uint64_t start_index, vertex_t edges_per_page)
	{
		if (index <= edges_per_page)
		{
			return iterator[index].destination;
		}
		else
		{
			// We need traversal
			EdgeData* tmp_iterator = iterator;
			while (index > edges_per_page)
			{
				tmp_iterator += edges_per_page;
				pointerHandlingTraverse(tmp_iterator, memory, page_size, edges_per_page, start_index);
				index -= edges_per_page;
			}
			return tmp_iterator[index].destination;
		}
	}

	__forceinline__ __device__ void adjacencySetup(int& loop_index, vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, vertex_t* adjacency, int offset_index, index_t& block_index, int& pages, const vertex_t& pageindexes_per_page, uint64_t& start_pageindex, PageIndexIterator& pageindex_iterator, index_t& block_pageindex)
	{
		setAdjacency(iterator, adjacency, offset_index + loop_index, edges_per_page);
		++iterator;
		if ((loop_index) && (loop_index % edges_per_page == 0))
		{
			pointerHandlingSetup(iterator, memory, block_index, page_size, pageindexes_per_page, start_index);
			pageindex_iterator.pageindexSetup(pages, block_index, edges_per_page, memory, page_size, start_pageindex, block_pageindex);
			pages++;
		}
	}

	__forceinline__ __device__ void advanceIterator(int& loop_index, vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index)
	{
		++iterator;
		if (((loop_index) % (edges_per_page)) == (edges_per_page - 1))
		{
			// Edgeblock is full, place pointer to next block, reset adjacency_list pointer to next block
			pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
		}
	}

	__forceinline__ __device__ void blockTraversalAbsolute(vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index)
	{
		iterator += edges_per_page;
		pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
	}

	__forceinline__ __device__ void blockTraversalAbsolutePageIndex(vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, index_t& page_index)
	{
		pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index, page_index);
	}

	__forceinline__ __device__ void blockTraversalAbsolute(vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, index_t& page_index)
	{
		iterator += edges_per_page;
		page_index = getPageIndex(edges_per_page);
		pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
	}

	__forceinline__ __device__ void cleanPageExclusive(vertex_t& edges_per_page)
	{
		++iterator;
		for (int i = 1; i < edges_per_page; ++i)
		{
			setDeletionMarker(iterator, edges_per_page);
			++iterator;
		}
	}

	__forceinline__ __device__ void cleanPageInclusive(vertex_t& edges_per_page)
	{
		for (int i = 0; i < edges_per_page; ++i)
		{
			setDeletionMarker(iterator, edges_per_page);
			++iterator;
		}
	}

	template <typename PageIndexDataType, typename CapacityDataType>
	__forceinline__ __device__ void advanceIteratorEndCheck(int& loop_index, vertex_t& edges_per_page, memory_t*& memory, PageIndexDataType& page_size, uint64_t& start_index, CapacityDataType& capacity)
	{
		++iterator;
		if (((loop_index) % (edges_per_page)) == (edges_per_page - 1) && loop_index != (capacity - 1))
		{
			// Edgeblock is full, place pointer to next block, reset adjacency_list pointer to next block
			pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
		}
	}

	__forceinline__ __device__ bool advanceIteratorEndCheckBoolReturn(int& loop_index, vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, int& capacity, index_t& edge_block_index)
	{
		++iterator;
		if (((loop_index) % (edges_per_page)) == (edges_per_page - 1) && loop_index != (capacity - 1))
		{
			// Edgeblock is full, place pointer to next block, reset adjacency_list pointer to next block
			edge_block_index = getPageIndex(edges_per_page);
			pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
			return true;
		}
		return false;
	}

	__forceinline__ __device__ void advanceIteratorEndCheck(int& loop_index, vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, int& capacity, index_t& edge_block_index)
	{
		++iterator;
		if (((loop_index) % (edges_per_page)) == (edges_per_page - 1) && loop_index != (capacity - 1))
		{
			// Edgeblock is full, place pointer to next block, reset adjacency_list pointer to next block
			edge_block_index = getPageIndex(edges_per_page);
			pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
		}
	}

	__forceinline__ __device__ void advanceIteratorDeletionCompaction(int& loop_index, vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, index_t& edge_block_index, AdjacencyIterator& search_list, vertex_t& shuffle_index) {
		++iterator;
		if (((loop_index) % (edges_per_page)) == (edges_per_page - 1))
		{
			// Edgeblock is full, place pointer to next block, reset adjacency_list pointer to next block
			edge_block_index = *(getBlockIndex(iterator, edges_per_page));
			pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
			search_list.setIterator(iterator);
			shuffle_index -= edges_per_page;
		}
	}

	__forceinline__ __device__ void advanceIteratorToIndex(vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, index_t& edge_block_index, vertex_t& shuffle_index)
	{
		while (shuffle_index >= edges_per_page)
		{
			iterator += edges_per_page;
			edge_block_index = getPageIndex(edges_per_page);
			pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
			shuffle_index -= edges_per_page;
		}
	}

	__forceinline__ __device__ void advanceIteratorToIndex(vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, vertex_t& shuffle_index, int& neighbours, int& capacity)
	{
		while (shuffle_index > edges_per_page)
		{
			iterator += edges_per_page;
			pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
			shuffle_index -= edges_per_page;
		}
		if (shuffle_index == edges_per_page && neighbours < capacity)
		{
			iterator += edges_per_page;
			pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
		}
		else
		{
			iterator += shuffle_index;
		}
	}

protected:
	EdgeData* iterator;
};

/* PageIndex Iterator */
class PageIndexIterator : public PageIterator
{
public:
	__device__ PageIndexIterator(index_t* it) : iterator{ it } {}
	__device__ PageIndexIterator(const PageIndexIterator& it) : iterator{ it.iterator } {}
	__device__ PageIndexIterator() {}

	__forceinline__ __device__ void setIterator(PageIndexIterator& it) { iterator = it.iterator; }

	__forceinline__ __device__ void setIterator(index_t* it) { iterator = it; }

	__forceinline__ __device__ bool isValid() { return iterator != nullptr; }

	__forceinline__ __device__ bool isNotValid() { return iterator == nullptr; }

	__forceinline__ __device__ index_t*& getIterator() { return iterator; }

	__forceinline__ __device__ index_t* getIteratorAt(index_t index) { return iterator + index; }

	__forceinline__ __device__ index_t getPageIndexNext(vertex_t& pageindexes_per_page) { return *getBlockIndexAbsolute(iterator, pageindexes_per_page); }

	__forceinline__ __device__ void setPageindex(index_t index) { *iterator = index; }

	__forceinline__ __device__ index_t at(index_t index, memory_t*& memory, int page_size, uint64_t start_index, vertex_t pageindexes_per_page)
	{
		if (index <= pageindexes_per_page)
		{
			return iterator[index];
		}
		else
		{
			// We need traversal
			index_t* tmp_iterator = iterator;
			while (index > pageindexes_per_page)
			{
				tmp_iterator += edges_per_page;
				PageindexPointerHandlingTraverse(tmp_iterator, memory, page_size, pageindexes_per_page, start_index);
				index -= pageindexes_per_page;
			}
			return tmp_iterator[index];
		}
	}

	__forceinline__ __device__ void setPageindexAt(int index, memory_t*& memory, int page_size, uint64_t start_index, vertex_t pageindexes_per_page, index_t value)
	{ 
		index_t tmp = at(index, memory, page_size, start_index, pageindexes_per_page);
		index_t * p_pageindex= reinterpret_cast<index_t*>(&tmp);
		*p_pageindex = value;
	}

	__forceinline__ __device__ void setPageindex(PageIndexIterator& it) { *iterator = *(it.iterator); }

	__forceinline__ __device__ PageIndexIterator& operator++() { ++iterator; return *this; }

	__forceinline__ __device__ PageIndexIterator operator++(int) { PageIndexIterator result(*this); ++iterator; return result; }

	__forceinline__ __device__ PageIndexIterator& operator+=(int pageindexes_per_page) { iterator += pageindexes_per_page; return *this; }

	__forceinline__ __device__ PageIndexIterator& operator-=(int pageindexes_per_page) { iterator -= pageindexes_per_page; return *this; }
	
	__forceinline__ __device__ vertex_t& operator[](int index)
	{
		return iterator[index];
	}

	__forceinline__ __device__ void pageindexSetup(int& loop_index, index_t value, const vertex_t& pageindexes_per_page, memory_t*& memory, const int& page_size, const uint64_t& start_pageindex, index_t& block_pageindex)
	{
		iterator[loop_index] = value;
		++iterator;
		if (((loop_index) % (pageindexes_per_page)) == (pageindexes_per_page - 1))
		{
			PageindexPointerHandlingSetup(iterator, memory, block_pageindex, page_size, pageindexes_per_page, start_pageindex);
		}
	}

	__forceinline__ __device__ void advanceIterator(int& loop_index, vertex_t& pageindexes_per_page, memory_t*& memory, int& page_size, uint64_t& start_index)
	{
		++iterator;
		if (((loop_index) % (pageindexes_per_page)) == (pageindexes_per_page - 1))
		{
			// Edgeblock is full, place pointer to next block, reset adjacency_list pointer to next block
			pointerHandlingTraverse(iterator, memory, page_size, pageindexes_per_page, start_index);
		}
	}

	__forceinline__ __device__ void blockTraversalAbsolute(vertex_t& pageindexes_per_page, memory_t*& memory, int& page_size, uint64_t& start_index)
	{
		iterator += pageindexes_per_page;
		pointerHandlingTraverse(iterator, memory, page_size, pageindexes_per_page, start_index);
	}

	__forceinline__ __device__ void blockTraversalAbsolutePageIndex(vertex_t& pageindexes_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, index_t& page_index)
	{
		pointerHandlingTraverse(iterator, memory, page_size, pageindexes_per_page, start_index, page_index);
	}

	__forceinline__ __device__ void blockTraversalAbsolute(vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, index_t& page_index)
	{
		iterator += edges_per_page;
		page_index = getPageIndex(edges_per_page);
		pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
	}

	__forceinline__ __device__ void cleanPageExclusive(vertex_t& pageindexes_per_page)
	{
		++iterator;
		for (int i = 1; i < pageindexes_per_page; ++i)
		{
			iterator[i] = DELETIONMARKER;
			++iterator;
		}
	}

	__forceinline__ __device__ void cleanPageInclusive(vertex_t& pageindexes_per_page)
	{
		for (int i = 0; i < pageindexes_per_page; ++i)
		{
			iterator[i] = DELETIONMARKER;
			++iterator;
		}
	}

	template <typename PageIndexDataType, typename CapacityDataType>
	__forceinline__ __device__ void advanceIteratorEndCheck(int& loop_index, vertex_t& pageindexes_per_page, memory_t*& memory, PageIndexDataType& page_size, uint64_t& start_index, CapacityDataType& capacity)
	{
		++iterator;
		if (((loop_index) % (pageindexes_per_page)) == (pageindexes_per_page - 1) && loop_index != (capacity - 1))
		{
			// Edgeblock is full, place pointer to next block, reset adjacency_list pointer to next block
			pointerHandlingTraverse(iterator, memory, page_size, pageindexes_per_page, start_index);
		}
	}

	__forceinline__ __device__ bool advanceIteratorEndCheckBoolReturn(int& loop_index, vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, int& capacity, index_t& edge_block_index)
	{
		++iterator;
		if (((loop_index) % (edges_per_page)) == (edges_per_page - 1) && loop_index != (capacity - 1))
		{
			// Edgeblock is full, place pointer to next block, reset adjacency_list pointer to next block
			edge_block_index = getPageIndex(edges_per_page);
			pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
			return true;
		}
		return false;
	}

	__forceinline__ __device__ void advanceIteratorEndCheck(int& loop_index, vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, int& capacity, index_t& edge_block_index)
	{
		++iterator;
		if (((loop_index) % (edges_per_page)) == (edges_per_page - 1) && loop_index != (capacity - 1))
		{
			// Edgeblock is full, place pointer to next block, reset adjacency_list pointer to next block
			edge_block_index = getPageIndex(edges_per_page);
			pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
		}
	}

	__forceinline__ __device__ void advanceIteratorDeletionCompaction(int& loop_index, vertex_t& pageindexes_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, index_t& edge_block_index, AdjacencyIterator<EdgeDataType>& search_list, vertex_t& shuffle_index) {
		++iterator;
		if (((loop_index) % (pageindexes_per_page)) == (pageindexes_per_page - 1))
		{
			// Edgeblock is full, place pointer to next block, reset adjacency_list pointer to next block
			edge_block_index = *(getBlockIndex(iterator, pageindexes_per_page));
			PageindexPointerHandlingTraverse(iterator, memory, page_size, pageindexes_per_page, start_index);
			search_list.setIterator(iterator);
			shuffle_index -= pageindexes_per_page;
		}
	}

	__forceinline__ __device__ void advanceIteratorToIndex(vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, index_t& edge_block_index, vertex_t& shuffle_index)
	{
		while (shuffle_index >= edges_per_page)
		{
			iterator += edges_per_page;
			edge_block_index = getPageIndex(edges_per_page);
			pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
			shuffle_index -= edges_per_page;
		}
	}

	__forceinline__ __device__ void advanceIteratorToIndex(vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, vertex_t& shuffle_index, int& neighbours, int& capacity)
	{
		while (shuffle_index > edges_per_page)
		{
			iterator += edges_per_page;
			pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
			shuffle_index -= edges_per_page;
		}
		if (shuffle_index == edges_per_page && neighbours < capacity)
		{
			iterator += edges_per_page;
			pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
		}
		else
		{
			iterator += shuffle_index;
		}
	}

protected:
	index_t* iterator;
};