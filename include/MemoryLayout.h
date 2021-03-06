//------------------------------------------------------------------------------
// filename: MemoryLayout.h
// Author : @SH
// Description: the struct of data type(edge and vertex)
// Date：2019.5.8
//
//------------------------------------------------------------------------------
//

#pragma once

//------------------------------------------------------------------------------
// EdgeData Variants for simple graphs AOS
//------------------------------------------------------------------------------
//

struct EdgeData
{
	vertex_t destination;
	friend __host__ __device__ bool operator<(const EdgeData &lhs, const EdgeData &rhs) { return (lhs.destination < rhs.destination); }
	static size_t sizeOfUpdateData() { return sizeof(vertex_t) + sizeof(vertex_t); }
};

//------------------------------------------------------------------------------
// EdgeData Variants for simple graphs SOA
//------------------------------------------------------------------------------
//

struct EdgeDataSOA  // 未使用
{
	vertex_t destination;
	static size_t sizeOfUpdateData() { return sizeof(vertex_t) + sizeof(vertex_t); }
};

//------------------------------------------------------------------------------
// EdgeUpdate Variants for simple graphs
//------------------------------------------------------------------------------
//

struct EdgeUpdate
{
	vertex_t source;
  	EdgeData update;
  	index_t neighbor;
	friend __host__ __device__ bool operator<(const EdgeUpdate &lhs, const EdgeUpdate &rhs) 
	{ 
		if (lhs.neighbor == rhs.neighbor) {
		return ((lhs.source > rhs.source) || (lhs.source == rhs.source && (lhs.update < rhs.update)));
		}
		return lhs.neighbor > rhs.neighbor;
	}
};

//------------------------------------------------------------------------------
// VertexData Variants for simple graphs
//------------------------------------------------------------------------------
//

struct VertexData
{
	int locking;
	vertex_t mem_index;
	vertex_t neighbours;
	vertex_t capacity;
	index_t host_identifier;
};

//------------------------------------------------------------------------------
// VertexUpdate Variants for simple graphs
//------------------------------------------------------------------------------
//

struct VertexUpdate
{
	index_t identifier;
};

__forceinline__ __host__ __device__ bool operator<(const VertexUpdate &lhs, const VertexUpdate &rhs) { return (lhs.identifier < rhs.identifier); };

