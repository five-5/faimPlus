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
	friend __host__ __device__ bool operator<(const EdgeDataUpdate &lhs, const EdgeDataUpdate &rhs)
	{
		return ((lhs.source < rhs.source) || (lhs.source == rhs.source && (lhs.update < rhs.update)));
	}
};

//------------------------------------------------------------------------------
// VertexData Variants for simple graphs
//------------------------------------------------------------------------------
//

typedef struct VertexData
{
	int locking;
	vertex_t mem_index;
	vertex_t neighbours;
	vertex_t capacity;
	index_t host_identifier;
}VertexData;

//------------------------------------------------------------------------------
// VertexUpdate Variants for simple graphs
//------------------------------------------------------------------------------
//

typedef struct VertexUpdate
{
	index_t identifier;
}VertexUpdate;

__forceinline__ __host__ __device__ bool operator<(const VertexUpdate &lhs, const VertexUpdate &rhs) { return (lhs.identifier < rhs.identifier); };

