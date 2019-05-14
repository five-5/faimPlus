//------------------------------------------------------------------------------
// filename: Vector.h
// Author : @SH
// Description: DenseVector
// Date：2019.5.8
//
//------------------------------------------------------------------------------
//

#pragma once
#include <memory>

template<typename T>
struct DenseVector
{
	size_t size;
	std::unique_ptr<T[]> data;

	DenseVector() : size(0) { }
	void alloc(size_t s)
	{
		data = std::make_unique<T[]>(s);
		size = s;
	}
};