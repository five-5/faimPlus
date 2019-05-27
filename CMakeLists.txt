# CMake 最低版本要求
cmake_minimum_required(VERSION 3.2)

# 項目名称
project(faimPlus)

if (UNIX)
set(CUDA_HOST_COMPILER /vpublic01/frog/wyh/gcc-6.1.0/bin/g++)
endif (UNIX)

FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(FindCUDA)

option(CUDA_BUILD_CC20 "Build with compute capability 2.0 support" FALSE)
option(CUDA_BUILD_CC21 "Build with compute capability 2.1 support" FALSE)
option(CUDA_BUILD_CC30 "Build with compute capability 3.0 support" FALSE)
option(CUDA_BUILD_CC35 "Build with compute capability 3.5 support" FALSE)
option(CUDA_BUILD_CC50 "Build with compute capability 5.0 support" FALSE)
option(CUDA_BUILD_CC52 "Build with compute capability 5.2 support" FALSE)
option(CUDA_BUILD_CC61 "Build with compute capability 6.1 support" TRUE)
option(CUDA_BUILD_INFO "Build with kernel statistics and line numbers" TRUE)
option(CUDA_BUILD_DEBUG "Build with kernel debug" FALSE)

INCLUDE_DIRECTORIES(/usr/local/cuda-9.1/include)
INCLUDE_DIRECTORIES(include)
   
if(WIN32)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
SET(CUDA_PROPAGATE_HOST_FLAGS ON)
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lrt -std=c++14")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
# SET(CUDA_PROPAGATE_HOST_FLAGS ON)
endif()

# set(CMAKE_BUILD_TYPE Debug)
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -G -DTHRUST_DEBUG")

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4464 /wd4514 /wd4820 /wd4668 /wd4574 /wd4571 /wd4324 /wd4710 /wd4711 /wd4365 /wd4515 /wd4201 /wd4267 /wd5027 /wd4626")
endif()

LIST(APPEND CUDA_NCCC_FLAGS --compiler-options)
if(CUDA_BUILD_CC35)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_35,code=sm_35;")
endif ()


LIST(APPEND CUDA_NVCC_FLAGS "-DVERBOSE;")
LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler -Wall -D_FORCE_INLINES; --expt-extended-lambda;-use_fast_math;-Xcompiler -rdynamic")

if (CUDA_BUILD_INFO)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-keep;--ptxas-options=-v;-lineinfo")
endif ()

if (CUDA_BUILD_DEBUG)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-G")
endif ()

SET(FAIMGRAPHHEADERS
	include/ConfigurationParser.h
	include/COO.h 
	include/CSR.h 
	include/CSVWriter.h 
	include/Definitions.h 
	include/EdgeUpdate.h 
	include/EdgeInsertion.cuh 
	include/EdgeUtility.cuh 
	include/faimGraph.h 
	include/faimGraph.cuh 
	include/GraphParser.h 
	include/MemoryLayout.h 
	include/MemoryManager.h 
	include/MemoryManager.cuh 
	include/Queue.h 
	include/Utility.h 
	include/Vector.h 
	include/VertexMapper.h 
	include/VertexUpdate.h 
)

cuda_add_library(faimPlus 
					src/COO.cpp
					src/CSR.cpp
          			src/ConfigurationParser.cpp
					src/CSVWriter.cpp
					src/MemoryManager.cpp
					src/GraphParser.cpp
					src/Utility.cpp
					src/faimGraph.cpp
					src/EdgeUpdate.cpp
					src/Instantiations.cu
					${FAIMGRAPHHEADERS}
        )

# 指定生成目标 将main.cpp 生成为mainfaimPlus
cuda_add_executable(mainfaimPlus 
          src/main.cpp
           )

# 添加链接库
TARGET_LINK_LIBRARIES(mainfaimPlus faimPlus ${CUDA_cudart_static_LIBRARY})
