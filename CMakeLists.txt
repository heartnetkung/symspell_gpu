cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(SymspellGPU VERSION 0.0.1 LANGUAGES CXX CUDA)

option(BUILD_CUB "Build with cub library" OFF)
option(BUILD_BINARY "Build binary" ON)

if(NOT BUILD_CUB)
	add_executable(test_codec test/test_codec.cu)
	add_executable(test_brute_force test/test_brute_force.cu)
	add_executable(test_generate_combination test/test_generate_combination.cu)
	add_executable(test_n_way_merge test/test_n_way_merge.cu)
elseif(BUILD_BINARY)
	add_executable(symspell_gpu src/cli.cu)
else()
	add_executable(test_cub test/test_cub.cu)
endif(NOT BUILD_CUB)
