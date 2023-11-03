cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(SymspellGPU VERSION 0.0.1 LANGUAGES CXX CUDA)

add_executable(symspell_gpu src/cli.cu)

# test cases
add_executable(test_codec test/test_codec.cu)
add_executable(test_brute_force test/test_brute_force.cu)
add_executable(test_generate_combination test/test_generate_combination.cu)
