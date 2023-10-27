cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(SymspellGPU VERSION 0.0.1 LANGUAGES CXX CUDA)

add_executable(symspell_gpu src/cli.cu)

add_executable(test_codec test/test_codec.cu)
