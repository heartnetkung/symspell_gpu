#include <cub/device/device_scan.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include "codec.cu"

struct Int3Comparator {
	CUB_RUNTIME_FUNCTION __forceinline__ __device__
	bool operator()(const Int3 &lhs, const Int3 &rhs) {
		if (lhs.entry[0] != rhs.entry[0])
			return lhs.entry[0] < rhs.entry[0];
		if (lhs.entry[1] != rhs.entry[1])
			return lhs.entry[0] < rhs.entry[0];
		return lhs.entry[2] < rhs.entry[2];
	}
};

struct Int2Comparator {
	CUB_RUNTIME_FUNCTION __forceinline__ __device__
	bool operator()(const Int2 &lhs, const Int2 &rhs) {
		if (lhs.x != rhs.x)
			return lhs.x < rhs.x;
		return lhs.y < rhs.y;
	}
};

void inclusive_sum(int* input, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	cub::DeviceScan::InclusiveSum(buffer, bufferSize, input, input, n);
	cudaMalloc(&buffer, bufferSize);
	cub::DeviceScan::InclusiveSum(buffer, bufferSize, input, input, n);
	cudaFree(buffer);
}

void sort_key_values(Int3* keys, int* values, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	Int3Comparator op;
	cub::DeviceMergeSort::SortPairs(buffer, bufferSize, keys, values, n, op);
	cudaMalloc(&buffer, bufferSize);
	cub::DeviceMergeSort::SortPairs(buffer, bufferSize, keys, values, n, op);
	cudaFree(buffer);
}

void unique_counts(Int3* keys, int* output, int* nUnique, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	Int3* dummy;
	cudaMalloc(&dummy, sizeof(Int3)*n);
	cub::DeviceRunLengthEncode::Encode(
	    buffer, bufferSize, keys, dummy, output, nUnique, n);
	cudaMalloc(&buffer, bufferSize);
	cub::DeviceRunLengthEncode::Encode(
	    buffer, bufferSize, keys, dummy, output, nUnique, n);
	cudaFree(buffer);
	cudaFree(dummy);
}