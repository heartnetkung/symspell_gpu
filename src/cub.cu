#include <cub/device/device_scan.cuh>
#include <cub/device/device_merge_sort.cuh>
#include "codec.cu"

struct Int3Comparator {
	CUB_RUNTIME_FUNCTION __forceinline__ __device__
	bool operator()(const Int3 &lhs, const Int3 &rhs) {
		unint32_t temp;
		temp = lhs.entry[0] - rhs.entry[0];
		if (temp != 0)
			return temp;
		temp = lhs.entry[1] - rhs.entry[1];
		if (temp != 0)
			return temp;
		return lhs.entry[2] - rhs.entry[2];
	}
};

void inclusive_sum(int* input, int* output, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	cub::DeviceScan::InclusiveSum(buffer, bufferSize, input, output, n);
	cudaMalloc(&buffer, bufferSize);
	cub::DeviceScan::InclusiveSum(buffer, bufferSize, input, output, n);
	cudaFree(buffer);
}

void sort_pairs(Int3* keys, int* values, int n) {
	void *buffer = NULL;
	size_t bufferSize = 0;
	Int3Comparator op;
	cub::DeviceMergeSort::SortPairs(buffer, bufferSize, keys, values, n, op);
	cudaMalloc(&buffer, bufferSize);
	cub::DeviceMergeSort::SortPairs(buffer, bufferSize, keys, values, n, op);
	cudaFree(buffer);
}