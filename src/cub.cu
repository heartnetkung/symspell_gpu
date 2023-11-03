#include <cub/device/device_scan.cuh>
#include "../src/codec.cu"

__global__
void to_len(Int3* input, int distance, int* output, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	int len = len_decode(input[tid]);
	int newValue = 1 + len;
	switch (distance < len ? distance : len) {
	case 4:
		newValue += len * (len - 1) * (len - 2) * (len - 3) / 24;
	case 3:
		newValue += len * (len - 1) * (len - 2) / 6;
	case 2:
		newValue += len * (len - 1) / 2;
	}
	output[tid] = newValue;
}

void cal_combination_offset(Int3* input, int distance, int* output, int n) {
	int* lens;
	cudaMalloc(&lens, sizeof(int)*n);

	int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize;
	to_len <<< numBlocks, blockSize>>>(input, distance, lens, n);

	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cub::DeviceScan::InclusiveSum(
	    d_temp_storage, temp_storage_bytes, lens, output, n);

	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	cub::DeviceScan::InclusiveSum(
	    d_temp_storage, temp_storage_bytes, lens, output, n);

	cudaFree(d_temp_storage);
	cudaFree(lens);
}