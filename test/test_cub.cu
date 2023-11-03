#include "test_util.cu"
#include "../src/cub.cu"

TEST(cal_combination_offset, {
	int inputLen = 4;
	char input[inputLen][7] = {"AC", "ACCC", "ACDEFG", "A"};
	Int3* input2;
	int* output;
	int distance = 2;
	int expected[] = {4, 15, 37, 39};

	cudaMallocManaged(&input2, sizeof(Int3)*inputLen);
	cudaMallocManaged(&output, sizeof(int)*inputLen);

	for (int i = 0; i < inputLen; i++)
		input2[i] = str_encode(input[i]);

	cal_combination_offset(input2, distance, output, inputLen);

	cudaDeviceSynchronize();
	for (int i = 0; i < inputLen; i++)
		check(expected[i] == output[i]);

	cudaFree(input2);
	cudaFree(output);
})