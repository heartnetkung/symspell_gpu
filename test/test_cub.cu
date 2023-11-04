#include "test_util.cu"
#include "../src/cub.cu"
#include "../src/kernel.cu"

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

	cal_combination_len <<< inputLen, 1>>>(input2, distance, output, inputLen);
	inclusive_sum(output, inputLen);

	cudaDeviceSynchronize();
	for (int i = 0; i < inputLen; i++)
		check(expected[i] == output[i]);

	cudaFree(input2);
	cudaFree(output);
})

TEST(sort_key_values, {
	int inputLen = 4;
	char keys[inputLen][3] = {"AC", "AC", "A", "AC"};
	Int3* keys2;
	int* values;
	char expectedKeys[inputLen][3] = {"A", "AC", "AC", "AC"};
	int expectedValues[] = {2, 0, 1, 3};

	cudaMallocManaged(&keys2, sizeof(Int3)*inputLen);
	cudaMallocManaged(&values, sizeof(int)*inputLen);
	for (int i = 0; i < inputLen; i++) {
		keys2[i] = str_encode(keys[i]);
		values[i] = i;
	}

	sort_key_values(keys2, values, inputLen);

	cudaDeviceSynchronize();
	for (int i = 0; i < inputLen; i++) {
		checkstr(str_decode(keys2[i]), expectedKeys[i]);
		check(values[i] == expectedValues[i]);
	}

	cudaFree(keys2);
	cudaFree(values);
})

TEST(generate_pairs, {
	int inputLen = 8;
	char keys[inputLen][3] = {
		"AC", "AC", "AC", "A",
		"C", "C", "B", "B"
	};
	Int3* keys2;
	int* values;
	int* valueOffsets;
	int* pairOffsets;
	int* nUnique;

	cudaMallocManaged(&keys2, sizeof(Int3)*inputLen);
	cudaMallocManaged(&values, sizeof(int)*inputLen);
	cudaMallocManaged(&valueOffsets, sizeof(int)*inputLen);
	cudaMallocManaged(&nUnique, sizeof(int));
	for (int i = 0; i < inputLen; i++) {
		keys2[i] = str_encode(keys[i]);
		values[i] = i;
	}


	unique_counts(keys2, valueOffsets, nUnique, inputLen);
	cudaDeviceSynchronize();

	cudaMallocManaged(&pairOffsets, sizeof(int)*nUnique[0]);
	cal_pair_len(valueOffsets, pairOffsets);
	inclusive_sum(valueOffsets, nUnique[0]);
	inclusive_sum(pairOffsets, nUnique[0]);

	cudaDeviceSynchronize();
	print_int_arr(valueOffsets, nUnique[0]);
	print_int_arr(pairOffsets, nUnique[0]);

	cudaFree(keys2);
	cudaFree(values);
	cudaFree(valueOffsets);
	cudaFree(nUnique);
	// cudaFree(pairOffsets);
})