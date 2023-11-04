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

	_cudaFree(input2, output);
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

	_cudaFree(keys2, values);
})

TEST(generate_pairs, {
	int inputLen = 8;
	char keys[inputLen][3] = {
		"AC", "AC", "AC", "A",
		"C", "C", "C", "C"
	};
	Int3* keys2;
	int *values, *valueOffsets, *pairOffsets, *nUnique;
	Int2* pairs;
	int expectedValueOffsets[] = {3, 4, 8};
	int expectedPairOffsets[] = {3, 3, 9};
	Int2 expectedPairs[9];
	expectedPairs[0] = {.x = 0, .y = 1};
	expectedPairs[1] = {.x = 0, .y = 2};
	expectedPairs[2] = {.x = 1, .y = 2};
	expectedPairs[3] = {.x = 4, .y = 5};
	expectedPairs[4] = {.x = 4, .y = 6};
	expectedPairs[5] = {.x = 4, .y = 7};
	expectedPairs[6] = {.x = 5, .y = 6};
	expectedPairs[7] = {.x = 5, .y = 7};
	expectedPairs[8] = {.x = 6, .y = 7};

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
	cal_pair_len <<< nUnique[0], 1>>>(valueOffsets, pairOffsets, nUnique[0]);
	inclusive_sum(valueOffsets, nUnique[0]);
	inclusive_sum(pairOffsets, nUnique[0]);

	int pairLength = transfer_last_element(pairOffsets, nUnique[0]);
	cudaMallocManaged(&pairs, sizeof(Int2)*pairLength);
	generate_pairs <<< nUnique[0], 1>>>(values, pairs, valueOffsets, pairOffsets, nUnique[0]);

	cudaDeviceSynchronize();
	for (int i = 0; i < nUnique[0]; i++) {
		check(valueOffsets[i] == expectedValueOffsets[i]);
		check(pairOffsets[i] == expectedPairOffsets[i]);
	}
	for (int i = 0; i < pairLength; i++) {
		check(pairs[i].x == expectedPairs[i].x);
		check(pairs[i].y == expectedPairs[i].y);
	}

	_cudaFree(keys2, values, valueOffsets, nUnique, pairs, pairOffsets);
})

TEST(unique_pairs, {
	int inputLen = 5;
	Int2 * input, *input2, *output;
	int* outputLen;
	Int2 expectedPairs[] = {{.x = 0, .y = 1}, {.x = 0, .y = 2}, {.x = 1, .y = 2}};

	cudaMallocManaged(&input, sizeof(Int2)*inputLen);
	cudaMallocManaged(&input2, sizeof(Int2)*inputLen);
	cudaMallocManaged(&output, sizeof(Int2)*inputLen);
	cudaMallocManaged(&outputLen, sizeof(int));
	input[0] = {.x = 0, .y = 1};
	input[1] = {.x = 0, .y = 2};
	input[2] = {.x = 1, .y = 2};
	input[3] = {.x = 0, .y = 1};
	input[4] = {.x = 0, .y = 1};

	sort_int2(input, inputLen);
	unique(input, output, outputLen, inputLen);

	cudaDeviceSynchronize();
	check(outputLen[0] == 3);
	for (int i = 0; i < outputLen[0]; i++) {
		check(output[i].x == expectedPairs[i].x);
		check(output[i].y == expectedPairs[i].y);
	}

	_cudaFree(input, input2, output, outputLen);
})