#include "test_util.cu"
#include "../src/n_way_merge.cu"

TEST(n_way_merge, {
	int n = 3;
	Int2** pairInput = (Int2**)malloc(sizeof(Int2*) * n);
	char** distanceInput = (char**)malloc(sizeof(char*) * n);
	int inputSizes[] = {3, 2, 4};
	SymspellOutput output;
	int count = 0;
	char expectDistanceOutput[] = {1, 2, 5, 3};
	Int2 expectedPairOutput[] = {{.x = 1, .y = 1}, {.x = 2, .y = 2}, {.x = 3, .y = 3}, {.x = 4, .y = 4}};

	for (int i = 0; i < n; i++) {
		int m = inputSizes[i];
		pairInput[i] = (Int2*)malloc(sizeof(Int2) * m);
		distanceInput[i] = (char*)malloc(sizeof(char) * m);
		for (int j = 0; j < m; j++)
			distanceInput[i][j] = ++count;
	}

	pairInput[0][0] = {.x = 1, .y = 1}; pairInput[0][1] = {.x = 2, .y = 2};
	pairInput[0][2] = {.x = 4, .y = 4};
	pairInput[1][0] = {.x = 2, .y = 2}; pairInput[1][1] = {.x = 3, .y = 3};
	pairInput[2][0] = {.x = 1, .y = 1}; pairInput[2][1] = {.x = 1, .y = 1};
	pairInput[2][2] = {.x = 1, .y = 1}; pairInput[2][3] = {.x = 1, .y = 1};

	n_way_merge(pairInput,  distanceInput, inputSizes, &output, n);

	check(output.len == 4);
	for (int i = 0; i < n; i++) {
		check(expectDistanceOutput[i] == output.pairwiseDistances[i]);
		check(expectedPairOutput[i].x == output.indexPairs[i].x);
		check(expectedPairOutput[i].y == output.indexPairs[i].y);
	}
})