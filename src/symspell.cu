#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "codec.cu"
#include "generate_combination.cu"
#include "cub.cu"
#include "kernel.cu"

const int NUM_THREADS = 256;

int gen_combinations(Int3* seq, int distance, Int3* &outputKeys, int* &outputValues, int n) {
	int *combinationOffsets;
	int seq1LenBlocks = (n + NUM_THREADS) / NUM_THREADS;

	// cal combinationOffsets
	cudaMalloc((void**)&combinationOffsets, sizeof(int)*n);
	cal_combination_len <<< seq1LenBlocks, NUM_THREADS >>>(
	    seq, distance, combinationOffsets, n);
	inclusive_sum(combinationOffsets, n);
	int outputLen = transfer_last_element(combinationOffsets, n);

	// generate combinations
	cudaMalloc(&outputKeys, sizeof(Int3)*outputLen);
	cudaMalloc(&outputValues, sizeof(int)*outputLen);
	gen_combination <<< seq1LenBlocks, NUM_THREADS >>> (
	    seq, combinationOffsets, distance, outputKeys, outputValues, n);

	cudaDeviceSynchronize();
	cudaFree(combinationOffsets);
	return outputLen;
}

int gen_pairs(Int3* inputKeys, int* inputValues, Int2* &output, int n, int* buffer, int verbose) {
	int* valueOffsets, *pairOffsets;

	// cal valueOffsets
	cudaMalloc(&valueOffsets, sizeof(int)*n);
	sort_key_values(inputKeys, inputValues, n);
	unique_counts(inputKeys, valueOffsets, buffer, n);

	// cal pairOffsets
	int nUnique = transfer_last_element(buffer, 1);
	if (verbose)
		printf("step 2.1 completed %d\n", nUnique);
	int nUniqueBlock = (nUnique + NUM_THREADS) / NUM_THREADS;
	cudaMalloc(&pairOffsets, sizeof(int)*nUnique);
	cal_pair_len <<< nUniqueBlock, NUM_THREADS>>>(valueOffsets, pairOffsets, nUnique);
	inclusive_sum(valueOffsets, nUnique);
	inclusive_sum(pairOffsets, nUnique);

	// generate pairs
	int outputLen = transfer_last_element(pairOffsets, nUnique);
	if (verbose)
		printf("step 2.2 completed %d\n", outputLen);
	cudaMalloc(&output, sizeof(Int2)*outputLen);
	generate_pairs <<< nUniqueBlock, NUM_THREADS>>>(inputValues, output, valueOffsets, pairOffsets, nUnique);

	cudaDeviceSynchronize();
	_cudaFree(valueOffsets, pairOffsets);
	return outputLen;
}

int postprocessing(Int3* seq, Int2* input, int distance,
                   Int2* &pairOutput, char* &distanceOutput, int n, int* buffer) {
	Int2* uniquePairs;
	char* uniqueDistances, *flags;

	// filter duplicate
	cudaMalloc(&uniquePairs, sizeof(Int2)*n);
	sort_int2(input, n);
	unique(input, uniquePairs, buffer, n);

	// cal levenshtein
	int uniqueLen = transfer_last_element(buffer, 1);
	int byteRequirement = sizeof(char) * uniqueLen;
	int uniqueLenBlock = (uniqueLen + NUM_THREADS) / NUM_THREADS;
	cudaMalloc(&flags, byteRequirement);
	cudaMalloc(&uniqueDistances, byteRequirement);
	cudaMalloc(&distanceOutput, byteRequirement);
	cudaMalloc(&pairOutput, sizeof(Int2)*uniqueLen);
	cal_levenshtein <<< uniqueLenBlock, NUM_THREADS>>>(seq, uniquePairs, distance, uniqueDistances, flags, uniqueLen);

	//filter levenshtein
	double_flag(uniquePairs, uniqueDistances, flags, pairOutput, distanceOutput, buffer, uniqueLen);

	int outputLen =  transfer_last_element(buffer, 1);
	_cudaFree(uniquePairs, uniqueDistances, flags);
	return outputLen;
	// return transfer_last_element(buffer, 1);
}

int symspell_perform(SymspellArgs args, Int3* seq1, SymspellOutput* output) {
	int distance = args.distance, verbose = args.verbose, seq1Len = args.seq1Len;
	int* deviceInt;
	cudaMalloc((void**)&deviceInt, sizeof(int));

	//=====================================
	// step 1: transfer input to GPU
	//=====================================
	Int3* seq1Device;
	int seq1Bytes = sizeof(Int3) * seq1Len;

	cudaMalloc(&seq1Device, seq1Bytes);
	cudaMemcpy(seq1Device, seq1, seq1Bytes, cudaMemcpyHostToDevice);

	if (verbose)
		printf("step 1 completed %d\n", seq1Len);

	//=====================================
	// step 2: generate deletion combinations
	//=====================================
	Int3* combinationKeys;
	int* combinationValues;
	int combinationLen =
	    gen_combinations(seq1Device, distance, combinationKeys, combinationValues, seq1Len);

	if (verbose)
		printf("step 2 completed %d\n", combinationLen);

	//=====================================
	// step 3: turn combinations into pairs
	//=====================================
	Int2* pairs;
	int pairLength =
	    gen_pairs(combinationKeys, combinationValues, pairs, combinationLen, deviceInt, verbose);

	if (verbose)
		printf("step 3 completed %d\n", pairLength);

	//=====================================
	// step 4: Levenshtein/duplicate postprocessing
	//=====================================
	Int2* outputPairs;
	char* outputDistances;
	int outputLen =
	    postprocessing(seq1Device, pairs, distance, outputPairs, outputDistances, pairLength, deviceInt);

	if (verbose)
		printf("step 4 completed %d\n", outputLen);

	//=====================================
	// step 5: transfer output to CPU
	//=====================================
	int pairBytes = sizeof(Int2) * outputLen;
	cudaMallocHost(&output->indexPairs, pairBytes);
	cudaMemcpy(output->indexPairs, outputPairs, pairBytes, cudaMemcpyDeviceToHost);

	int distanceBytes = sizeof(char) * outputLen;
	cudaMallocHost(&output->pairwiseDistances, distanceBytes);
	cudaMemcpy(output->pairwiseDistances, outputDistances, distanceBytes, cudaMemcpyDeviceToHost);

	output->len = outputLen;

	if (verbose)
		printf("step 5 completed\n");

	cudaDeviceSynchronize();
	_cudaFree(deviceInt, seq1Device, combinationKeys, combinationValues, pairs, outputPairs, outputDistances);
	return 0;
}

void symspell_free(SymspellOutput* output) {
	if (output->indexPairs) {
		cudaFreeHost(output->indexPairs);
		output->indexPairs = NULL;
	}
	if (output->pairwiseDistances) {
		cudaFreeHost(output->pairwiseDistances);
		output->pairwiseDistances = NULL;
	}
}

