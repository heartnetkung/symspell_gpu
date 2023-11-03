#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "codec.cu"
#include "generate_combination.cu"
// #include "cub.cu"

const int NUM_THREADS = 256;

int transfer_last_element(int* deviceArr, int n) {
	int ans[1];
	cudaMemcpy(ans, deviceArr + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	return ans[0];
}

int symspell_perform(SymspellArgs args, Int3* seq1, SymspellOutput* output) {
	// //=====================================
	// // step 1: transfer input to GPU
	// //=====================================
	// Int3* seq1Device;

	// int seq1Bytes = sizeof(Int3) * args.seq1Len;
	// cudaMalloc((void**)&seq1Device, seq1Bytes);
	// cudaMemcpy(seq1Device, seq1, seq1Bytes, cudaMemcpyHostToDevice);

	// if (args.verbose)
	// 	printf("step 1 completed\n");

	// //=====================================
	// // step 2: generate deletion combinations
	// //=====================================
	// int* combinationOffsets;
	// int* combinationLengths;
	// Int3* combinationKeys;
	// int* combinationValues;
	// int combinationLength;
	// int numBlocks1 = (int)ceil(args.seq1Len / NUM_THREADS);

	// cudaMalloc((void**)&combinationOffsets, sizeof(int)*args.seq1Len);
	// cudaMalloc((void**)&combinationLengths, sizeof(int)*args.seq1Len);

	// cal_combination_len <<< numBlocks1, NUM_THREADS >>>(
	//     seq1Device, args.distance, combinationLengths, args.seq1Len);
	// inclusive_sum(combinationLengths, combinationOffsets, args.seq1Len);
	// combinationLength = transfer_last_element(combinationOffsets, args.seq1Len);

	// cudaMalloc((void**)&combinationKeys, sizeof(Int3)*combinationLength);
	// cudaMalloc((void**)&combinationValues, sizeof(int)*combinationLength);

	// gen_combination <<< numBlocks1, NUM_THREADS >>> (
	//     seq1, combinationOffsets, args.distance, combinationKeys, combinationValues, args.seq1Len);

	// if (args.verbose)
	// 	printf("step 2 completed\n");

	// //=====================================
	// // step 3: sort combinations
	// //=====================================
	// sort_keys(combinationKeys, combinationValues, combinationLength);

	// if (args.verbose)
	// 	printf("step 3 completed\n");

	// //=====================================
	// // step 4: turn combinations into pairs
	// //=====================================
	// int* pairOffsets;
	// Int2* pairs;
	// int pairLength;

	// //TODO

	// if (args.verbose)
	// 	printf("step 4 completed\n");

	// //=====================================
	// // step 5: Levenshtein/duplicate postprocessing
	// //=====================================
	// Int2* validPairs;
	// Int2* uniquePairs;
	// int* pairwiseDistances;
	// int uniquePairLength, validPairLength;

	// cudaMalloc((void**)validPairs, sizeof(Int2)*pairLength);
	// validPairLength = validate_pairs(pairs, validPairs, pairLength);

	// cudaMalloc((void**)uniquePairs, sizeof(Int2)*validPairLength);
	// uniquePairLength = unique(validPairs, uniquePairs, validPairLength);

	// cudaMalloc((void**)pairwiseDistances, sizeof(int)*uniquePairLength);
	// cal_levenshtein(uniquePairs, seq1Device, pairwiseDistances, uniquePairLength);

	// if (args.verbose)
	// 	printf("step 5 completed\n");
	// //=====================================
	// // step 6: transfer output to CPU
	// //=====================================
	// int pairBytes = sizeof(Int2) * uniquePairLength;
	// cudaMallocHost((void**)&output->indexPairs, pairBytes);
	// cudaMemcpy(uniquePairs, output->indexPairs, pairBytes, cudaMemcpyDeviceToHost);

	// int distanceBytes = sizeof(int) * uniquePairLength;
	// cudaMallocHost((void**)&output->pairwiseDistances, distanceBytes);
	// cudaMemcpy(pairwiseDistances, output->pairwiseDistances, distanceBytes, cudaMemcpyDeviceToHost);

	// output->len = uniquePairLength;
	// cudaDeviceSynchronize();

	// if (args.verbose)
	// 	printf("step 6 completed\n");

	// //=====================================
	// // step 7: clean up
	// //=====================================
	// cudaFree(seq1Device); //step 1
	// cudaFree(combinationOffsets);//step 2
	// cudaFree(combinationKeys);
	// cudaFree(combinationValues);
	// cudaFree(combinationLengths);
	// cudaFree(pairOffsets); //step 4
	// cudaFree(pairs);
	// cudaFree(validPairs);//step 5
	// cudaFree(uniquePairs);
	// cudaFree(pairwiseDistances);

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

