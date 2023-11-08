#include <stdio.h>
#include <stdlib.h>
#include "codec.cu"
#include "generate_combination.cu"
#include "cub.cu"
#include "kernel.cu"
#include "n_way_merge.cu"

const int NUM_THREADS = 256;

int gen_combinations(Int3* seq, int distance, Int3* &outputKeys, int* &outputValues, int n) {
	int *combinationOffsets;
	int seq1LenBlocks = divideCeil(n, NUM_THREADS);

	// cal combinationOffsets
	cudaMalloc((void**)&combinationOffsets, sizeof(int)*n);
	gpuerr();
	cal_combination_len <<< seq1LenBlocks, NUM_THREADS >>>(
	    seq, distance, combinationOffsets, n);
	gpuerr();
	inclusive_sum(combinationOffsets, n);
	gpuerr();
	int outputLen = transfer_last_element(combinationOffsets, n);
	gpuerr();

	// generate combinations
	cudaMalloc(&outputKeys, sizeof(Int3)*outputLen);
	gpuerr();
	cudaMalloc(&outputValues, sizeof(int)*outputLen);
	gpuerr();
	gen_combination <<< seq1LenBlocks, NUM_THREADS >>> (
	    seq, combinationOffsets, distance, outputKeys, outputValues, n);
	gpuerr();

	cudaFree(combinationOffsets);
	gpuerr();
	return outputLen;
}

int cal_offsets(Int3* inputKeys, int* inputValues, int* &inputOffsets, int* &outputLengths, int n, int* buffer) {
	// cal valueOffsets
	cudaMalloc(&inputOffsets, sizeof(int)*n);
	gpuerr();
	sort_key_values(inputKeys, inputValues, n);
	gpuerr();
	unique_counts(inputKeys, inputOffsets, buffer, n);
	gpuerr();

	// cal pairOffsets
	int nUnique = transfer_last_element(buffer, 1);
	gpuerr();
	int nUniqueBlock = divideCeil(nUnique, NUM_THREADS);
	cudaMalloc(&outputLengths, sizeof(int)*nUnique);
	gpuerr();
	cal_pair_len <<< nUniqueBlock, NUM_THREADS>>>(inputOffsets, outputLengths, nUnique);
	gpuerr();
	inclusive_sum(inputOffsets, nUnique);
	gpuerr();
	return nUnique;
}

//inputOffsets, outputLengths, n moved as per loop
int gen_pairs(int* input, int* inputOffsets, int &carry, int* outputLengths, Int2* &output, int n, int* buffer) {
	// generate output offsets
	int* outputOffsets;
	cudaMalloc(&outputOffsets, sizeof(int)*n);
	gpuerr();
	inclusive_sum(outputLengths, outputOffsets, n);
	gpuerr();

	// generate pairs
	int outputLen = transfer_last_element(outputOffsets, n);
	gpuerr();
	int nBlock = divideCeil(n, NUM_THREADS);
	cudaMalloc(&output, sizeof(Int2)*outputLen);
	gpuerr();
	generate_pairs <<< nBlock, NUM_THREADS>>>(input, carry, output, inputOffsets, outputOffsets, n);
	gpuerr();

	carry = transfer_last_element(inputOffsets, n);
	gpuerr();
	cudaFree(outputOffsets);
	gpuerr();
	return outputLen;
}

int postprocessing(Int3* seq, Int2* input, int distance,
                   Int2* &pairOutput, char* &distanceOutput,
                   int n, int* buffer, int seqLen) {
	Int2* uniquePairs;
	char* uniqueDistances, *flags;

	// filter duplicate
	cudaMalloc(&uniquePairs, sizeof(Int2)*n);
	gpuerr();
	sort_int2(input, n);
	gpuerr();
	unique(input, uniquePairs, buffer, n);
	gpuerr();

	// cal levenshtein
	int uniqueLen = transfer_last_element(buffer, 1);
	int byteRequirement = sizeof(char) * uniqueLen;
	int uniqueLenBlock = divideCeil(uniqueLen, NUM_THREADS);
	cudaMalloc(&flags, byteRequirement);
	gpuerr();
	cudaMalloc(&uniqueDistances, byteRequirement);
	gpuerr();
	cudaMalloc(&distanceOutput, byteRequirement);
	gpuerr();
	cudaMalloc(&pairOutput, sizeof(Int2)*uniqueLen);
	gpuerr();
	cal_levenshtein <<< uniqueLenBlock, NUM_THREADS>>>(
	    seq, uniquePairs, distance, uniqueDistances, flags, uniqueLen, seqLen);
	gpuerr();

	// filter levenshtein
	double_flag(uniquePairs, uniqueDistances, flags, pairOutput,
	            distanceOutput, buffer, uniqueLen);
	gpuerr();

	_cudaFree(uniquePairs, uniqueDistances, flags);
	gpuerr();
	return transfer_last_element(buffer, 1);
}

void symspell_perform(SymspellArgs args, Int3* seq1, SymspellOutput* output) {
	int distance = args.distance, verbose = args.verbose, seq1Len = args.seq1Len, nSegment = args.nSegment;
	int* deviceInt;
	cudaMalloc((void**)&deviceInt, sizeof(int));

	//=====================================
	// step 1: transfer input to GPU
	//=====================================
	Int3* seq1Device = host_to_device(seq1, seq1Len);
	print_tp(verbose, "1", seq1Len);

	//=====================================
	// step 2: generate deletion combinations
	//=====================================
	Int3* combinationKeys;
	int* combinationValues;
	int combinationLen =
	    gen_combinations(seq1Device, distance, combinationKeys, combinationValues, seq1Len);

	print_tp(verbose, "2", combinationLen);

	//=====================================
	// step 3: calculate pair offsets from combination values
	//=====================================
	int* combinationValueOffsets, *pairLengths;
	int offsetLen =
	    cal_offsets(combinationKeys, combinationValues, combinationValueOffsets,
	                pairLengths, combinationLen, deviceInt);

	print_tp(verbose, "3", offsetLen);
	cudaFree(combinationKeys);

	//=====================================
	// step 4: generate output buffers segment by segment
	//=====================================
	Int2** pairBuffer = (Int2**)malloc(nSegment * sizeof(Int2*));
	char** distanceBuffer = (char**)malloc(nSegment * sizeof(char*));
	int* bufferLengths = (int*)malloc(nSegment * sizeof(int));

	int chunkPerSegment = divideCeil(offsetLen, nSegment);
	int carry = 0;
	int *pairLengthsP = pairLengths, *combinationValueOffsetsP = combinationValueOffsets;

	for (int i = 0; i < nSegment; i++) {
		// the last segment can be smaller than others
		if ((i == nSegment - 1) && (nSegment != 1)) {
			if (offsetLen % chunkPerSegment != 0)
				chunkPerSegment = offsetLen % chunkPerSegment;
		}

		if (verbose)
			printf("iteration #%d----------------\n", i);

		Int2* pairTemp, *pairOut;
		char* distanceOut;

		// 4.1
		int pairTempLen =
		    gen_pairs(combinationValues, combinationValueOffsetsP, carry,
		              pairLengthsP, pairTemp, chunkPerSegment, deviceInt);
		print_tp(verbose, "4.1", pairTempLen);

		// 4.2
		int outputLen =
		    postprocessing(seq1Device, pairTemp, distance, pairOut,
		                   distanceOut, pairTempLen, deviceInt, seq1Len);
		print_tp(verbose, "4.2", outputLen);

		// transfer data to CPU
		bufferLengths[i] = outputLen;
		pairBuffer[i] = device_to_host(pairOut, outputLen);
		distanceBuffer[i] = device_to_host(distanceOut, outputLen);

		// update loop variable
		combinationValueOffsetsP += chunkPerSegment;
		pairLengthsP += chunkPerSegment;
		_cudaFree(pairTemp, pairOut, distanceOut);
	}

	//=====================================
	// step 5: merge output at CPU
	//=====================================
	n_way_merge(pairBuffer, distanceBuffer, bufferLengths, output, nSegment);
	print_tp(verbose, "5", output->len);

	//=====================================
	// step 6: deallocate
	//=====================================
	_cudaFree(deviceInt, seq1Device, combinationValues, combinationValueOffsets, pairLengths);
	for (int i = 0; i < nSegment; i++)
		_cudaFreeHost(pairBuffer[i], distanceBuffer[i]);
	_free(pairBuffer, distanceBuffer, bufferLengths);
}

void symspell_free(SymspellOutput *output) {
	if (output->indexPairs) {
		cudaFreeHost(output->indexPairs);
		output->indexPairs = NULL;
	}
	if (output->pairwiseDistances) {
		cudaFreeHost(output->pairwiseDistances);
		output->pairwiseDistances = NULL;
	}
}