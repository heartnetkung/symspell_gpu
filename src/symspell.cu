#include <stdio.h>
#include <stdlib.h>
#include "codec.cu"
#include "generate_combination.cu"
#include "cub.cu"
#include "kernel.cu"

const int NUM_THREADS = 256;

int gen_combinations(Int3* seq, int distance, Int3* &outputKeys, int* &outputValues, int n) {
	int *combinationOffsets;
	int seq1LenBlocks = divideCeil(n, NUM_THREADS);

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

	cudaFree(combinationOffsets);
	return outputLen;
}

int cal_offsets(Int3* inputKeys, int* inputValues, int* &inputOffsets, int* &outputLengths, int n, int* buffer) {
	// cal valueOffsets
	cudaMalloc(&inputOffsets, sizeof(int)*n);
	sort_key_values(inputKeys, inputValues, n);
	unique_counts(inputKeys, inputOffsets, buffer, n);

	// cal pairOffsets
	int nUnique = transfer_last_element(buffer, 1);
	int nUniqueBlock = divideCeil(nUnique, NUM_THREADS);
	cudaMalloc(&outputLengths, sizeof(int)*nUnique);
	cal_pair_len <<< nUniqueBlock, NUM_THREADS>>>(inputOffsets, outputLengths, nUnique);
	inclusive_sum(inputOffsets, nUnique);
	return nUnique;
}

//inputOffsets, outputLengths, n moved as per loop
int gen_pairs(int* input, int* inputOffsets, int* outputLengths, Int2* &output, int n, int* buffer) {
	// generate output offsets
	int* outputOffsets;
	cudaMalloc(&outputOffsets, sizeof(int)*n);
	inclusive_sum(outputLengths, outputOffsets, n);

	// generate pairs
	int outputLen = transfer_last_element(outputOffsets, n);
	int nBlock = divideCeil(n, NUM_THREADS);
	cudaMalloc(&output, sizeof(Int2)*outputLen);
	generate_pairs <<< nBlock, NUM_THREADS>>>(input, output, inputOffsets, outputOffsets, n);

	cudaFree(outputOffsets);
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
	int uniqueLenBlock = divideCeil(uniqueLen, NUM_THREADS);
	cudaMalloc(&flags, byteRequirement);
	cudaMalloc(&uniqueDistances, byteRequirement);
	cudaMalloc(&distanceOutput, byteRequirement);
	cudaMalloc(&pairOutput, sizeof(Int2)*uniqueLen);
	cal_levenshtein <<< uniqueLenBlock, NUM_THREADS>>>(seq, uniquePairs, distance, uniqueDistances, flags, uniqueLen);

	//filter levenshtein
	double_flag(uniquePairs, uniqueDistances, flags, pairOutput, distanceOutput, buffer, uniqueLen);

	_cudaFree(uniquePairs, uniqueDistances, flags);
	return transfer_last_element(buffer, 1);
}

int concat_buffers(Int2** keyBuffer, char** valueBuffer, int* bufferLengths,
                   Int2* &keyOutput, char* &valueOutput, int n) {
	int totalBufferLength = 0;
	for (int i = 0; i < n; i++)
		totalBufferLength += bufferLengths[i];

	cudaMalloc(&keyOutput, sizeof(Int2)*totalBufferLength);
	cudaMalloc(&valueOutput, sizeof(char)*totalBufferLength);

	Int2* keyOutputP = keyOutput;
	char* valueOutputP = valueOutput;
	int bufferLength;
	for (int i = 0; i < n; i++) {
		bufferLength = bufferLengths[i];
		cudaMemcpy(keyOutputP, keyBuffer[i], sizeof(Int2)*bufferLength, cudaMemcpyDeviceToDevice);
		cudaMemcpy(valueOutputP, valueBuffer[i], sizeof(char)*bufferLength, cudaMemcpyDeviceToDevice);
		keyOutputP += bufferLength;
		valueOutputP += bufferLength;
	}

	return totalBufferLength;
}

int remove_duplicate(Int2* keyInput, char* valueInput, Int2* &keyOutput, char* &valueOutput, int n, int* buffer) {
	cudaMalloc(&keyOutput, sizeof(Int2)*n);
	cudaMalloc(&valueOutput, sizeof(char)*n);
	sort_key_values2(keyInput, valueInput, n);
	unique_by_key(keyInput, valueInput, keyOutput, valueOutput, buffer, n);
	return transfer_last_element(buffer, 1);
}

int symspell_perform(SymspellArgs args, Int3* seq1, SymspellOutput* output) {
	int distance = args.distance, verbose = args.verbose, seq1Len = args.seq1Len, nSegment = args.nSegment;
	int* deviceInt;
	cudaMalloc((void**)&deviceInt, sizeof(int));

	//=====================================
	// step 1: transfer input to GPU
	//=====================================
	Int3* seq1Device;
	int seq1Bytes = sizeof(Int3) * seq1Len;

	cudaMalloc(&seq1Device, seq1Bytes);
	cudaMemcpy(seq1Device, seq1, seq1Bytes, cudaMemcpyHostToDevice);

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
	Int2** pairBuffer;
	char** distanceBuffer;
	int* bufferLengths;
	cudaMallocHost(&pairBuffer, sizeof(Int2*)*nSegment);
	cudaMallocHost(&distanceBuffer, sizeof(char*)*nSegment);
	cudaMallocHost(&bufferLengths, sizeof(int)*nSegment);

	int chunkPerSegment = divideCeil(offsetLen, nSegment);
	Int2* tempPairs;
	int tempPairLength;
	int* combinationValueOffsetsP = combinationValueOffsets;
	int* pairLengthsP = pairLengths;
	int segment = 0;
	for (; segment < nSegment - 1; segment++) {
		tempPairLength =
		    gen_pairs(combinationValues, combinationValueOffsetsP,
		              pairLengthsP, tempPairs, chunkPerSegment, deviceInt);
		bufferLengths[segment] =
		    postprocessing(seq1Device, tempPairs, distance, pairBuffer[segment],
		                   distanceBuffer[segment], tempPairLength, deviceInt);
		print_tp(verbose, "4.1", tempPairLength);
		print_tp(verbose, "4.2", bufferLengths[segment]);

		combinationValueOffsetsP += chunkPerSegment;
		pairLengthsP += chunkPerSegment;
	}

	chunkPerSegment = offsetLen % nSegment;
	segment = n - 1;
	tempPairLength =
	    gen_pairs(combinationValues, combinationValueOffsetsP,
	              pairLengthsP, tempPairs, chunkPerSegment, deviceInt);
	bufferLengths[segment] =
	    postprocessing(seq1Device, tempPairs, distance, pairBuffer[segment],
	                   distanceBuffer[segment], tempPairLength, deviceInt);
	print_tp(verbose, "4.1", tempPairLength);
	print_tp(verbose, "4.2", bufferLengths[segment]);

	_cudaFree(seq1Device, combinationValues, combinationValueOffsets, pairLengths);

	//=====================================
	// step 5: merge buffers
	//=====================================
	Int2* outputPairs;
	char* outputDistances;
	int outputLen;

	if (nSegment == 1) {
		outputPairs = pairBuffer[0];
		outputDistances = distanceBuffer[0];
		outputLen = bufferLengths[0];
	} else {
		Int2* pairAllBuffer;
		char* distanceAllBuffer;
		int allBufferLen = concat_buffers(pairBuffer, distanceBuffer, bufferLengths,
		                                  pairAllBuffer, distanceAllBuffer, nSegment);
		print_tp(verbose, "5.1", allBufferLen);

		outputLen = remove_duplicate(
		                pairAllBuffer, distanceAllBuffer, outputPairs,
		                outputDistances, allBufferLen, deviceInt);
		_cudaFree(pairAllBuffer, distanceAllBuffer);
	}

	print_tp(verbose, "5", outputLen);
	for (int i = 0; i < nSegment; i++)
		_cudaFree(pairBuffer[i], distanceBuffer[i]);
	_cudaFreeHost(pairBuffer, distanceBuffer, bufferLengths);

	//=====================================
	// step 6: transfer output to CPU
	//=====================================
	int pairBytes = sizeof(Int2) * outputLen;
	cudaMallocHost(&output->indexPairs, pairBytes);
	cudaMemcpy(output->indexPairs, outputPairs, pairBytes, cudaMemcpyDeviceToHost);

	int distanceBytes = sizeof(char) * outputLen;
	cudaMallocHost(&output->pairwiseDistances, distanceBytes);
	cudaMemcpy(output->pairwiseDistances, outputDistances, distanceBytes, cudaMemcpyDeviceToHost);

	print_tp(verbose, "6", outputLen);
	_cudaFree(deviceInt, outputPairs, outputDistances);
	return 0;
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