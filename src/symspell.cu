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

	carry += transfer_last_element(inputOffsets, n);
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

int concat_buffers(Int2** keyBuffer, char** valueBuffer, int* bufferLengths,
                   Int2* &keyOutput, char* &valueOutput, int n) {
	int totalBufferLength = 0;
	for (int i = 0; i < n; i++)
		totalBufferLength += bufferLengths[i];

	cudaMalloc(&keyOutput, sizeof(Int2)*totalBufferLength);
	gpuerr();
	cudaMalloc(&valueOutput, sizeof(char)*totalBufferLength);
	gpuerr();

	Int2* keyOutputP = keyOutput;
	char* valueOutputP = valueOutput;
	int bufferLength;
	for (int i = 0; i < n; i++) {
		bufferLength = bufferLengths[i];
		cudaMemcpy(keyOutputP, keyBuffer[i], sizeof(Int2)*bufferLength, cudaMemcpyDeviceToDevice);
		gpuerr();
		cudaMemcpy(valueOutputP, valueBuffer[i], sizeof(char)*bufferLength, cudaMemcpyDeviceToDevice);
		gpuerr();
		keyOutputP += bufferLength;
		valueOutputP += bufferLength; // divided by 4?
	}

	return totalBufferLength;
}

int remove_duplicate(Int2* keyInput, char* valueInput, Int2* &keyOutput, char* &valueOutput, int n, int* buffer) {
	char* flags;
	int* runOffsets, *runLengths;

	cudaMalloc(&keyOutput, sizeof(Int2)*n);
	gpuerr();
	cudaMalloc(&valueOutput, sizeof(char)*n);
	gpuerr();
	cudaMalloc(&flags, sizeof(char)*n);
	gpuerr();
	cudaMalloc(&runOffsets, sizeof(int)*n);
	gpuerr();
	cudaMalloc(&runLengths, sizeof(int)*n);
	gpuerr();

	// sort
	sort_key_values2(keyInput, valueInput, n);
	gpuerr();

	// make flag
	non_trivial_runs(keyInput, runOffsets, runLengths, buffer, n);
	gpuerr();
	int runLength = transfer_last_element(buffer, 1);
	gpuerr();
	cudaMemset(flags, 1, sizeof(char)*n);
	gpuerr();
	int runBlock = divideCeil(runLength, NUM_THREADS);
	non_trivial_runs_flag <<< runBlock, NUM_THREADS>>>(runOffsets, runLengths, flags, runLength);
	gpuerr();

	//filter
	double_flag(keyInput, valueInput, flags, keyOutput, valueOutput, buffer, n);
	gpuerr();

	_cudaFree(flags, runOffsets, runLengths);
	gpuerr();
	return transfer_last_element(buffer, 1);
}

int symspell_perform(SymspellArgs args, Int3* seq1, SymspellOutput* output) {
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
	Int2** pairBuffer = (Int2**)calloc(nSegment,sizeof(Int2*));
	char** distanceBuffer = (char**)calloc(nSegment,sizeof(char*));
	int* bufferLengths = (int*)calloc(nSegment,sizeof(int));

	int chunkPerSegment = divideCeil(offsetLen, nSegment);
	Int2* tempPairs;
	int tempPairLength;
	int carry = 0;
	int *pairLengthsP = pairLengths, *combinationValueOffsetsP = combinationValueOffsets;

	for (int i = 0; i < nSegment; i++) {
		// the last segment can be smaller than others
		if ((i == nSegment - 1) && (nSegment != 1)) {
			if (offsetLen % chunkPerSegment != 0)
				chunkPerSegment = offsetLen % chunkPerSegment;
		}

		tempPairLength =
		    gen_pairs(combinationValues, combinationValueOffsetsP, carry,
		              pairLengthsP, tempPairs, chunkPerSegment, deviceInt);
		bufferLengths[i] =
		    postprocessing(seq1Device, tempPairs, distance, pairBuffer[i],
		                   distanceBuffer[i], tempPairLength, deviceInt, seq1Len);
		print_tp(verbose, "4.1", tempPairLength);
		print_tp(verbose, "4.2", bufferLengths[i]);

		combinationValueOffsetsP += chunkPerSegment;
		pairLengthsP += chunkPerSegment;

		cudaFree(tempPairs);
	}

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

	//=====================================
	// step 6: transfer output to CPU
	//=====================================
	output->indexPairs = device_to_host(outputPairs, outputLen);
	output->pairwiseDistances = device_to_host(outputDistances, outputLen);
	output->len = outputLen;

	print_tp(verbose, "6", outputLen);
	_cudaFree(deviceInt, outputPairs, outputDistances);
	for (int i = 0; i < nSegment; i++)
		_cudaFree(pairBuffer[i], distanceBuffer[i]);
	_free(pairBuffer, distanceBuffer, bufferLengths);
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