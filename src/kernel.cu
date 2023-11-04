#include "codec.cu"

int transfer_last_element(int* deviceArr, int n) {
	int ans[1];
	cudaMemcpy(ans, deviceArr + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	return ans[0];
}

__global__
void cal_combination_len(Int3* input, int distance, int* output, int n) {
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

__global__
void cal_pair_len(int* input, int* output, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	output[tid] = input[tid] * (input[tid] - 1) / 2;
}

__global__
void generate_pairs(int* indexes, Int2* outputs, int* inputOffsets, int* outputOffsets, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	int start = tid == 0 ? 0 : inputOffsets[tid - 1];
	int end = inputOffsets[tid];
	int outputIndex = tid == 0 ? 0 : outputOffsets[tid - 1];

	for (int i = start; i < end; i++) {
		for (int j = i + 1; j < end; j++) {
			Int2 newValue;
			if (indexes[i] < indexes[j]) {
				newValue.x = indexes[i];
				newValue.y = indexes[j];
			} else {
				newValue.x = indexes[j];
				newValue.y = indexes[i];
			}
			outputs[outputIndex++] = newValue;
		}
	}
}

__global__
void cal_levenshtein(Int3* seq, Int2* index, int distance, char* distanceOutput, char* flagOutput, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n)
		return;

	Int2 indexPair = index[tid];
	char newOutput = levenshtein(seq[indexPair.x], seq[indexPair.y]);
	distanceOutput[tid] = newOutput;
	flagOutput[tid] =  newOutput <= distance;
}

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

__device__
char levenshtein(Int3 x1, Int3 x2) {
	char x, y, lastdiag, olddiag, c1, c2;
	char s1[MAX_INPUT_LENGTH + 1] = str_decode(x1);
	char s2[MAX_INPUT_LENGTH + 1] = str_decode(x2);
	char s1len = (char)len_decode(x1), s2len = (char)len_decode(x2);
	char column[MAX_INPUT_LENGTH + 1];

	for (y = 1; y <= s1len; y++)
		column[y] = y;
	for (x = 1; x <= s2len; x++) {
		column[0] = x;
		for (y = 1, lastdiag = x - 1; y <= s1len; y++) {
			olddiag = column[y];
			column[y] = MIN3(column[y] + 1, column[y - 1] + 1, lastdiag + (s1[y - 1] == s2[x - 1] ? 0 : 1));
			lastdiag = olddiag;
		}
	}
	return column[s1len];
}
