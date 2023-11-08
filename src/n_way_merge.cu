#include <vector>
#include <limits.h>
#include "codec.cu"

bool lessThan(Int2 lhs, Int2 rhs) {
	if (lhs.x != rhs.x)
		return lhs.x < rhs.x;
	return lhs.y < rhs.y;
}

void n_way_merge(Int2** pairInput, char** distanceInput, int* inputSizes,
                 SymspellOutput* output, int n) {
	if (n == 1) {
		output->indexPairs = pairInput[0];
		output->pairwiseDistances = distanceInput[0];
		output->len = inputSizes[0];
		return;
	}

	size_t* currentOffsets = (size_t*)calloc(n, sizeof(size_t));
	std::vector<Int2> pairOutput;
	std::vector<char> distanceOutput;
	Int2 minimum = {.x = INT_MAX, .y = INT_MAX};

	while (true) {
		bool hasMore = false;
		Int2 candidate = minimum;
		int candidateIndex = -1;

		// get next candidate
		for (int i = 0; i < n; i++) {
			size_t currentOffset = currentOffsets[i];
			int currentSize = inputSizes[i];
			if (currentOffset >= currentSize)
				continue;

			hasMore = true;
			Int2 currentPair = pairInput[i][currentOffset];
			if (lessThan(currentPair, candidate) || (candidateIndex == -1)) {
				candidate = currentPair;
				candidateIndex = i;
			}
		}

		// insert if not duplicate
		if ((candidate.x != minimum.x) || (candidate.y != minimum.y)) {
			size_t currentOffset = currentOffsets[candidateIndex];
			pairOutput.push_back(pairInput[candidateIndex][currentOffset]);
			distanceOutput.push_back(distanceInput[candidateIndex][currentOffset]);
		}

		// advance
		currentOffsets[candidateIndex]++;
		minimum = candidate;
		if (!hasMore)
			break;
	}

	output->len = pairOutput.size();
	output->indexPairs = (Int2*)malloc(output->len * sizeof(Int2));
	output->pairwiseDistances = (char*)malloc(output->len * sizeof(char));
	memcpy(output->indexPairs, pairOutput.data(), sizeof(Int2)*output->len);
	memcpy(output->pairwiseDistances, distanceOutput.data(), sizeof(char)*output->len);
}