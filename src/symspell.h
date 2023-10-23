#include <stdint.h>

const int MAX_INPUT_LENGTH = 18;
const int MAX_DISTANCE = 4;

struct Int3 {
	uint32_t entry[3] = {0L, 0L, 0L};
};

struct SymspellArgs {
	int distance = 1;
	int verbose = 0;
	char* seqPath = NULL;
	int seq1Len = 0;
	Int3* seq1 = NULL;
	// Int3* seq2 = NULL;
	// int seq2Len = 0;
};
//TODO add seq2 later


struct SymspellResult {
	int i;
	int j;
	int distance;
};