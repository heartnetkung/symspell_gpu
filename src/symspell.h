const int MAX_INPUT_LENGTH = 18;
// up to 3060? or 816?
const int MAX_DISTANCE = 4;

struct Int3 {
	unsigned int entry[3] = {0L, 0L, 0L};
};

struct Int2 {
	unsigned int x = 0L, y = 0L;
};

struct SymspellArgs {
	int distance = 1;
	int verbose = 0;
	char* seq1Path = NULL;
	int seq1Len = 0;
	char* outputPath = NULL;
	int checkOutput = 0;
	// Int3* seq2 = NULL;
	// int seq2Len = 0;
};

struct SymspellOutput {
	Int2* indexPairs = NULL;
	int* pairwiseDistances = NULL;
	int len = 0;
};

enum ReturnCode {SUCCESS, ERROR, EXIT};

int symspell_perform(SymspellArgs args, Int3* seq1, SymspellOutput* output);
void symspell_free(SymspellOutput* output);
