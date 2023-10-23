#include "symspell.h"

int printErr(const char* str) {
	fprintf(stderr, "%s\n", str);
	return 1;
}

void printInt3(Int3* seqs, int len, char prefix) {
	int n_elements = len < 5 ? len : 5;
	for (int i = 0; i < n_elements; i++) {
		uint32_t* entry = seqs[i].entry;
		printf("%c %08X %08X %08X \n", prefix, entry[0], entry[1], entry[2]);
	}
}

void printResults(SymspellResult* results) {

}

void printArgs(SymspellArgs args) {
	printf("SymspellArgs{\n");
	printf("\tdistance: %d\n", args.distance);
	printf("\tverbose: %d\n", args.verbose);
	printf("\tseq1Len: %d\n", args.seq1Len);
	printInt3(args.seq1, args.seq1Len, '\t');
	printf("}\n");
}