#include "symspell.h"

int print_err(const char* str) {
#ifndef CATCH_TEST_MACROS_HPP_INCLUDED
	fprintf(stderr, "Error: %s\n", str);
#endif
	return ERROR;
}

void print_int3(Int3* seqs, int len, char prefix) {
	int n_elements = len < 5 ? len : 5;
	for (int i = 0; i < n_elements; i++) {
		uint32_t* entry = seqs[i].entry;
		printf("%c %08X %08X %08X \n", prefix, entry[0], entry[1], entry[2]);
	}
}

void print_results(SymspellResult* results, int len) {
	printf("SymspellResult(len=%d){\n", len);
	int n_elements = len < 5 ? len : 5;
	for (int i = 0; i < n_elements; i++) {
		SymspellResult result = results[i];
		printf("  %d %d %d \n", result.i, result.j, result.distance);
	}
	printf("}\n");
}

void print_args(SymspellArgs args) {
	printf("SymspellArgs{\n");
	printf("\tdistance: %d\n", args.distance);
	printf("\tverbose: %d\n", args.verbose);
	printf("\tseq1Len: %d\n", args.seq1Len);
	printf("\tseq1Path: \"%s\"\n", args.seq1Path);
	printf("}\n");
}