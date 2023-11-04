#include "symspell.h"

int print_err(const char* str) {
#ifndef TEST_ENV
	fprintf(stderr, "Error: %s\n", str);
#endif
	return ERROR;
}

void print_int3(Int3* seqs, int len, char prefix) {
	int n_elements = len < 5 ? len : 5;
	for (int i = 0; i < n_elements; i++) {
		unsigned int* entry = seqs[i].entry;
		printf("%c %08X %08X %08X \n", prefix, entry[0], entry[1], entry[2]);
	}
}

void print_args(SymspellArgs args) {
	printf("SymspellArgs{\n");
	printf("\tdistance: %d\n", args.distance);
	printf("\tverbose: %d\n", args.verbose);
	printf("\tseq1Len: %d\n", args.seq1Len);
	printf("\tseq1Path: \"%s\"\n", args.seq1Path);
	printf("}\n");
}

void print_int_arr(int* arr, int n) {
	for (int i = 0; i < n; i++)
		printf("%d ", arr[i]);
	printf("\n");
}

void print_char_arr(char* arr, int n) {
	for (int i = 0; i < n; i++)
		printf("%d ", arr[i]);
	printf("\n");
}

void print_int2_arr(Int2* arr, int n) {
	for (int i = 0; i < n; i++)
		printf("(%d,%d) ", arr[i].x, arr[i].y);
	printf("\n");
}

void _cudaFree(void* a) {
	cudaFree(a);
}
void _cudaFree(void* a, void* b) {
	cudaFree(a);
	cudaFree(b);
}
void _cudaFree(void* a, void* b, void* c) {
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}
void _cudaFree(void* a, void* b, void* c, void* d) {
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	cudaFree(d);
}
void _cudaFree(void* a, void* b, void* c, void* d, void* e) {
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	cudaFree(d);
	cudaFree(e);
}
void _cudaFree(void* a, void* b, void* c, void* d, void* e, void* f) {
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	cudaFree(d);
	cudaFree(e);
	cudaFree(f);
}
void _cudaFree(void* a, void* b, void* c, void* d, void* e, void* f, void* g) {
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	cudaFree(d);
	cudaFree(e);
	cudaFree(g);
}