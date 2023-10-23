#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <regex.h>
#include "codec.cu"

const char VERSION[] = "0.0.1\n";
const char HELP_TEXT[] = "symspell_gpu\n"
                         "\t description: perform symspell algorithm for near neighbor search / clustering of T cell receptor's CDR3 sequences\n"
                         "\t -v or --version: print the version of the program then exit\n"
                         "\t -h or --help: print the help text of the program then exit\n"
                         "\t -V or --verbose: print extra detail as the program runs for debugging purpose\n"
                         "\t -d or --distance [number]: set the distance threshold defining the neighbor\n"
                         "\t -p or --input-path [str] (required): set the path of input file which is a text file containing one CDR3 sequence per line\n"
                         "\t -n or --input-length [number] (required): set the number of sequences given in the input file\n";


int parse_file(char* path, int arrLen, Int3* result) {
	FILE* file = fopen(path, "r");
	const int BUFFER_SIZE = 256;
	char line[BUFFER_SIZE];

	int count = 0;

	while (fgets(line, BUFFER_SIZE, file)) {
		printf("%d %s\n", count, line);
		str_encode(line);
		count++;
	}

	fclose(file);
	return 0;
}

int parse_opts(int argc, char **argv, SymspellArgs ans) {
	char* current;

	for (int i = 1; i < argc; i++) {
		current = argv[i];

		if (strcmp(current, "-v") == 0 || strcmp(current, "--version") == 0) {
			printf("%s", VERSION);
			return 0;
		}
		else if (strcmp(current, "-h") == 0 || strcmp(current, "--help") == 0) {
			printf("%s", HELP_TEXT);
			return 0;
		}
		else if (strcmp(current, "-V") == 0 || strcmp(current, "--verbose") == 0) {
			ans.verbose = 1;
		}
		else if (strcmp(current, "-d") == 0 || strcmp(current, "--distance") == 0) {
			ans.distance = atoi(argv[++i]);
			if (ans.distance < 1 || ans.distance > MAX_DISTANCE)
				return print_err("Error: distance must be a valid number ranging from 1-4");
		}
		else if (strcmp(current, "-p") == 0 || strcmp(current, "--input-path") == 0)
			ans.seq1Path = argv[++i];
		else if (strcmp(current, "-n") == 0 || strcmp(current, "--input-length") == 0) {
			ans.seq1Len = atoi(argv[++i]);
			if (ans.seq1Len == 0)
				return print_err("Error: invalid input length");
		}
		else
			return print_err("Error: unknown option");
	}

	if (ans.seq1Path == NULL)
		return print_err("Error: missing path for seq1");
	if (ans.seq1Len == 0)
		return print_err("Error: missing length for seq1");

	return -1;
}


int main(int argc, char **argv) {
	SymspellArgs args;
	int returnCode = parse_opts(argc, argv, args);
	if (returnCode != -1)
		return returnCode;

	if (args.verbose)
		print_args(args);

	return 0;
}