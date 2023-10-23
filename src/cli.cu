#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "symspell.h"

const char VERSION[] = "0.0.1";
const char HELP_TEXT[] = "";

struct Args {
	int returnCode = -1;
	int distance = 1;
	char* path1 = NULL;
	char* path2 = NULL;
};

Args parseOpts(int argc, char **argv) {
	Args ans;
	char* current;

	for (int i = 1; i < argc; i++) {
		current = argv[i];
		if (current[0] != '-') {
			ans.path1 = current;
		}
		else if (strcmp(current, "-v") == 0 || strcmp(current, "--version") == 0) {
			ans.returnCode = 0;
			printf("%s", VERSION);
			return ans;
		}
		else if (strcmp(current, "-h") == 0 || strcmp(current, "--help") == 0) {
			ans.returnCode = 0;
			printf("%s", HELP_TEXT);
			return ans;
		}
		else if (strcmp(current, "-d") == 0 || strcmp(current, "--distance") == 0) {
			ans.distance = atoi(argv[++i]);
			if (ans.distance < 1 || ans.distance > MAX_DISTANCE) {
				ans.returnCode = 1;
				perror("Error: distance must be a valid number ranging from 1-4");
				return ans;
			}
		}
		else {
			ans.returnCode = 1;
			perror("Error: unknown option");
			return ans;
		}
	}
	return ans;
}

int main(int argc, char **argv) {
	Args args = parseOpts(argc, argv);
	if (args.returnCode != -1)
		return args.returnCode;

	printf("returnCode: %d\n", args.returnCode);
	printf("distance: %d\n", args.distance);
	if (args.path1 != NULL)
		printf("path1: %s\n", args.path1);
	if (args.path2 != NULL)
		printf("path2: %s\n", args.path2);

	return 0;
}