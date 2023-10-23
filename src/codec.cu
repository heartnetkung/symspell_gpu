#include "util.cu"

// A CDEFGHI KLMN PQRST VW Y
const int A_CHAR = (int)'A';
const int BEFORE_A_CHAR = A_CHAR - 1;
const int Y_CHAR = (int) 'Y';

/**
 * encode character into 5 bit value (0-31).
 * -1 for non amino acid character
*/
int char_encode(char amino_acid) {
	if (amino_acid < A_CHAR || amino_acid > Y_CHAR)
		return -1;
	switch (amino_acid) {
	case 'B':
	case 'J':
	case 'O':
	case 'U':
	case 'X':
		return -1;
	default:
		return amino_acid - BEFORE_A_CHAR;
	}
}

/**
 * encode peptide string into int3 struct with 6 characters encoded into an integer.
*/
Int3 str_encode(char *str) {
	Int3 ans;
	for (int i = 0; i < MAX_INPUT_LENGTH; i++) {
		char c = str[i];
		if (c == '\0')
			break; // end

		int value = char_encode(c);
		if (value == -1) {
			ans.entry[0] = 0;
			break; // invalid character
		}

		ans.entry[i / 6] |= value << (27 - 5 * (i % 6));
	}

	return ans;
}


/**
 * decode binary form into peptide string
*/
char* str_decode(Int3 binary) {
	char* ans = (char*) malloc((MAX_INPUT_LENGTH + 1) * sizeof(char));

	for (int i = 0; i < MAX_INPUT_LENGTH; i++) {
		char c = (binary.entry[i / 6] >> (27 - 5 * (i % 6))) & 0x1F;
		if (c == 0) {
			ans[i] = '\0';
			return ans;
		}

		ans[i] = BEFORE_A_CHAR + c;
	}

	ans[MAX_INPUT_LENGTH] = '\0';
	return ans;
}