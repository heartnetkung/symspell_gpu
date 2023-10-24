#include <catch2/catch_test_macros.hpp>
#include <stdio.h>
#include <string.h>
#include "../src/codec.cu"

TEST_CASE( "char_encode()", "[codec]" ) {
	REQUIRE(char_encode('A') == 1 );
	REQUIRE(char_encode('X') == -1 );
	REQUIRE(char_encode('0') == -1 );
	REQUIRE(char_encode('Y') == 25 );
}

TEST_CASE( "str_encode()", "[codec]" ) {
	char input[] = "A";
	Int3 output = str_encode(input);
	REQUIRE(output.entry[0] == 0x08000000);
	REQUIRE(output.entry[1] == 0);
	REQUIRE(output.entry[2] == 0);

	char input2[] = "AC";
	REQUIRE(strcmp(str_decode(str_encode(input2)), input2) == 0);
}

TEST_CASE( "len_decode()", "[codec]" ) {
	char input[] = "ACC";
	Int3 output = str_encode(input);
	REQUIRE(len_decode(output) == 3);
}

TEST_CASE("remove_char()", "[codec]") {
	char input[] = "ACDEFGHIKLMNPQRSTV";
	Int3 binForm = str_encode(input);

	REQUIRE(strcmp(str_decode(remove_char(binForm,0)),"CDEFGHIKLMNPQRSTV") == 0);
	REQUIRE(strcmp(str_decode(remove_char(binForm,1)),"ADEFGHIKLMNPQRSTV") == 0);
	REQUIRE(strcmp(str_decode(remove_char(binForm,2)),"ACEFGHIKLMNPQRSTV") == 0);
	REQUIRE(strcmp(str_decode(remove_char(binForm,3)),"ACDFGHIKLMNPQRSTV") == 0);
	REQUIRE(strcmp(str_decode(remove_char(binForm,4)),"ACDEGHIKLMNPQRSTV") == 0);

	REQUIRE(strcmp(str_decode(remove_char(binForm,5)),"ACDEFHIKLMNPQRSTV") == 0);
	REQUIRE(strcmp(str_decode(remove_char(binForm,6)),"ACDEFGIKLMNPQRSTV") == 0);
	REQUIRE(strcmp(str_decode(remove_char(binForm,7)),"ACDEFGHKLMNPQRSTV") == 0);
	REQUIRE(strcmp(str_decode(remove_char(binForm,8)),"ACDEFGHILMNPQRSTV") == 0);
	REQUIRE(strcmp(str_decode(remove_char(binForm,9)),"ACDEFGHIKMNPQRSTV") == 0);

	REQUIRE(strcmp(str_decode(remove_char(binForm,10)),"ACDEFGHIKLNPQRSTV") == 0);
	REQUIRE(strcmp(str_decode(remove_char(binForm,11)),"ACDEFGHIKLMPQRSTV") == 0);
	REQUIRE(strcmp(str_decode(remove_char(binForm,12)),"ACDEFGHIKLMNQRSTV") == 0);
	REQUIRE(strcmp(str_decode(remove_char(binForm,13)),"ACDEFGHIKLMNPRSTV") == 0);
	REQUIRE(strcmp(str_decode(remove_char(binForm,14)),"ACDEFGHIKLMNPQSTV") == 0);

	REQUIRE(strcmp(str_decode(remove_char(binForm,15)),"ACDEFGHIKLMNPQRTV") == 0);
	REQUIRE(strcmp(str_decode(remove_char(binForm,16)),"ACDEFGHIKLMNPQRSV") == 0);
	REQUIRE(strcmp(str_decode(remove_char(binForm,17)),"ACDEFGHIKLMNPQRST") == 0);

	char input2[] = "ACD";
	Int3 binForm2 = str_encode(input2);
	REQUIRE(strcmp(str_decode(remove_char(binForm2,0)),"CD") == 0);
	REQUIRE(strcmp(str_decode(remove_char(binForm2,1)),"AD") == 0);
	REQUIRE(strcmp(str_decode(remove_char(binForm2,2)),"AC") == 0);
}


char FILE1[] = "../test/sample/input1.txt";
char FILE2[] = "../test/sample/input2.txt";
char FILE3[] = "../test/sample/input3.txt";

TEST_CASE( "parseFile()", "[codec]" ) {

	int seq1Len = 3;
	Int3* seq1 = (Int3*)malloc(sizeof(Int3) * seq1Len);
	int result = parse_file(FILE1, seq1Len, seq1);
	REQUIRE(result == SUCCESS);
	REQUIRE(strcmp(str_decode(seq1[0]),"CAAA")==0);
	REQUIRE(strcmp(str_decode(seq1[1]),"AAAA")==0);
	REQUIRE(strcmp(str_decode(seq1[2]),"AAAD")==0);
	free(seq1);

	int seq2Len = 3;
	Int3* seq2 = (Int3*)malloc(sizeof(Int3) * seq2Len);
	result = parse_file(FILE2, seq2Len, seq2);
	REQUIRE(result == SUCCESS);
	REQUIRE(strcmp(str_decode(seq2[0]),"CAAA")==0);
	REQUIRE(strcmp(str_decode(seq2[1]),"AAAA")==0);
	REQUIRE(strcmp(str_decode(seq2[2]),"AAAD")==0);
	free(seq2);

	int seq3Len = 2;
	Int3* seq3 = (Int3*)malloc(sizeof(Int3) * seq3Len);
	result = parse_file(FILE1, seq3Len, seq3);
	REQUIRE(result == ERROR);
	free(seq3);

	int seq4Len = 3;
	Int3* seq4 = (Int3*)malloc(sizeof(Int3) * seq4Len);
	result = parse_file(FILE3, seq4Len, seq4);
	REQUIRE(result == ERROR);
	free(seq4);
}