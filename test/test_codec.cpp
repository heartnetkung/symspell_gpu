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
	REQUIRE(output.entry[0]==0x08000000);
	REQUIRE(output.entry[1]==0);
	REQUIRE(output.entry[2]==0);

	char input2[] = "AC";
	REQUIRE(strcmp(str_decode(str_encode(input2)), input2) == 0);
}