#include <catch2/catch_test_macros.hpp>
#include <stdio.h>
#include "../src/codec.cu"
#include <string.h>

TEST_CASE( "char_encode", "[add]" ) {
	REQUIRE(char_encode('A') == 1 );
	REQUIRE(char_encode('X') == -1 );
	REQUIRE(char_encode('0') == -1 );
	REQUIRE(char_encode('Y') == 25 );
}

TEST_CASE( "str_encode", "[add]" ) {
	char input2[] = "AC";
	REQUIRE(strcmp(str_decode(str_encode(input2)), input2) == 0);
}