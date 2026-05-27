#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdalign.h>
#include <string.h>

////////////////////////////////////////////////////////////////////////////////

#define P               (10883)
#define N0              (2)
#define V               (71)
#define NUM_ERRORS_T    (133)
#define CATEGORY        (1)

////////////////////////////////////////////////////////////////////////////////

#define SYNDROME_TRESH_LOOKUP_TABLE { 0, 38},\
{ 2248, 39},\
{ 3292, 40},\
{ 3835, 41},\
{ 4176, 42},\
{ 4391, 43}
unsigned int synd_corrt_vec[][2]= {SYNDROME_TRESH_LOOKUP_TABLE};

#define ITERATIONS_MAX 6

////////////////////////////////////////////////////////////////////////////////

#define DIGIT                 uint64_t
#define SIGNED_DIGIT          int64_t
#define DIGIT_SIZE_B          8
#define DIGIT_SIZE_b          (DIGIT_SIZE_B << 3)
#define POS            uint32_t
#define SIGNED_POS     int32_t

////////////////////////////////////////////////////////////////////////////////

#define I32_IN_YMM            (256/32)
#define PAD32                 (I32_IN_YMM - 1)

////////////////////////////////////////////////////////////////////////////////

#define                NUM_BITS_GF2X_ELEMENT (P)
#define              NUM_DIGITS_GF2X_ELEMENT ((P+DIGIT_SIZE_b-1)/DIGIT_SIZE_b)
#define MSb_POSITION_IN_MSB_DIGIT_OF_ELEMENT ( (P % DIGIT_SIZE_b) ? (P % DIGIT_SIZE_b)-1 : DIGIT_SIZE_b-1 )

#define                NUM_BITS_GF2X_MODULUS (P+1)
#define              NUM_DIGITS_GF2X_MODULUS ((P+1+DIGIT_SIZE_b-1)/DIGIT_SIZE_b)
#define MSb_POSITION_IN_MSB_DIGIT_OF_MODULUS (P-DIGIT_SIZE_b*(NUM_DIGITS_GF2X_MODULUS-1))

#define                    INVALID_POS_VALUE (P)

////////////////////////////////////////////////////////////////////////////////

#define IS_REPRESENTABLE_IN_D_BITS(D, N) (((unsigned long) N >= (1UL << (D - 1)) && (unsigned long) N < (1UL << D)) ? D : -1)
#define BITS_TO_REPRESENT(N) (N == 0 ? 1 : (31 + IS_REPRESENTABLE_IN_D_BITS( 1, N) + IS_REPRESENTABLE_IN_D_BITS( 2, N) + IS_REPRESENTABLE_IN_D_BITS( 3, N) + IS_REPRESENTABLE_IN_D_BITS( 4, N) + IS_REPRESENTABLE_IN_D_BITS( 5, N) + IS_REPRESENTABLE_IN_D_BITS( 6, N) + IS_REPRESENTABLE_IN_D_BITS( 7, N) + IS_REPRESENTABLE_IN_D_BITS( 8, N) + IS_REPRESENTABLE_IN_D_BITS( 9, N) + IS_REPRESENTABLE_IN_D_BITS(10, N) + IS_REPRESENTABLE_IN_D_BITS(11, N) + IS_REPRESENTABLE_IN_D_BITS(12, N) + IS_REPRESENTABLE_IN_D_BITS(13, N) + IS_REPRESENTABLE_IN_D_BITS(14, N) + IS_REPRESENTABLE_IN_D_BITS(15, N) + IS_REPRESENTABLE_IN_D_BITS(16, N) + IS_REPRESENTABLE_IN_D_BITS(17, N) + IS_REPRESENTABLE_IN_D_BITS(18, N) + IS_REPRESENTABLE_IN_D_BITS(19, N) + IS_REPRESENTABLE_IN_D_BITS(20, N) + IS_REPRESENTABLE_IN_D_BITS(21, N) + IS_REPRESENTABLE_IN_D_BITS(22, N) + IS_REPRESENTABLE_IN_D_BITS(23, N) + IS_REPRESENTABLE_IN_D_BITS(24, N) + IS_REPRESENTABLE_IN_D_BITS(25, N) + IS_REPRESENTABLE_IN_D_BITS(26, N) + IS_REPRESENTABLE_IN_D_BITS(27, N) + IS_REPRESENTABLE_IN_D_BITS(28, N) + IS_REPRESENTABLE_IN_D_BITS(29, N) + IS_REPRESENTABLE_IN_D_BITS(30, N) + IS_REPRESENTABLE_IN_D_BITS(31, N) + IS_REPRESENTABLE_IN_D_BITS(32, N) ) )

////////////////////////////////////////////////////////////////////////////////





