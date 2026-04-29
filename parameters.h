#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdalign.h>
#include <string.h>

////////////////////////////////////////////////////////////////////////////////

#define P (10883)
#define N0 (2)
#define V (71)
#define NUM_ERRORS_T (133)

////////////////////////////////////////////////////////////////////////////////

#define IS_REPRESENTABLE_IN_D_BITS(D, N) (((unsigned long) N >= (1UL << (D - 1)) && (unsigned long) N < (1UL << D)) ? D : -1)
#define BITS_TO_REPRESENT(N) (N == 0 ? 1 : (31 + IS_REPRESENTABLE_IN_D_BITS( 1, N) + IS_REPRESENTABLE_IN_D_BITS( 2, N) + IS_REPRESENTABLE_IN_D_BITS( 3, N) + IS_REPRESENTABLE_IN_D_BITS( 4, N) + IS_REPRESENTABLE_IN_D_BITS( 5, N) + IS_REPRESENTABLE_IN_D_BITS( 6, N) + IS_REPRESENTABLE_IN_D_BITS( 7, N) + IS_REPRESENTABLE_IN_D_BITS( 8, N) + IS_REPRESENTABLE_IN_D_BITS( 9, N) + IS_REPRESENTABLE_IN_D_BITS(10, N) + IS_REPRESENTABLE_IN_D_BITS(11, N) + IS_REPRESENTABLE_IN_D_BITS(12, N) + IS_REPRESENTABLE_IN_D_BITS(13, N) + IS_REPRESENTABLE_IN_D_BITS(14, N) + IS_REPRESENTABLE_IN_D_BITS(15, N) + IS_REPRESENTABLE_IN_D_BITS(16, N) + IS_REPRESENTABLE_IN_D_BITS(17, N) + IS_REPRESENTABLE_IN_D_BITS(18, N) + IS_REPRESENTABLE_IN_D_BITS(19, N) + IS_REPRESENTABLE_IN_D_BITS(20, N) + IS_REPRESENTABLE_IN_D_BITS(21, N) + IS_REPRESENTABLE_IN_D_BITS(22, N) + IS_REPRESENTABLE_IN_D_BITS(23, N) + IS_REPRESENTABLE_IN_D_BITS(24, N) + IS_REPRESENTABLE_IN_D_BITS(25, N) + IS_REPRESENTABLE_IN_D_BITS(26, N) + IS_REPRESENTABLE_IN_D_BITS(27, N) + IS_REPRESENTABLE_IN_D_BITS(28, N) + IS_REPRESENTABLE_IN_D_BITS(29, N) + IS_REPRESENTABLE_IN_D_BITS(30, N) + IS_REPRESENTABLE_IN_D_BITS(31, N) + IS_REPRESENTABLE_IN_D_BITS(32, N) ) )

#define DIGIT uint64_t
#define DIGIT_SIZE_B (8)
#define DIGIT_SIZE_b (DIGIT_SIZE_B << 3)
#define NUM_BITS_GF2X_ELEMENT (P)
#define NUM_DIGITS_GF2X_ELEMENT ((P+DIGIT_SIZE_b-1)/DIGIT_SIZE_b)
#define BITSLICED_OPERAND_WIDTH (BITS_TO_REPRESENT(V)+1)
#define SLICE_TYPE __m256i
#define NUM_BITS_IN_BITSLICED_OP (256)
#define NUM_SLICES_GF2X_ELEMENT ( (NUM_DIGITS_GF2X_ELEMENT+3)/ (NUM_BITS_IN_BITSLICED_OP/DIGIT_SIZE_b) )
#define POSITION_T uint32_t
#define SIGNED_POSITION_T int32_t
#define SLACK_SIZE (DIGIT_SIZE_b-(P%DIGIT_SIZE_b))
#define SLACK_CLEAR_MASK ( ((DIGIT) 0 - 1) >> (DIGIT_SIZE_b-(P%DIGIT_SIZE_b)))
#define SLACK_EXTRACT(digit_to_extract)  (digit_to_extract >> (P%DIGIT_SIZE_b) )
#define LO_SHIFT_AMT_BITS (BITS_TO_REPRESENT(DIGIT_SIZE_b-1))
#define HI_SHIFT_AMT_BITS (BITS_TO_REPRESENT(P) - LO_SHIFT_AMT_BITS)

#define N_REGS ((V + 7) / 8)

typedef struct {
   SLICE_TYPE slice[BITSLICED_OPERAND_WIDTH];
} bs_operand_t;

#define WORD_LEVEL_SHIFT word_level_shift_VT

////////////////////////////////////////////////////////////////////////////////

#ifdef SKIP_INLINE
#define INLINE
#else
#define INLINE inline
#endif

#define RESTRICT

////////////////////////////////////////////////////////////////////////////////