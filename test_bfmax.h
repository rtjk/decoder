#pragma once
#include "helpers.h"
#include "parameters.h"
#include "test_utils.h"
////////////////////////////////////////////////////////////////////////////////
// from test_utils:
// - population_count
// - gf2x_toggle_coeff
////////////////////////////////////////////////////////////////////////////////
#define N_REGS_H ((V + 7) / 8)
#define N_REGS_UPC (PAD8(N0 * P) / I8_IN_YMM)
////////////////////////////////////////////////////////////////////////////////
#define WORD_LEVEL_SHIFT word_level_shift_VT
#define SLACK_SIZE (DIGIT_SIZE_b-(P%DIGIT_SIZE_b))
#define SLACK_CLEAR_MASK ( ((DIGIT) 0 - 1) >> (DIGIT_SIZE_b-(P%DIGIT_SIZE_b)))
#define SLACK_EXTRACT(digit_to_extract)  (digit_to_extract >> (P%DIGIT_SIZE_b) )
#define LO_SHIFT_AMT_BITS (BITS_TO_REPRESENT(DIGIT_SIZE_b-1))
#define HI_SHIFT_AMT_BITS (BITS_TO_REPRESENT(P) - LO_SHIFT_AMT_BITS)
////////////////////////////////////////////////////////////////////////////////
POS argmax_u8(CONST uint8_t arr[PAD8(N0 * P)]) {
   /* find max */
   __m256i max_vec = _mm256_setzero_si256();
   for (int i = 0; i < N_REGS_UPC; i++) {
      __m256i v = _mm256_loadu_si256((__m256i *)&arr[i * I8_IN_YMM]);
      max_vec = _mm256_max_epu8(max_vec, v);
   }
   /* extract max from YMM using horizontal reduction */
   __m128i lo = _mm256_castsi256_si128(max_vec);
   __m128i hi = _mm256_extracti128_si256(max_vec, 1);
   __m128i m = _mm_max_epu8(lo, hi);
   m = _mm_max_epu8(m, _mm_srli_si128(m, 8));
   m = _mm_max_epu8(m, _mm_srli_si128(m, 4));
   m = _mm_max_epu8(m, _mm_srli_si128(m, 2));
   m = _mm_max_epu8(m, _mm_srli_si128(m, 1));
   uint8_t max_val = (uint8_t)_mm_extract_epi8(m, 0);
   /* find position of max */
   __m256i vmax = _mm256_set1_epi8((uint8_t)max_val);
   for (int i = 0; i < N_REGS_UPC; i++) {
      __m256i v = _mm256_loadu_si256((__m256i *)&arr[i * I8_IN_YMM]);
      __m256i cmp = _mm256_cmpeq_epi8(v, vmax);
      int mask = _mm256_movemask_epi8(cmp);
      if (mask) {
         return (POS)(i * I8_IN_YMM + __builtin_ctz(mask));
      }
   }
   return (POS)-1;
}
////////////////////////////////////////////////////////////////////////////////
static INLINE int update_syndrome_and_upcs(
   OUT uint8_t *upc, 
   IN  CONST POS Htr_sparse[N0][PAD32(V)], 
   IN  CONST POS H_sparse[N0][PAD32(V)], 
   IN  POS flip, 
   OUT uint8_t* syndrome_bits,
   IN  int hw)
{
   int flip_block = flip / P;
   int flip_bit = flip - flip_block * P;
   __m256i vp = _mm256_set1_epi32((uint32_t)P);
   __m256i vpos = _mm256_set1_epi32((uint32_t)flip_bit);
   /* vectorize H_sparse */
   __m256i v_H_row[N0][N_REGS_H];
   for (int block = 0; block < N0; block++) {
      for (int r = 0; r < N_REGS_H; r++) {
         v_H_row[block][r] = _mm256_loadu_si256((__m256i *)&H_sparse[block][r * 8]);
      }
   }
   /* update syndrome and save upc positions to update */
   __m256i up_pos[V][N0][N_REGS_H];
   int up_sign[V];
   for (int col_reg = 0; col_reg < N_REGS_H; col_reg++) {
      /* get the column of H corresponding to the flipped bit */
      uint32_t tmp[8] = {0};
      __m256i htr = _mm256_loadu_si256((__m256i *)&Htr_sparse[flip_block][col_reg * 8]);
      __m256i sum = _mm256_add_epi32(htr, vpos);
      __m256i sub = _mm256_sub_epi32(sum, vp);
      __m256i res = _mm256_min_epu32(sum, sub);
      _mm256_storeu_si256((__m256i *)tmp, res);
      /* scan each idx in the column */
      for (int i = 0; (i < 8) && (col_reg * 8 + i < V); i++) {
         /* update the syndrome */
         POS row = tmp[i];
         syndrome_bits[row] ^= 1;
         int delta = (syndrome_bits[row] == 0) ? -1 : 1;
         hw += delta;
         up_sign[col_reg * 8 + i] = delta;
         /* save upc positions to update (faster than updating upcs directly) */
         __m256i vrow = _mm256_set1_epi32((uint32_t)row);
         for (int block = 0; block < N0; block++) {
            for (int row_reg = 0; row_reg < N_REGS_H; row_reg++) {
               __m256i col = _mm256_add_epi32(v_H_row[block][row_reg], vrow);
               __m256i sub = _mm256_sub_epi32(col, vp);
               __m256i res = _mm256_min_epu32(col, sub);
               up_pos[col_reg * 8 + i][block][row_reg] = res;
            }
         }
      }
   }
   /* update upcs */
   uint32_t *up_pos_u32 = (uint32_t *)up_pos;
   for (int i = 0; i < V; i++) {
      int delta = up_sign[i];
      for (int block = 0; block < N0; block++) {
         for (int row_reg = 0; row_reg < N_REGS_H; row_reg++) {
            for (int lane = 0; (lane < 8) && (row_reg * 8 + lane < V); lane++) {
               int up_idx = (((i * N0 + block) * N_REGS_H + row_reg) * 8) + lane;
               uint32_t col = up_pos_u32[up_idx];
               upc[block * P + col] += delta;
            }
         }
      }
   }
   return hw;
}
////////////////////////////////////////////////////////////////////////////////
static INLINE DIGIT get_coeff(
   IN DIGIT poly[],
   IN unsigned int exponent) {
   unsigned int straightIdx = (NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_b -1) - exponent;
   unsigned int digitIdx = straightIdx / DIGIT_SIZE_b;
   unsigned int inDigitIdx = straightIdx % DIGIT_SIZE_b;
   return (poly[digitIdx] >> (DIGIT_SIZE_b-1-inDigitIdx)) & ((DIGIT) 1) ;
}
////////////////////////////////////////////////////////////////////////////////
static INLINE int hamming_weight(
   IN uint8_t a[P])
{
   int hw = 0;
   for (int i = 0; i < P; i++) hw += a[i];
   return hw;
}
////////////////////////////////////////////////////////////////////////////////
/* for r in rows
 *   for c in columns
 *     upc[c] += s[r] & H[r][c]
 */
static INLINE void compute_upcs(
   OUT uint8_t upc[PAD8(N0 * P)],
   IN  POS Htr_sparse[N0][PAD32(V)],
   IN  uint8_t syndrome_bits[P])
{
   /* for each block */
   for (int block = 0; block < N0; block++) {
      uint8_t *upc_block = &upc[block * P];
      /* scan each idx in the first column */
      for (int i = 0; i < V; i++) {
         int idx = Htr_sparse[block][i];
         int wrap = P - idx;
         /* increment idx to shift the column (first half of upcs) */
         for (int j = 0; j < wrap; j++) {
            upc_block[j] += syndrome_bits[idx + j];
         }
         /* increment idx to shift the column (second half of upcs) */
         for (int k = wrap; k < wrap + idx; k++) {
            upc_block[k] += syndrome_bits[k - wrap];
         }
      }
   }
}
////////////////////////////////////////////////////////////////////////////////
static INLINE void dense_to_u8(
   OUT uint8_t u8[],
   IN  DIGIT dense[],
   IN  int len)
{
   for (int i = 0; i < len; i++) {
      u8[i] = get_coeff(dense, i);
   }
}
////////////////////////////////////////////////////////////////////////////////
int bfmax_decoder(
   OUT DIGIT error[N0*NUM_DIGITS_GF2X_ELEMENT], 
   IN  POS Htr_sparse[N0][PAD32(V)], 
   IN  POS H_sparse[N0][PAD32(V)], 
   IN  DIGIT syndrome[NUM_DIGITS_GF2X_ELEMENT])
{
   /* expand each syndome bit to u8 */
   uint8_t syndrome_bits[P];
   dense_to_u8(syndrome_bits, syndrome, P);
   /* compute unsatisfied parity checks */
   ALIGNED uint8_t upc[PAD8(N0 * P)] = {0};
   compute_upcs(upc, Htr_sparse, syndrome_bits);
   /* decoding iterations */
   int iter = 0;
   int hw = population_count(syndrome);
   do {
      POS col = argmax_u8(upc);
      int col_block = col / P;
      int col_bit = col % P;
      gf2x_toggle_coeff(error + col_block * NUM_DIGITS_GF2X_ELEMENT, col_bit);
      hw = update_syndrome_and_upcs(upc, Htr_sparse, H_sparse, col, syndrome_bits, hw);
      DEBUG_PRINT("i: %d \t hw(s): %d \n", iter, hw);
      iter++;
   } while ((iter < 1.5 * NUM_ERRORS_T) && (hw != 0));
   return 1;
}
////////////////////////////////////////////////////////////////////////////////

