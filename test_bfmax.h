#pragma once
#include "helpers.h"
#include "parameters.h"
#include "test_utils.h"
////////////////////////////////////////////////////////////////////////////////
// from test_utils:
// - population_count
// - gf2x_toggle_coeff
////////////////////////////////////////////////////////////////////////////////
#define N_REGS_H   (PAD32(V) / I32_IN_YMM)
#define N_REGS_UPC (PAD8(N0 * P) / I8_IN_YMM)
////////////////////////////////////////////////////////////////////////////////
#define WORD_LEVEL_SHIFT word_level_shift_VT
#define SLACK_SIZE (DIGIT_SIZE_b - (P % DIGIT_SIZE_b))
#define SLACK_CLEAR_MASK (((DIGIT)0 - 1) >> (DIGIT_SIZE_b - (P % DIGIT_SIZE_b)))
#define SLACK_EXTRACT(digit_to_extract) (digit_to_extract >> (P % DIGIT_SIZE_b))
#define LO_SHIFT_AMT_BITS (BITS_TO_REPRESENT(DIGIT_SIZE_b - 1))
#define HI_SHIFT_AMT_BITS (BITS_TO_REPRESENT(P) - LO_SHIFT_AMT_BITS)
////////////////////////////////////////////////////////////////////////////////
static INLINE POS argmax_u8(CONST uint8_t arr[PAD8(N0 * P)]) {
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
   OUT uint8_t upc[PAD8(N0 * P)], 
   IN  CONST POS Htr_sparse[N0][PAD32(V)], 
   IN  CONST POS H_sparse[N0][PAD32(V)], 
   IN  POS flip, 
   OUT uint8_t syndrome_bits[P],
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
      uint32_t tmp[I32_IN_YMM] = {0};
      __m256i htr = _mm256_loadu_si256((__m256i *)&Htr_sparse[flip_block][col_reg * I32_IN_YMM]);
      __m256i sum = _mm256_add_epi32(htr, vpos);
      __m256i sub = _mm256_sub_epi32(sum, vp);
      __m256i res = _mm256_min_epu32(sum, sub);
      _mm256_storeu_si256((__m256i *)tmp, res);
      /* scan each idx in the column */
      for (int i = 0; (i < I32_IN_YMM) && (col_reg * I32_IN_YMM + i < V); i++) {
         /* update the syndrome */
         POS row = tmp[i];
         syndrome_bits[row] ^= 1;
         int delta = (syndrome_bits[row] == 0) ? -1 : 1;
         hw += delta;
         up_sign[col_reg * I32_IN_YMM + i] = delta;
         /* save upc positions to update (faster than updating upcs directly) */
         __m256i vrow = _mm256_set1_epi32((uint32_t)row);
         for (int block = 0; block < N0; block++) {
            for (int row_reg = 0; row_reg < N_REGS_H; row_reg++) {
               __m256i col = _mm256_add_epi32(v_H_row[block][row_reg], vrow);
               __m256i sub = _mm256_sub_epi32(col, vp);
               __m256i res = _mm256_min_epu32(col, sub);
               up_pos[col_reg * I32_IN_YMM + i][block][row_reg] = res;
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
   IN unsigned int exponent)
{
   unsigned int straightIdx = (NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_b -1) - exponent;
   unsigned int digitIdx = straightIdx / DIGIT_SIZE_b;
   unsigned int inDigitIdx = straightIdx % DIGIT_SIZE_b;
   return (poly[digitIdx] >> (DIGIT_SIZE_b-1-inDigitIdx)) & ((DIGIT) 1) ;
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
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define BITS_FOR_V (BITS_TO_REPRESENT(V))                                                                               // 7
#define SLICE __m256i
#define NUM_BITS_IN_BITSLICED_OP (256)
#define SLICES_OF_P ((NUM_DIGITS_GF2X_ELEMENT + 3) / (NUM_BITS_IN_BITSLICED_OP / DIGIT_SIZE_b))                         // 43
////////////////////////////////////////////////////////////////////////////////
typedef struct {
   SLICE slice[BITS_FOR_V];
} SLICE_BUNDLE;
////////////////////////////////////////////////////////////////////////////////
static INLINE SLICE_BUNDLE bs_increment(SLICE_BUNDLE a, SLICE b)
{
   SLICE_BUNDLE result;
   SLICE carry = _mm256_and_si256(a.slice[0], b);
   result.slice[0] = _mm256_xor_si256(a.slice[0], b);
   for (int i = 1; i < BITS_FOR_V; i++)
   {
      SLICE ai = a.slice[i];
      result.slice[i] = _mm256_xor_si256(ai, carry);
      carry = _mm256_and_si256(ai, carry);
   }
   return result;
}
////////////////////////////////////////////////////////////////////////////////
static INLINE SLICE_BUNDLE bs_decrement(SLICE_BUNDLE a, SLICE b)
{
   SLICE_BUNDLE result;
   SLICE borrow = _mm256_andnot_si256(a.slice[0], b);
   result.slice[0] = _mm256_xor_si256(a.slice[0], b);
   for (int i = 1; i < BITS_FOR_V; i++) {
      SLICE ai = a.slice[i];
      result.slice[i] = _mm256_xor_si256(ai, borrow);
      borrow = _mm256_andnot_si256(ai, borrow);
   }
   return result;
}
////////////////////////////////////////////////////////////////////////////////
void word_level_shift_VT(DIGIT *RESTRICT shifted_param,
                         POS high_shift_amt,
                         DIGIT *RESTRICT to_shift)
{
   /* condit-pull whole digits towards the MSB, starting from the
    *high_shift_amt - th  one, that is including the one which will have slack */
   {
      int j = 0;
      __m256i tmp;
      /*if there is enough material to have at least a full AVX2 reg to shift*/
      if(NUM_DIGITS_GF2X_ELEMENT-high_shift_amt >=4) {
         for (; j < NUM_DIGITS_GF2X_ELEMENT-high_shift_amt-3; j = j+4) {
            tmp = _mm256_lddqu_si256 ((__m256i *) (to_shift+j+high_shift_amt));
            _mm256_storeu_si256(  (__m256i *) (shifted_param+j), tmp);
         }
      }
      for (; j < NUM_DIGITS_GF2X_ELEMENT-high_shift_amt; j++) {
         shifted_param[j] = to_shift[j+high_shift_amt];
      }
   }
   /* collect the slack carryover */
   DIGIT slack_carryover = SLACK_EXTRACT(shifted_param[0]);
   shifted_param[0] &= SLACK_CLEAR_MASK;
   /* move the remaining topmost high_shift_amt words (0th to high_shift_amt-1
    * one) taking care of tucking in the slack carryover */
   if(high_shift_amt >=4) {
      /* I have enough word material to move to gain something substantial
       * via AVX2*/
      DIGIT next_carryover;
      int j;
      for (j = high_shift_amt-4 ; j >=0 ; j = j-4) {
         __m256i in_motion = _mm256_lddqu_si256((__m256i *) (to_shift+j));
         __m256i hi_tgt_part  = _mm256_slli_epi64(in_motion, SLACK_SIZE  );
         __m256i low_tgt_part = _mm256_srli_epi64(in_motion,(DIGIT_SIZE_b - SLACK_SIZE));
         next_carryover = _mm256_extract_epi64 (low_tgt_part, 0);
         low_tgt_part = _mm256_insert_epi64 (low_tgt_part,slack_carryover, 0);
         slack_carryover = next_carryover;
         low_tgt_part = _mm256_permute4x64_epi64 (low_tgt_part, 0x39);
         int target_idx = NUM_DIGITS_GF2X_ELEMENT-high_shift_amt+j;
         __m256i final = _mm256_or_si256(hi_tgt_part,low_tgt_part);
         _mm256_storeu_si256( (__m256i *) (shifted_param+target_idx),final);
      }
      j+=3;
      for (; j >=0 ; j--) {
         int target_idx = NUM_DIGITS_GF2X_ELEMENT-high_shift_amt+j;
         DIGIT to_write = slack_carryover;
         to_write = to_write | (to_shift[j] << SLACK_SIZE);
         slack_carryover = SLACK_EXTRACT(to_shift[j]);
         shifted_param[target_idx] = to_write;
      }
      /* handle trailing words */
   } else {
      for (int j = high_shift_amt-1 ; j >=0 ; j--) {
         int target_idx = NUM_DIGITS_GF2X_ELEMENT-high_shift_amt+j;
         DIGIT to_write = slack_carryover;
         to_write = to_write | (to_shift[j] << SLACK_SIZE);
         slack_carryover = SLACK_EXTRACT(to_shift[j]);
         shifted_param[target_idx] = to_write;
      }
   }
}
////////////////////////////////////////////////////////////////////////////////
void gf2x_mod_mul_monom(DIGIT shifted[],
                        POS shift_amt,
                        CONST DIGIT to_shift[])
{
   DIGIT mask;
   /*shift_amt is split bitwise :  |------------------shift_amt------------------|
    *                              | inter word shift amt | intra word shift amt |
    *                                  HI_SHIFT_AMT_BITS     LO_SHIFT_AMT_BITS
    */
   /* inter word shifting, done speculatively shifting the entire operand by a
    * power of two, and conditionally committing the result */
   POS high_shift_amt = shift_amt >> LO_SHIFT_AMT_BITS;
   POS low_shift_amt = shift_amt & (((POS)1 << LO_SHIFT_AMT_BITS)
                                           -1);
   WORD_LEVEL_SHIFT(shifted,high_shift_amt,(DIGIT * RESTRICT)to_shift);
   /* cyclic shifts inside DIGITs */
   /* extract low_shift_amt MSB for cyclic shift */
   DIGIT carryover = (shifted[0] << SLACK_SIZE) | (shifted[1] >> (DIGIT_SIZE_b -SLACK_SIZE));
   carryover = carryover >> (DIGIT_SIZE_b - low_shift_amt);
   /* pure shift, carried over from left_bit_shift_n*/
   mask = ~(( (DIGIT)1 << (DIGIT_SIZE_b - low_shift_amt) ) - 1);
   /* must deal with C99 UB when shifting by variable size*/
   DIGIT zeroshift_mask = (DIGIT)0 - (!!(low_shift_amt));
   {
      int j;
      __m256i a,b;
      for(j = 0 ; j < NUM_DIGITS_GF2X_ELEMENT-4; j = j+4) {
         a = _mm256_lddqu_si256( (__m256i *) &shifted[0] +
                                 j/4);  //load from in[j] to in[j+3]
         b = _mm256_lddqu_si256( (__m256i *) &shifted[1] +
                                 j/4);  //load from in[j+1] to in[j+4]
         a = _mm256_slli_epi64(a, low_shift_amt);
         b = _mm256_srli_epi64(b, (DIGIT_SIZE_b-low_shift_amt));
         /* no need to zeromask, the srli behavior is well defined for amount > 63 */
         _mm256_storeu_si256( (__m256i *) &shifted[0] + j/4, _mm256_or_si256(a,b));
      }
      for (; j < NUM_DIGITS_GF2X_ELEMENT-1 ; j++) {
         shifted[j] = (shifted[j] << low_shift_amt) |
                      ( zeroshift_mask & (shifted[j+1] & mask) >> (DIGIT_SIZE_b - low_shift_amt) );
      }
   }
   shifted[NUM_DIGITS_GF2X_ELEMENT-1] = (shifted[NUM_DIGITS_GF2X_ELEMENT-1] << low_shift_amt);
   shifted[NUM_DIGITS_GF2X_ELEMENT-1] |=(zeroshift_mask & carryover);
   shifted[0] &= SLACK_CLEAR_MASK;
}
////////////////////////////////////////////////////////////////////////////////
void lift_mul_dense_to_sparse_CT(SLICE_BUNDLE bs_res[], CONST DIGIT dense[], CONST POS sparse[], unsigned int nPos) {
   SLICE tmp[SLICES_OF_P];
   for (int i = 0; i < nPos; i++) {
      gf2x_mod_mul_monom((DIGIT *)tmp, sparse[i], dense);
      for (int j = 0; j < SLICES_OF_P; j++) {
         bs_res[j] = bs_increment(bs_res[j], tmp[j]);
      }
   }
}
////////////////////////////////////////////////////////////////////////////////
static INLINE void bs_compute_upcs(
   OUT SLICE_BUNDLE bs_upc[N0 * SLICES_OF_P],
   IN  POS H_sparse[N0][PAD32(V)],
   IN  DIGIT syndrome[NUM_DIGITS_GF2X_ELEMENT])
{
   memset(bs_upc, 0, sizeof(SLICE_BUNDLE) * N0 * SLICES_OF_P);
   for (int i = 0; i < N0; i++) {
      lift_mul_dense_to_sparse_CT(
          bs_upc + (i * SLICES_OF_P),
          syndrome,
          H_sparse[i],
          V);
   }
}
////////////////////////////////////////////////////////////////////////////////
static INLINE int bs_argmax(SLICE_BUNDLE bs_upc[N0 * SLICES_OF_P])
{

   /* set candidate to all ones */
   ALIGNED __m256i candidate[N0 * SLICES_OF_P];
   for (int z = 0; z < N0 * SLICES_OF_P; z++)
      candidate[z] = _mm256_cmpeq_epi32(candidate[z], candidate[z]);

   /* phase 1: scan MSB→LSB with early exit */
   int active_blocks = N0 * SLICES_OF_P;
   for (int i = BITS_FOR_V - 1; i >= 0; i--) {
      // move this outside the loop
      ALIGNED __m256i new_cand[N0 * SLICES_OF_P];
      uint32_t any_set = 0;
      for (int z = 0; z < N0 * SLICES_OF_P; z++) {
         new_cand[z] = _mm256_and_si256(candidate[z], bs_upc[z].slice[i]);
         any_set |= !_mm256_testz_si256(new_cand[z], new_cand[z]);
      }
      if (any_set) {
         active_blocks = 0;
         for (int z = 0; z < N0 * SLICES_OF_P; z++) {
            candidate[z] = new_cand[z];
            active_blocks += !_mm256_testz_si256(candidate[z], candidate[z]);
         }
         if (active_blocks == 1)
            break; // early exit
      }
   }
   /* phase 2: find physical location of the argmax */
   int phys_pos = -1;
   for (int i = 0; i < N0 * SLICES_OF_P && phys_pos == -1; i++) {
      if (_mm256_testz_si256(candidate[i], candidate[i]))
         continue;
      uint64_t lanes[4];
      memcpy(lanes, &candidate[i], sizeof(lanes));
      for (int l = 0; l < 4 && phys_pos == -1; l++) {
         if (lanes[l] != 0) {
            int bit = __builtin_clzll(lanes[l]); // big-endian → clzll this is needed because the counter array are stored from the MSB to the LSB
            phys_pos = i * 256 + l * 64 + bit;
         }
      }
   }
   if (phys_pos == -1)
      return -1;
   /* phase 3: conversion from physical position to polynomio index */
   int circulant_block = phys_pos / (SLICES_OF_P * 256);
   int local_phys = phys_pos % (SLICES_OF_P * 256);
   int poly_idx = local_phys - SLACK_SIZE;
   int j = (P - 1) - poly_idx;
   return circulant_block * P + j;
}
////////////////////////////////////////////////////////////////////////////////
static INLINE int bs_update_syndrome_and_upcs(
   OUT SLICE_BUNDLE bs_upc[N0 * SLICES_OF_P],
   IN  CONST POS Htr_sparse[N0][PAD32(V)], 
   IN  DIGIT H_full_dense[N0][P][PAD64(NUM_DIGITS_GF2X_ELEMENT)],
   IN  POS flip, 
   OUT uint8_t syndrome_bits[P],
   IN  int hw)
{
   int flip_block = flip / P;
   int flip_bit = flip - flip_block * P;
   __m256i vp = _mm256_set1_epi32((uint32_t)P);
   __m256i vpos = _mm256_set1_epi32((uint32_t)flip_bit);
   /* update syndrome and upcs */
   for (int col_reg = 0; col_reg < N_REGS_H; col_reg++) {
      /* get the column of H corresponding to the flipped bit */
      uint32_t tmp[I32_IN_YMM] = {0};
      __m256i htr = _mm256_loadu_si256((__m256i *)&Htr_sparse[flip_block][col_reg * I32_IN_YMM]);
      __m256i sum = _mm256_add_epi32(htr, vpos);
      __m256i sub = _mm256_sub_epi32(sum, vp);
      __m256i res = _mm256_min_epu32(sum, sub);
      _mm256_storeu_si256((__m256i *)tmp, res);
      /* scan each idx in the column */
      for (int i = 0; (i < I32_IN_YMM) && (col_reg * I32_IN_YMM + i < V); i++) {
         /* update the syndrome */
         POS row = tmp[i];
         syndrome_bits[row] ^= 1;
         int delta = (syndrome_bits[row] == 0) ? -1 : 1;
         hw += delta;
         /* update upcs */
         for (int block = 0; block < N0; block++) {
            for (int j = 0; j < SLICES_OF_P; j++) {
               __m256i H_vec = _mm256_loadu_si256((__m256i *)&H_full_dense[block][row][j * 4]);
               if(delta == 1) {
                  bs_upc[block * SLICES_OF_P + j] = bs_increment(bs_upc[block * SLICES_OF_P + j], H_vec);
               } else {
                  bs_upc[block * SLICES_OF_P + j] = bs_decrement(bs_upc[block * SLICES_OF_P + j], H_vec);
               }
            }
         }
      }
   }
   return hw;
}
////////////////////////////////////////////////////////////////////////////////
int bfmax_decoder_full_dense(
   OUT DIGIT error[N0*NUM_DIGITS_GF2X_ELEMENT], 
   IN  POS Htr_sparse[N0][PAD32(V)], 
   IN  POS H_sparse[N0][PAD32(V)], 
   IN  DIGIT H_full_dense[N0][P][PAD64(NUM_DIGITS_GF2X_ELEMENT)],
   IN  DIGIT syndrome[NUM_DIGITS_GF2X_ELEMENT])
{
   /* expand each syndome bit to u8 */
   uint8_t syndrome_bits[P];
   dense_to_u8(syndrome_bits, syndrome, P);

   /* compute unsatisfied parity checks */
   SLICE_BUNDLE bs_upc[N0 * SLICES_OF_P];
   bs_compute_upcs(bs_upc, H_sparse, syndrome);

   /* decoding iterations */
   int iter = 0;
   int hw = population_count(syndrome);
   do {
      POS col = bs_argmax(bs_upc);
      int col_block = col / P;
      int col_bit = col % P;
      gf2x_toggle_coeff(error + col_block * NUM_DIGITS_GF2X_ELEMENT, col_bit);
      hw = bs_update_syndrome_and_upcs(bs_upc, Htr_sparse, H_full_dense, col, syndrome_bits, hw);
      DEBUG_PRINT("i: %d \t hw(s): %d \n", iter, hw);
      iter++;
   } while ((iter < 1.5 * NUM_ERRORS_T) && (hw != 0));
   return 1;
}
////////////////////////////////////////////////////////////////////////////////