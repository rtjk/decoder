#pragma once
#include "helpers.h"
#include "parameters.h"
#include "test_utils.h"
#include <stdint.h>
////////////////////////////////////////////////////////////////////////////////
#define N_REGS_H   (PAD32(V) / I32_IN_YMM)
#define N_REGS_UPC (PAD8(N0 * P) / I8_IN_YMM)
#define N_REGS_UPC512      (N_REGS_UPC / 2)
#define N_REGS_UPC_TAIL     (N_REGS_UPC % 2)

#define I32_IN_ZMM      (512/32)
#define N_REGS_H512     (N_REGS_H / 2)
#define N_REGS_H_TAIL   (N_REGS_H % 2)

////////////////////////////////////////////////////////////////////////////////
#define WORD_LEVEL_SHIFT word_level_shift_VT
#define SLACK_SIZE (DIGIT_SIZE_b-(P%DIGIT_SIZE_b))
#define SLACK_CLEAR_MASK ( ((DIGIT) 0 - 1) >> (DIGIT_SIZE_b-(P%DIGIT_SIZE_b)))
#define SLACK_EXTRACT(digit_to_extract)  (digit_to_extract >> (P%DIGIT_SIZE_b) )
#define LO_SHIFT_AMT_BITS (BITS_TO_REPRESENT(DIGIT_SIZE_b-1))
#define HI_SHIFT_AMT_BITS (BITS_TO_REPRESENT(P) - LO_SHIFT_AMT_BITS)
////////////////////////////////////////////////////////////////////////////////
static INLINE POS argmax_u8(uint8_t arr[PAD8(N0 * P)]) {
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

static INLINE POS argmax_u8_avx512(uint8_t arr[PAD8(N0 * P)]) {
   /* find max */

   __m512i max_vec512 = _mm512_setzero_si512();
    int i = 0;
    for (; i < N_REGS_UPC512; i++) {
        __m512i v = _mm512_loadu_si512((const void *)&arr[i * I8_IN_ZMM]);
        max_vec512 = _mm512_max_epu8(max_vec512, v);
    }
    __m256i max_vec = _mm256_max_epu8(
        _mm512_castsi512_si256(max_vec512),
        _mm512_extracti64x4_epi64(max_vec512, 1)
    );

#if N_REGS_UPC_TAIL
    /* AVX2 TAIL BEGIN: offset in byte = i * I8_IN_ZMM (fine dei blocchi da 64 byte),
     * NON i * I8_IN_YMM, perché qui i è ancora contato in unità ZMM */
    {
        __m256i v = _mm256_loadu_si256((__m256i *)&arr[i * I8_IN_ZMM]);
        max_vec = _mm256_max_epu8(max_vec, v);
    }
    /* AVX2 TAIL END */
#endif

    /* extract max: riduzione 256 -> 8 bit (invariata) */
    __m128i lo = _mm256_castsi256_si128(max_vec);
    __m128i hi = _mm256_extracti128_si256(max_vec, 1);
    __m128i m = _mm_max_epu8(lo, hi);
    m = _mm_max_epu8(m, _mm_srli_si128(m, 8));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 4));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 2));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 1));
    uint8_t max_val = (uint8_t)_mm_extract_epi8(m, 0);

    /* ---- find position of max (AVX-512, corpo principale) ---- */
    __m512i vmax512 = _mm512_set1_epi8((char)max_val);
    for (i = 0; i < N_REGS_UPC512; i++) {
        __m512i v = _mm512_loadu_si512((const void *)&arr[i * I8_IN_ZMM]);
        __mmask64 mask = _mm512_cmpeq_epi8_mask(v, vmax512);
        if (mask) {
            return (POS)(i * I8_IN_ZMM + __builtin_ctzll((unsigned long long)mask));
        }
    }

#if N_REGS_UPC_TAIL
    /* AVX2 TAIL BEGIN: stesso discorso, offset = i * I8_IN_ZMM */
    {
        __m256i vmax = _mm256_set1_epi8((uint8_t)max_val);
        __m256i v = _mm256_loadu_si256((__m256i *)&arr[i * I8_IN_ZMM]);
        __m256i cmp = _mm256_cmpeq_epi8(v, vmax);
        int mask = _mm256_movemask_epi8(cmp);
        if (mask) {
            return (POS)(i * I8_IN_ZMM + __builtin_ctz(mask));
        }
    }
    /* AVX2 TAIL END */
#endif

    return (POS)-1;
}


////////////////////////////////////////////////////////////////////////////////
static INLINE int update_syndrome_and_upcs(
   OUT uint8_t upc[PAD8(N0 * P)], 
   IN  POS Htr_sparse[N0][PAD32(V)], 
   IN  __m256i v_H_sparse[N0][N_REGS_H],
   IN  POS flip, 
   OUT uint8_t syndrome_bits[P],
   IN  int hw)
{
   int flip_block = flip / P;
   int flip_bit = flip - flip_block * P;
   __m256i vp = _mm256_set1_epi32((uint32_t)P);
   __m256i vpos = _mm256_set1_epi32((uint32_t)flip_bit);
   /* update syndrome and save upc positions to update */
   __m256i up_pos[V][N0][N_REGS_H];
   int up_sign[V];
   for (int col_reg = 0; col_reg < N_REGS_H; col_reg++) {
      /* get the column of H corresponding to the flipped bit */
      uint32_t tmp[I32_IN_YMM];
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
         //int delta = (syndrome_bits[row] == 0) ? -1 : 1;
         int delta = ((int)syndrome_bits[row] << 1) - 1;
         hw += delta;
         if (hw == 0) {
            return hw;
         }
         up_sign[col_reg * I32_IN_YMM + i] = delta;
         /* save upc positions to update (faster than updating upcs directly) */
         __m256i vrow = _mm256_set1_epi32((uint32_t)row);
         for (int block = 0; block < N0; block++) {
            for (int row_reg = 0; row_reg < N_REGS_H; row_reg++) {
               __m256i col = _mm256_add_epi32(v_H_sparse[block][row_reg], vrow);
               __m256i sub = _mm256_sub_epi32(col, vp);
               __m256i res = _mm256_min_epu32(col, sub);
               up_pos[col_reg * I32_IN_YMM + i][block][row_reg] = res;
            }
         }
      }
   }
   /* update upcs */
   uint32_t *RESTRICT up_pos_u32 = (uint32_t *)up_pos;
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
      u8[i] = gf2x_get_coeff(dense, i);
   }
}
////////////////////////////////////////////////////////////////////////////////
int bfmax_decoder(
   OUT DIGIT error[N0*NUM_DIGITS_GF2X_ELEMENT], 
   IN  POS Htr_sparse[N0][PAD32(V)], 
   IN  POS H_sparse[N0][PAD32(V)], 
   IN  uint8_t syndrome_bits[P],
   IN  int hw_start
)
{
   /* expand each syndome bit to u8 */
   //uint8_t syndrome_bits[P];
   //dense_to_u8(syndrome_bits, syndrome, P);
   /* vectorize H_sparse */
   __m256i v_H_sparse[N0][N_REGS_H];
   for (int block = 0; block < N0; block++) {
      for (int r = 0; r < N_REGS_H; r++) {
         v_H_sparse[block][r] = _mm256_loadu_si256((__m256i *)&H_sparse[block][r * 8]);
      }
   }
   /* compute unsatisfied parity checks */
   ALIGNED uint8_t upc[PAD8(N0 * P)] = {0};
   compute_upcs(upc, Htr_sparse, syndrome_bits);
   /* decoding iterations */
   int iter = 0;
   int hw = hw_start;
   do {
      POS col = argmax_u8_avx512(upc);
      int col_block = col / P;
      int col_bit = col % P;
      gf2x_toggle_coeff(error + col_block * NUM_DIGITS_GF2X_ELEMENT, col_bit);
      hw = update_syndrome_and_upcs(upc, Htr_sparse, v_H_sparse, col, syndrome_bits, hw);
      DEBUG_PRINT("i: %d \t hw(s): %d \n", iter, hw);
      iter++;
   } while ((iter < 1.5 * NUM_ERRORS_T) && (hw != 0));
   return 1;
}
////////////////////////////////////////////////////////////////////////////////

