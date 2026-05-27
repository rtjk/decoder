#pragma once
#include "helpers.h"
#include "parameters.h"
#include "test_utils.h"
////////////////////////////////////////////////////////////////////////////////
// from test_utils:
// - gf2x_mod_densify_VT
// - population_count
// - gf2x_toggle_coeff
////////////////////////////////////////////////////////////////////////////////
#define N_REGS ((V + 7) / 8)
////////////////////////////////////////////////////////////////////////////////
#define WORD_LEVEL_SHIFT word_level_shift_VT
#define SLACK_SIZE (DIGIT_SIZE_b-(P%DIGIT_SIZE_b))
#define SLACK_CLEAR_MASK ( ((DIGIT) 0 - 1) >> (DIGIT_SIZE_b-(P%DIGIT_SIZE_b)))
#define SLACK_EXTRACT(digit_to_extract)  (digit_to_extract >> (P%DIGIT_SIZE_b) )
#define LO_SHIFT_AMT_BITS (BITS_TO_REPRESENT(DIGIT_SIZE_b-1))
#define HI_SHIFT_AMT_BITS (BITS_TO_REPRESENT(P) - LO_SHIFT_AMT_BITS)
////////////////////////////////////////////////////////////////////////////////
// #define BITSLICED_OPERAND_WIDTH (BITS_TO_REPRESENT(V)+1)
#define BITSLICED_OPERAND_WIDTH (BITS_TO_REPRESENT(V))
#define SLICE_TYPE __m256i
#define NUM_BITS_IN_BITSLICED_OP (256)
// TODO: reduce
#define NUM_SLICES_GF2X_ELEMENT ( (NUM_DIGITS_GF2X_ELEMENT+3)/ (NUM_BITS_IN_BITSLICED_OP/DIGIT_SIZE_b) )
////////////////////////////////////////////////////////////////////////////////
typedef struct {
   SLICE_TYPE slice[BITSLICED_OPERAND_WIDTH];
} bs_operand_t;
////////////////////////////////////////////////////////////////////////////////
static INLINE
void bitslice_half_adder(SLICE_TYPE  addend_a,
                         SLICE_TYPE  addend_b,
                         SLICE_TYPE *result,
                         SLICE_TYPE *carry_out)
{
   // TODO:
   // *result = _mm256_xor_si256(...);
   // *carry_out = _mm256_and_si256(...);
   _mm256_storeu_si256 (result, _mm256_xor_si256(addend_a, addend_b));
   _mm256_storeu_si256 (carry_out, _mm256_and_si256(addend_a, addend_b));
   return;
}
////////////////////////////////////////////////////////////////////////////////
static INLINE
bs_operand_t bitslice_inc(bs_operand_t a, SLICE_TYPE b)
{
   bs_operand_t result;
   SLICE_TYPE carry;
   bitslice_half_adder(a.slice[0],b,&(result.slice[0]),&carry);
   for(int i = 1; i<BITSLICED_OPERAND_WIDTH; i++) {
      bitslice_half_adder(a.slice[i],
                          carry,
                          &(result.slice[i]),
                          &carry);
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
   DIGIT carryover = (shifted[0] << SLACK_SIZE) | (shifted[1] >>
                     (DIGIT_SIZE_b -SLACK_SIZE));
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
   shifted[NUM_DIGITS_GF2X_ELEMENT-1] = (shifted[NUM_DIGITS_GF2X_ELEMENT-1] <<
                                         low_shift_amt);
   shifted[NUM_DIGITS_GF2X_ELEMENT-1] |=(zeroshift_mask & carryover);
   shifted[0] &= SLACK_CLEAR_MASK;
}
////////////////////////////////////////////////////////////////////////////////
// bs_res   bs_unsatParityChecks + (block * NUM_SLICES_GF2X_ELEMENT)
// dense    privateSyndrome
// sparse   HPosOnes[block]
// nPos     V
void lift_mul_dense_to_sparse_CT(bs_operand_t bs_res[], CONST DIGIT dense[], CONST POS sparse[], unsigned int nPos){
   SLICE_TYPE tmp[NUM_SLICES_GF2X_ELEMENT];
   for(int i =0; i< nPos; i++) {
      gf2x_mod_mul_monom((DIGIT *)tmp,sparse[i],dense);     
      for(int j = 0 ; j < NUM_SLICES_GF2X_ELEMENT; j++) {
         bs_res[j] = bitslice_inc(bs_res[j], tmp[j]);
      }
   }
}
////////////////////////////////////////////////////////////////////////////////
POS argmax_uint8(CONST uint8_t *arr, size_t len) {
   size_t i = 0;
   __m256i max_vec = _mm256_setzero_si256();
   for (; i <= len - 32; i += 32) {
      __m256i v = _mm256_loadu_si256((__m256i *)&arr[i]);
      max_vec = _mm256_max_epu8(max_vec, v);
   }
   // horizontal reduction (256 → scalar)
   __m128i lo = _mm256_castsi256_si128(max_vec);
   __m128i hi = _mm256_extracti128_si256(max_vec, 1);
   __m128i m = _mm_max_epu8(lo, hi);
   m = _mm_max_epu8(m, _mm_srli_si128(m, 8));
   m = _mm_max_epu8(m, _mm_srli_si128(m, 4));
   m = _mm_max_epu8(m, _mm_srli_si128(m, 2));
   m = _mm_max_epu8(m, _mm_srli_si128(m, 1));
   uint8_t max_val = (uint8_t)_mm_extract_epi8(m, 0);
   // scalar մն remainder
   for (; i < len; i++)
      if (arr[i] > max_val)
         max_val = arr[i];
   __m256i vmax = _mm256_set1_epi8((char)max_val);
   i = 0;
   for (; i <= len - 32; i += 32) {
      __m256i v = _mm256_loadu_si256((__m256i *)&arr[i]);
      __m256i cmp = _mm256_cmpeq_epi8(v, vmax);
      int mask = _mm256_movemask_epi8(cmp);
      if (mask) {
         return (POS)(i + __builtin_ctz(mask));
      }
   }
   for (; i < len; i++)
      if (arr[i] == max_val)
         return (POS)i;
   return (POS)-1;
}
////////////////////////////////////////////////////////////////////////////////
// uint32_t argmax_uint8(uint8_t *arr, size_t len) {
//    uint32_t max_idx = 0;
//    uint8_t max_val = 0;
//    for (uint32_t i = 1; i < len; i++) {
//       if (arr[i] > max_val) {
//          max_val = arr[i];
//          max_idx = i;
//       }
//    }
//    return max_idx;
// }
////////////////////////////////////////////////////////////////////////////////
static INLINE int update_syndrome_and_upcs(
   OUT uint8_t *upc, 
   IN  CONST POS Htr_sparse[N0][PAD32+V], 
   IN  CONST POS H_sparse[N0][PAD32+V], 
   IN  POS flip, 
   OUT uint8_t* syndrome_bits,
   IN  int hw)
{
   int flip_block = flip / P;
   int flip_bit = flip - flip_block * P;
   __m256i vp = _mm256_set1_epi32((uint32_t)P);
   __m256i vpos = _mm256_set1_epi32((uint32_t)flip_bit);
   /* vectorize H_sparse */
   __m256i v_H_row[N0][N_REGS];
   for (int block = 0; block < N0; block++) {
      int r;
      for (r = 0; r < N_REGS; r++) {
         v_H_row[block][r] = _mm256_loadu_si256((__m256i *)&H_sparse[block][r * 8]);
      }
   }
   /* update syndrome and save upc positions to update */
   __m256i up_pos[V][N0][N_REGS];
   int up_sign[V];
   for (int col_reg = 0; col_reg < N_REGS; col_reg++) {
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
            for (int row_reg = 0; row_reg < N_REGS; row_reg++) {
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
         for (int row_reg = 0; row_reg < N_REGS; row_reg++) {
            for (int lane = 0; (lane < 8) && (row_reg * 8 + lane < V); lane++) {
               int up_idx = (((i * N0 + block) * N_REGS + row_reg) * 8) + lane;
               uint32_t col = up_pos_u32[up_idx];
               upc[block * P + col] += delta;
            }
         }
      }
   }
   return hw;
}
////////////////////////////////////////////////////////////////////////////////
static INLINE void gf2x_xor(DIGIT Res[], CONST DIGIT A[], CONST DIGIT B[]){
    unsigned i;
    for (i = 0; i < NUM_DIGITS_GF2X_ELEMENT/4; i++) {
        __m256i a = _mm256_lddqu_si256((__m256i *)A + i);
        __m256i b = _mm256_lddqu_si256((__m256i *)B + i);
        _mm256_storeu_si256((__m256i *)Res + i, _mm256_xor_si256(a, b));
    }
    for (i = i*4; i < NUM_DIGITS_GF2X_ELEMENT; i++) {
        Res[i] = A[i] ^ B[i];
    }
}
////////////////////////////////////////////////////////////////////////////////
void sliced_to_uint8(CONST bs_operand_t* bs, uint8_t* ctrs, int total_elements, int bitsliced_width) {
   memset(ctrs, 0, total_elements * sizeof(uint8_t));
   for (int i = 0; i < N0; i++) {
      // global offset 
      CONST bs_operand_t* bs_block = bs + i * NUM_SLICES_GF2X_ELEMENT;
      for (int j = 0; j < P; j++) {
         // inversione dell'ordine dato che il polinomio è rappresentato in big endian mentre il counter array considera le posizioni in little endian
         int poly_idx    = (P - 1) - j;
         // tocca aggiungere il padding di 61 bit che si trova all'inizio
         int adjusted    = poly_idx + SLACK_SIZE;
         int block       = adjusted / 256;
         int lane        = (adjusted / 64) % 4;
         int bit_in_lane = 63 - (adjusted % 64);

         uint64_t lanes[4];
         uint8_t val = 0;
         for (int k = 0; k < bitsliced_width; k++) {
            memcpy(lanes, &bs_block[block].slice[k], sizeof(lanes));
            uint64_t bit = (lanes[lane] >> bit_in_lane) & 1ULL;
            val += (uint8_t)(bit << k);
         }
         ctrs[i * P + j] = val;
      }
  }
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
   OUT uint8_t upc[N0 * P],
   IN  POS Htr_sparse[N0][PAD32+V],
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
int bfmax_decoder_1(
   OUT DIGIT error[N0*NUM_DIGITS_GF2X_ELEMENT], 
   IN  POS Htr_sparse[N0][PAD32+V], 
   IN  POS H_sparse[N0][PAD32+V], 
   IN  DIGIT syndrome[NUM_DIGITS_GF2X_ELEMENT])
{
   /* expand each syndome bit to u8 */
   uint8_t syndrome_bits[P];
   dense_to_u8(syndrome_bits, syndrome, P);
   /* compute unsatisfied parity checks */
   ALIGNED uint8_t upc[N0 * P] = {0};
   compute_upcs(upc, Htr_sparse, syndrome_bits);
   /* decoding iterations */
   int iter = 0;
   int hw = population_count(syndrome);
   do {
      POS flip = argmax_uint8(upc, N0 * P);
      int block = flip / P; // quale blocco di HTr
      int x = flip % P;     // di quanto ruotare dentro quel blocco
      gf2x_toggle_coeff(error + block * NUM_DIGITS_GF2X_ELEMENT, x);
      hw = update_syndrome_and_upcs(upc, Htr_sparse, H_sparse, flip, syndrome_bits, hw);
      DEBUG_PRINT("i: %d \t hw(s): %d \n", iter, hw);
      iter++;
   } while ((iter < 1.5 * NUM_ERRORS_T) && (hw != 0));
   return 1;
}
////////////////////////////////////////////////////////////////////////////////
int bfmax_decoder_2(
   OUT DIGIT error[N0*NUM_DIGITS_GF2X_ELEMENT],
   IN  POS Htr_sparse[N0][PAD32+V], 
   IN  POS H_sparse[N0][PAD32+V], 
   IN  DIGIT syndrome[NUM_DIGITS_GF2X_ELEMENT])
{
   /* expand each syndome bit to u8 */
   uint8_t syndrome_bits[P];
   dense_to_u8(syndrome_bits, syndrome, P);
   /* compute unsatisfied parity checks */
   ALIGNED uint8_t upc[N0 * P] = {0};
   compute_upcs(upc, Htr_sparse, syndrome_bits);
   /* decoding iterations */
   int hw = hamming_weight(syndrome_bits);
   for(int iter = 0; (iter < 200) && (hw > 0); iter++) {
      /* flip the error bit corresponding to the maximum upc */
      int col = argmax_uint8(upc, N0 * P);
      int col_block = col / P;
      int col_bit = col % P;
      gf2x_toggle_coeff(error + col_block * NUM_DIGITS_GF2X_ELEMENT, col_bit); // TODO: save positions and toggle later
      /* update the syndrome */
      for (int i = 0; i < V; i++) {
         int row = (Htr_sparse[col_block][i] + col_bit) % P;
         uint8_t old = syndrome_bits[row];
         syndrome_bits[row] ^= 1;
         int delta = syndrome_bits[row] - old; // +1 or -1
         hw += delta;
         /* propagate the update to upcs */
         for (int b = 0; b < N0; b++) {
            for (int k = 0; k < V; k++) {
               int pos = (H_sparse[b][k] + row) % P;
               upc[b * P + pos] += delta;
            }
         }
      }
      DEBUG_PRINT("i: %d \t hw(s): %d \n", iter, hw);
   }
   return 1;
}
////////////////////////////////////////////////////////////////////////////////
