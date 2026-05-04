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
#define NUM_SLICES_GF2X_ELEMENT ( (NUM_DIGITS_GF2X_ELEMENT+3)/ \
                                  (NUM_BITS_IN_BITSLICED_OP/DIGIT_SIZE_b) )
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
                         POSITION_T high_shift_amt,
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
                        POSITION_T shift_amt,
                        CONST DIGIT to_shift[])
{
   DIGIT mask;
   /*shift_amt is split bitwise :  |------------------shift_amt------------------|
    *                              | inter word shift amt | intra word shift amt |
    *                                  HI_SHIFT_AMT_BITS     LO_SHIFT_AMT_BITS
    */
   /* inter word shifting, done speculatively shifting the entire operand by a
    * power of two, and conditionally committing the result */
   POSITION_T high_shift_amt = shift_amt >> LO_SHIFT_AMT_BITS;
   POSITION_T low_shift_amt = shift_amt & (((POSITION_T)1 << LO_SHIFT_AMT_BITS)
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
void lift_mul_dense_to_sparse_CT(bs_operand_t bs_res[], CONST DIGIT dense[], CONST POSITION_T sparse[], unsigned int nPos){
   SLICE_TYPE tmp[NUM_SLICES_GF2X_ELEMENT];
   for(int i =0; i< nPos; i++) {
      gf2x_mod_mul_monom((DIGIT *)tmp,sparse[i],dense);     
      for(int j = 0 ; j < NUM_SLICES_GF2X_ELEMENT; j++) {
         bs_res[j] = bitslice_inc(bs_res[j], tmp[j]);
      }
   }
}
////////////////////////////////////////////////////////////////////////////////
POSITION_T argmax_uint8(CONST uint8_t* arr, size_t len) {
    size_t i = 0;
    __m256i max_vec = _mm256_setzero_si256();
    for (; i <= len - 32; i += 32) {
        __m256i v = _mm256_loadu_si256((__m256i*)&arr[i]);
        max_vec = _mm256_max_epu8(max_vec, v);
    }
    // horizontal reduction (256 → scalar)
    __m128i lo = _mm256_castsi256_si128(max_vec);
    __m128i hi = _mm256_extracti128_si256(max_vec, 1);
    __m128i m  = _mm_max_epu8(lo, hi);
    m = _mm_max_epu8(m, _mm_srli_si128(m, 8));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 4));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 2));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 1));
    uint8_t max_val = (uint8_t)_mm_extract_epi8(m, 0);
    // scalar մն remainder
    for (; i < len; i++)
        if (arr[i] > max_val) max_val = arr[i];
    __m256i vmax = _mm256_set1_epi8((char)max_val);
    i = 0;
    for (; i <= len - 32; i += 32) {
        __m256i v   = _mm256_loadu_si256((__m256i*)&arr[i]);
        __m256i cmp = _mm256_cmpeq_epi8(v, vmax);

        int mask = _mm256_movemask_epi8(cmp);
        if (mask) {
            return (POSITION_T)(i + __builtin_ctz(mask));
        }
    }
    for (; i < len; i++)
        if (arr[i] == max_val) return (POSITION_T)i;

    return (POSITION_T)-1;
}
////////////////////////////////////////////////////////////////////////////////
static INLINE void update_counters_uint8(uint8_t *sigma, CONST POSITION_T HtrPosOnes[N0][V], CONST POSITION_T  HPosOnes[N0][V], POSITION_T pos_flip, DIGIT* syndrome){
    int b = pos_flip >= P ? 1 : 0;
    POSITION_T local_pos = pos_flip - b * P;
    __m256i vp   = _mm256_set1_epi32((uint32_t)P);
    __m256i vpos = _mm256_set1_epi32((uint32_t)local_pos);
    // pre-carica HPosOnes[b2] nei registri AVX2 una volta sola
    __m256i h2_regs[N0][N_REGS];
    for (int b2 = 0; b2 < N0; b2++) {
        for (int r = 0; r < N_REGS; r++) {
            uint32_t tmp[8] = {0};
            for (int j = 0; j < 8 && r*8+j < V; j++)
                tmp[j] = HPosOnes[b2][r*8+j];
            h2_regs[b2][r] = _mm256_loadu_si256((__m256i *)tmp);
        }
    }
    // calcola row_indices, ds e aggiorna counter in un unico loop
    for (int r = 0; r < N_REGS; r++) {
        uint32_t tmp[8] = {0};
        for (int i = 0; i < 8 && r*8+i < V; i++)
            tmp[i] = HtrPosOnes[b][r*8+i];
        __m256i htr = _mm256_loadu_si256((__m256i *)tmp);
        __m256i sum = _mm256_add_epi32(htr, vpos);
        __m256i sub = _mm256_sub_epi32(sum, vp);
        __m256i res = _mm256_min_epu32(sum, sub);
        _mm256_storeu_si256((__m256i *)tmp, res);
        for (int i = 0; i < 8 && r*8+i < V; i++) {
            POSITION_T row_index = tmp[i];
            // ds branch-free
            int straightIdx = (NUM_DIGITS_GF2X_ELEMENT * DIGIT_SIZE_b - 1) - row_index;
            DIGIT bit       = (syndrome[straightIdx / DIGIT_SIZE_b] >>
                              (DIGIT_SIZE_b - 1 - straightIdx % DIGIT_SIZE_b)) & 1;
            int d           = (int)(2 * bit) - 1;
            __m256i vrow = _mm256_set1_epi32((uint32_t)row_index);
            for (int b2 = 0; b2 < N0; b2++) {
                POSITION_T offset = b2 * P;
                for (int r2 = 0; r2 < N_REGS; r2++) {
                    // col = (HPosOnes[b2][j] + row_index) % P
                    __m256i col = _mm256_add_epi32(h2_regs[b2][r2], vrow);
                    __m256i s   = _mm256_sub_epi32(col, vp);
                    col = _mm256_min_epu32(col, s);
                    uint32_t cols[8];
                    _mm256_storeu_si256((__m256i *)cols, col);
                    for (int j = 0; j < 8 && r2*8+j < V; j++)
                        sigma[offset + cols[j]] += d;
                }
            }
        }
    }
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
int bfmax_decoder(DIGIT out[], CONST POSITION_T HtrPosOnes[N0][V], CONST POSITION_T HPosOnes[N0][V], DIGIT privateSyndrome[]) {
   /* densify H^T */
   DIGIT HTr[N0][NUM_DIGITS_GF2X_ELEMENT] = {{0}};
   for (int i = 0; i < N0; i++) {
      gf2x_mod_densify_VT(HTr[i], HtrPosOnes[i], V);
   }
   /* compute bitsliced UPCs */
   DIGIT update[NUM_DIGITS_GF2X_ELEMENT];
   bs_operand_t bs_unsatParityChecks[N0 * NUM_SLICES_GF2X_ELEMENT];
   memset(bs_unsatParityChecks, 0, sizeof(bs_unsatParityChecks));
   for (int i = 0; i < N0; i++) {
      lift_mul_dense_to_sparse_CT(
         bs_unsatParityChecks + (i * NUM_SLICES_GF2X_ELEMENT),
         privateSyndrome,
         HPosOnes[i],
         V);
   }
   /* convert UPCs to uint8_t */
   uint8_t sigma[N0 * P] __attribute__((aligned(32)));
   memset(sigma, 0, N0 * P * sizeof(uint8_t));
   sliced_to_uint8(bs_unsatParityChecks, sigma, N0 * P, BITSLICED_OPERAND_WIDTH);
   // compute_counters_uint8(sigma, HtrPosOnes, privateSyndrome);

   /* decoding iterations */
   int iter = 0;
   int hw = population_count(privateSyndrome);
   do {
      memset(update, 0, NUM_DIGITS_GF2X_ELEMENT * DIGIT_SIZE_B);
      /* HYBRID APPROACH WITH COUNTER ARRAY FROM SLICED TO UINT8 */
      POSITION_T flip = argmax_uint8(sigma, N0 * P);
      int block = flip / P; // quale blocco di HTr
      int x = flip % P;     // di quanto ruotare dentro quel blocco
      gf2x_toggle_coeff(out + block * NUM_DIGITS_GF2X_ELEMENT, x);
      gf2x_mod_mul_monom(update, x == 0 ? 0 : x, HTr[block]);
      gf2x_xor(privateSyndrome, update, privateSyndrome);
      update_counters_uint8(sigma, HtrPosOnes, HPosOnes, flip, privateSyndrome);
      /* APPROACH WITH COUNTER ARRAY BITSLICED */
      // POSITION_T flip = argmax_bitsliced_impv(bs_unsatParityChecks, N0 * NUM_SLICES_GF2X_ELEMENT);
      // int block    = flip / P;  // quale blocco di HTr
      // int x        = flip % P;  // di quanto ruotare dentro quel blocco
      // gf2x_toggle_coeff(out + block * NUM_DIGITS_GF2X_ELEMENT, x);
      // update_counters_bitsliced(bs_unsatParityChecks, HtrPosOnes, HPosOnes, privateSyndrome, flip);
      hw = population_count(privateSyndrome);
      DEBUG_PRINT("i: %d \t hw(s): %d \n", iter, hw);
      iter++;
   } while ((iter < 1.5 * NUM_ERRORS_T) && (hw != 0));

   /* Check the solution of the decoder */
   int check = 0;
   while (check < NUM_DIGITS_GF2X_ELEMENT && privateSyndrome[check++] == 0)
      ;

   int hws = population_count(privateSyndrome);
   DEBUG_PRINT("i: L \t hw(s): %d \n", hws);
   if (hws != 0)
      ERROR("hw(s)=%d decoding failure", hws);

   // return (check == NUM_DIGITS_GF2X_ELEMENT);
   return 1;
}
////////////////////////////////////////////////////////////////////////////////