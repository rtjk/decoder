#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdalign.h>
#include <string.h>

#include "parameters.h"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
static INLINE
void update_counters_after_flip(uint8_t *sigma, CONST POSITION_T HtrPosOnes[N0][V], CONST POSITION_T  HPosOnes[N0][V], POSITION_T pos_flip, DIGIT* syndrome){
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
        __m256i msk = _mm256_cmpgt_epi32(vp, sum);
        __m256i res = _mm256_blendv_epi8(sub, sum, msk);
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
                    __m256i m   = _mm256_cmpgt_epi32(vp, col);
                    col = _mm256_blendv_epi8(s, col, m);

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
static INLINE
void gf2x_xor(DIGIT Res[], CONST DIGIT A[], CONST DIGIT B[])
{
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
static INLINE
void gf2x_toggle_coeff(DIGIT poly[], CONST unsigned int exponent)
{
   /* Reverse the index, this because the polynomial is saved in big endian */
   int straightIdx = (NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_b -1) - exponent;
   int digitIdx = straightIdx / DIGIT_SIZE_b;
   unsigned int inDigitIdx = straightIdx % DIGIT_SIZE_b;
   /* clear given coefficient */
   DIGIT mask = ( ((DIGIT) 1) << (DIGIT_SIZE_b-1-inDigitIdx));
   poly[digitIdx] = poly[digitIdx] ^ mask;
}
////////////////////////////////////////////////////////////////////////////////
// uint32_t argmax_avx512(CONST uint8_t* arr, size_t len)
// {
//     uint32_t max_idx = 0;
//     if (arr == NULL || len == 0)
//         return 0;
//     uint8_t max_val = arr[0];
//     for (size_t i = 1; i < len; i++)
//     {
//         if (arr[i] > max_val)
//         {
//             max_val = arr[i];
//             max_idx = (uint32_t)i;
//         }
//     }
//     return max_idx;
// }
POSITION_T argmax_avx512(CONST uint8_t* arr, size_t len) {
    /* ------------------------------------------------- */
    /*  Find Max (AVX2)                                  */
    /* ------------------------------------------------- */
    size_t i = 0;
    __m256i max_vec = _mm256_setzero_si256();

    for (; i <= len - 32; i += 32) {
        __m256i v = _mm256_loadu_si256((CONST __m256i*)&arr[i]);
        max_vec = _mm256_max_epu8(max_vec, v);
    }

    /* Horizontal reduction (256 → scalar) */
    __m128i lo = _mm256_castsi256_si128(max_vec);
    __m128i hi = _mm256_extracti128_si256(max_vec, 1);
    __m128i m  = _mm_max_epu8(lo, hi);
    m = _mm_max_epu8(m, _mm_srli_si128(m, 8));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 4));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 2));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 1));
    uint8_t max_val = (uint8_t)_mm_extract_epi8(m, 0);

    /* Scalar tail */
    for (; i < len; i++)
        if (arr[i] > max_val) max_val = arr[i];

    /* ------------------------------------------------- */
    /*  Find Argmax (AVX2)                               */
    /* ------------------------------------------------- */
    __m256i vmax = _mm256_set1_epi8((char)max_val);

    for (i = 0; i <= len - 32; i += 32) {
        __m256i v = _mm256_loadu_si256((CONST __m256i*)&arr[i]);
        __m256i cmp = _mm256_cmpeq_epi8(v, vmax);
        int mask = _mm256_movemask_epi8(cmp);

        if (mask) {
            return (POSITION_T)(i + __builtin_ctz(mask));
        }
    }

    /* Scalar tail */
    for (; i < len; i++)
        if (arr[i] == max_val) return (POSITION_T)i;

    return (POSITION_T)-1;
}
// POSITION_T argmax_avx512(CONST uint8_t* arr, size_t len) {
//     /* ------------------------------------------------- */
//     /*  Find Max                                         */
//     /* ------------------------------------------------- */
//     size_t i = 0;
//     __m512i max_vec = _mm512_setzero_si512();
//     for (; i <= len - 64; i += 64) {
//         __m512i v = _mm512_loadu_si512((__m512i*)&arr[i]);
//         max_vec = _mm512_max_epu8(max_vec, v);
//     }
//     // residui con AVX2
//     __m256i max256 = _mm256_max_epu8(
//                          _mm512_extracti64x4_epi64(max_vec, 0),
//                          _mm512_extracti64x4_epi64(max_vec, 1));
//     for (; i <= len - 32; i += 32) {
//         __m256i v = _mm256_loadu_si256((__m256i*)&arr[i]);
//         max256 = _mm256_max_epu8(max256, v);
//     }
//     // riduzione orizzontale AVX2
//     __m128i lo = _mm256_castsi256_si128(max256);
//     __m128i hi = _mm256_extracti128_si256(max256, 1);
//     __m128i m  = _mm_max_epu8(lo, hi);
//     m = _mm_max_epu8(m, _mm_srli_si128(m, 8));
//     m = _mm_max_epu8(m, _mm_srli_si128(m, 4));
//     m = _mm_max_epu8(m, _mm_srli_si128(m, 2));
//     m = _mm_max_epu8(m, _mm_srli_si128(m, 1));
//     uint8_t max_val = (uint8_t)_mm_extract_epi8(m, 0);
//     // residui scalari
//     for (; i < len; i++)
//         if (arr[i] > max_val) max_val = arr[i];
//     /* ------------------------------------------------- */
//     /*  Find Argmax                                      */
//     /* ------------------------------------------------- */
//     __m512i vmax = _mm512_set1_epi8((char)max_val);
//     for (i = 0; i <= len - 64; i += 64) {
//         __m512i v    = _mm512_loadu_si512((__m512i*)&arr[i]);
//         __mmask64 msk = _mm512_cmpeq_epi8_mask(v, vmax);
//         if (msk) return (POSITION_T)(i + __builtin_ctzll(msk));
//     }
//     for (; i <= len - 32; i += 32) {
//         __m256i v    = _mm256_loadu_si256((__m256i*)&arr[i]);
//         __m256i vmax256 = _mm256_set1_epi8((char)max_val);
//         __m256i cmp  = _mm256_cmpeq_epi8(v, vmax256);
//         int bits = _mm256_movemask_epi8(cmp);
//         if (bits) return (POSITION_T)(i + __builtin_ctz(bits));
//     }
//     for (; i < len; i++)
//         if (arr[i] == max_val) return (POSITION_T)i;
//     return (POSITION_T)-1;
// }
////////////////////////////////////////////////////////////////////////////////
void compute_counters_sliced(CONST bs_operand_t* bs, uint8_t* ctrs, int total_elements, int bitsliced_width) {
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
static INLINE
void word_level_shift_VT(DIGIT *RESTRICT shifted_param,
                         POSITION_T high_shift_amt,
                         DIGIT *RESTRICT to_shift)
{
   /* condit-pull whole digits towards the MSB, starting from the word_shift_amt - th
    * one, that is including the one which will have slack */
   for (int j = 0; j < NUM_DIGITS_GF2X_ELEMENT-high_shift_amt; j++) {
      shifted_param[j] = to_shift[j+high_shift_amt];
   }

   /* collect the slack carryover */
   DIGIT slack_carryover = SLACK_EXTRACT(shifted_param[0]);
   shifted_param[0] &= SLACK_CLEAR_MASK;

   /* move the remaining topmost word_shift_amt words (0th to word_shift_amt-1
    * one) taking care of tucking in the slack carryover */

   for (int j = high_shift_amt-1 ; j >=0 ; j--) {
      int target_idx = NUM_DIGITS_GF2X_ELEMENT-1-(high_shift_amt-1)+j;
      DIGIT to_write = slack_carryover;
      to_write = to_write | (to_shift[j] << SLACK_SIZE);
      slack_carryover = SLACK_EXTRACT(to_shift[j]);
      shifted_param[target_idx] = to_write;
   }
}
////////////////////////////////////////////////////////////////////////////////
static INLINE
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
   WORD_LEVEL_SHIFT(shifted,high_shift_amt,(DIGIT * restrict)to_shift);
   /* cyclic shifts inside DIGITs */
   /* extract low_shift_amt MSB for cyclic shift */
   DIGIT carryover = (shifted[0] << SLACK_SIZE) | (shifted[1] >>
                     (DIGIT_SIZE_b -SLACK_SIZE));
   carryover = carryover >> (DIGIT_SIZE_b - low_shift_amt);
   /* pure shift, carried over from left_bit_shift_n*/
   mask = ~(( (DIGIT)1 << (DIGIT_SIZE_b - low_shift_amt) ) - 1);
   /* must deal with C99 UB when shifting by variable size*/
   DIGIT zeroshift_mask = (DIGIT)0 - (!!(low_shift_amt));
// #if (defined HIGH_PERFORMANCE_X86_64)
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
// #else
//    for (int j = 0 ; j < NUM_DIGITS_GF2X_ELEMENT-1 ; j++) {
//       shifted[j] = (shifted[j] << low_shift_amt) | ( zeroshift_mask &
//                    (shifted[j+1] & mask) >> (DIGIT_SIZE_b - low_shift_amt) );
//    }
// #endif
   shifted[NUM_DIGITS_GF2X_ELEMENT-1] = (shifted[NUM_DIGITS_GF2X_ELEMENT-1] <<
                                         low_shift_amt);
   shifted[NUM_DIGITS_GF2X_ELEMENT-1] |=(zeroshift_mask & carryover);
   shifted[0] &= SLACK_CLEAR_MASK;
}
////////////////////////////////////////////////////////////////////////////////
static INLINE
void lift_mul_dense_to_sparse_CT(bs_operand_t bs_res[], CONST DIGIT dense[], CONST POSITION_T sparse[], unsigned int nPos){
   SLICE_TYPE tmp[NUM_SLICES_GF2X_ELEMENT];
   for(int i =0; i< nPos; i++) {
// #if (defined HIGH_PERFORMANCE_X86_64)
      /* note : last words of tmp will be intentionally garbage, in case
       * NUM_DIGITS_GF2X_ELEMENT is not divisible by 4, for alignment reasons
       * Their content won't be used */
      gf2x_mod_mul_monom((DIGIT *)tmp,sparse[i],dense);

// #else
//       gf2x_mod_mul_monom(tmp,sparse[i],dense);
// #endif
      for(int j = 0 ; j < NUM_SLICES_GF2X_ELEMENT; j++) {
         bs_res[j] = bitslice_inc(bs_res[j], tmp[j]);
      }
   }
}
////////////////////////////////////////////////////////////////////////////////
static INLINE
int population_count(DIGIT upc[])
{
   int ret = 0;
   for(int i = NUM_DIGITS_GF2X_ELEMENT - 1; i >= 0; i--) {
// #if defined(DIGIT_IS_ULLONG)
      ret += __builtin_popcountll((unsigned long long int) (upc[i]));
// #elif defined(DIGIT_IS_ULONG)
//       ret += __builtin_popcountl((unsigned long int) (upc[i]));
// #elif defined(DIGIT_IS_UINT)
//       ret += __builtin_popcount((unsigned int) (upc[i]));
// #elif defined(DIGIT_IS_UCHAR)
//       CONST unsigned char split_lookup[] = {
//          0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
//       };
//       ret += split_lookup[upc[i]&0xF] + split_lookup[upc[i]>>4];
// #else
// #error "Missing implementation for population_count(...)
// with this CPU word bitsize !!! "
// #endif
   }
   return ret;
} // end population_count
////////////////////////////////////////////////////////////////////////////////
static INLINE
void gf2x_set_coeff(DIGIT poly[], CONST unsigned int exponent, DIGIT value)
{
   int straightIdx = (NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_b -1) - exponent;
   int digitIdx = straightIdx / DIGIT_SIZE_b;

   unsigned int inDigitIdx = straightIdx % DIGIT_SIZE_b;

   /* clear given coefficient */
   DIGIT mask = ~( ((DIGIT) 1) << (DIGIT_SIZE_b-1-inDigitIdx));
   poly[digitIdx] = poly[digitIdx] & mask;
   poly[digitIdx] = poly[digitIdx] | (( value & ((DIGIT) 1)) <<
                                      (DIGIT_SIZE_b-1-inDigitIdx));
}
////////////////////////////////////////////////////////////////////////////////
static INLINE
void gf2x_mod_densify_VT(DIGIT dense[NUM_DIGITS_GF2X_ELEMENT],
                         CONST POSITION_T exponent[],
                         int num_exponents)
{
   for(int j=0; j<num_exponents; j++) {
      gf2x_set_coeff(dense, exponent[j], (DIGIT) 1);
   }
}
////////////////////////////////////////////////////////////////////////////////
int bf_decoding_CT(
    DIGIT out[], 
    CONST POSITION_T HtrPosOnes[N0][V], 
    CONST POSITION_T HPosOnes[N0][V], 
    DIGIT privateSyndrome[]
){
    DIGIT HTr[N0][NUM_DIGITS_GF2X_ELEMENT] = {{0}};
    for(int i=0; i<N0; i++) {
        gf2x_mod_densify_VT(HTr[i],HtrPosOnes[i],V);
    }
    // DIGIT H_dense[N0][NUM_DIGITS_GF2X_ELEMENT] = {{0}};
    // for(int i=0; i<N0; i++) {
    //     gf2x_mod_densify_VT(H_dense[i],HPosOnes[i],V);
    // }
    int iter = 0;
    int hw = population_count(privateSyndrome);
    //DIGIT update[NUM_DIGITS_GF2X_ELEMENT] = {0};
    bs_operand_t bs_unsatParityChecks[N0*NUM_SLICES_GF2X_ELEMENT];
    DIGIT update[NUM_DIGITS_GF2X_ELEMENT];
    memset(bs_unsatParityChecks, 0, sizeof(bs_unsatParityChecks));
    /* COMPUTE COUNTERS WITH BITSLICED STRUCTURE */
    for (int i = 0; i < N0; i++) {
         lift_mul_dense_to_sparse_CT(
            bs_unsatParityChecks+(i*NUM_SLICES_GF2X_ELEMENT),
            privateSyndrome,
            HPosOnes[i],
            V
        );
    }
    uint8_t sigma[N0*P] __attribute__((aligned(32)));
    memset(sigma, 0, N0*P*sizeof(uint8_t));
    /* CONVERSION OF THE COUNTERS */
    compute_counters_sliced(bs_unsatParityChecks, sigma, N0*P, BITSLICED_OPERAND_WIDTH);
    /*
    bs_operand_t product[N0*NUM_SLICES_GF2X_ELEMENT];
    for (int i = 0; i < N0; i++) {
        lift_mul_dense_to_sparse_CT(
            product+(i*NUM_SLICES_GF2X_ELEMENT),
            HTr[0],
            HPosOnes[i],
            V
        );
    }
    */
   do{
        memset(update, 0, NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_B);
        /* APPROACH WITH COUNTER ARRAY UINT8 */
        POSITION_T flip = argmax_avx512(sigma, N0*P);
        /* APPROACH WITH COUNTER ARRAY BITSLICED */
        //POSITION_T flip = argmax_bitsliced_impv(bs_unsatParityChecks, N0 * NUM_SLICES_GF2X_ELEMENT);
        /* FIND POSITION TO FLIP */
        int block    = flip / P;  // quale blocco di HTr
        int x        = flip % P;  // di quanto ruotare dentro quel blocco
        // the position is in little endian so it's ok because the conversion is done by the function
        gf2x_toggle_coeff(out + block * NUM_DIGITS_GF2X_ELEMENT, x);
        /* SCHOOLBOOK UPDATE OF THE SYNDROME */
        gf2x_mod_mul_monom(update, x == 0 ? 0 :  x, HTr[block]);
        gf2x_xor(privateSyndrome, update, privateSyndrome);
        // we can update the syndrome using the parallel shift position
        
        /* COUNTERS UPDATE  */
        /* APPROACH WITH COUNTER ARRAY UINT8 */
        update_counters_after_flip(sigma, HtrPosOnes, HPosOnes, flip, privateSyndrome);
        /* APPROACH WITH COUNTER ARRAY BITSLICED */
        //update_counters_bitsliced(bs_unsatParityChecks, HtrPosOnes, HPosOnes, privateSyndrome, flip);

        hw = population_count(privateSyndrome);

        iter++;
   } while( (iter < 1.5*NUM_ERRORS_T) && (hw != 0) );

   printf(">>>> hw: %d, iter: %d\n", hw, iter);


   /* Check the solution of the decoder */
   int check = 0;
   while (check < NUM_DIGITS_GF2X_ELEMENT && privateSyndrome[check++] == 0);
   //return (check == NUM_DIGITS_GF2X_ELEMENT);
   return 1;     
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
static inline void gf2x_add(CONST int nr, DIGIT Res[],
                            CONST int na, CONST DIGIT A[],
                            CONST int nb, CONST DIGIT B[])
{
   for (unsigned i = 0; i < nr; i++)
      Res[i] = A[i] ^ B[i];
}
////////////////////////////////////////////////////////////////////////////////
void right_bit_shift_n(CONST int length, DIGIT in[], CONST int amount)
{
   if ( amount == 0 ) return;
   int j;
   DIGIT mask;
   mask = ((DIGIT)0x01 << amount) - 1;
   for (j = length-1; j > 0 ; j--) {
      in[j] >>= amount;
      in[j] |=  (in[j-1] & mask) << (DIGIT_SIZE_b - amount);
   }
   in[j] >>= amount;
}
////////////////////////////////////////////////////////////////////////////////
void gf2x_mod(DIGIT out[],
              CONST int nin, CONST DIGIT in[])
{
   DIGIT aux[NUM_DIGITS_GF2X_ELEMENT+1];
   memcpy(aux, in, (NUM_DIGITS_GF2X_ELEMENT+1)*DIGIT_SIZE_B);
#if MSb_POSITION_IN_MSB_DIGIT_OF_MODULUS != 0
   right_bit_shift_n(NUM_DIGITS_GF2X_ELEMENT+1, aux,
                     MSb_POSITION_IN_MSB_DIGIT_OF_MODULUS);
#endif
   gf2x_add(NUM_DIGITS_GF2X_ELEMENT,out,
            NUM_DIGITS_GF2X_ELEMENT,aux+1,
            NUM_DIGITS_GF2X_ELEMENT,in+NUM_DIGITS_GF2X_ELEMENT);
#if MSb_POSITION_IN_MSB_DIGIT_OF_MODULUS != 0
   out[0] &=  ((DIGIT)1 << MSb_POSITION_IN_MSB_DIGIT_OF_MODULUS) - 1 ;
#endif

}
////////////////////////////////////////////////////////////////////////////////
void gf2x_fmac(DIGIT Res[],
               CONST DIGIT operand[],
               CONST unsigned int shiftAmt)
{
   unsigned int digitShift = shiftAmt / DIGIT_SIZE_b;
   unsigned int inDigitShift= shiftAmt % DIGIT_SIZE_b;
   DIGIT tmp,prevLo=0;
   int i;
   SIGNED_DIGIT inDigitShiftMask = ((SIGNED_DIGIT) (inDigitShift>0) << (DIGIT_SIZE_b-1)) >> (DIGIT_SIZE_b-1);
   for(i = NUM_DIGITS_GF2X_ELEMENT-1; i>=0 ; i--) {
      tmp = operand[i];
      Res[NUM_DIGITS_GF2X_ELEMENT+i-digitShift] ^= prevLo | (tmp << inDigitShift);
      prevLo = (tmp >> (DIGIT_SIZE_b - inDigitShift)) & inDigitShiftMask;
   }
   Res[NUM_DIGITS_GF2X_ELEMENT+i-digitShift] ^= prevLo;
}
////////////////////////////////////////////////////////////////////////////////
static inline void gf2x_mod_add(DIGIT Res[], CONST DIGIT A[], CONST DIGIT B[])
{
   gf2x_add(NUM_DIGITS_GF2X_ELEMENT, Res,
            NUM_DIGITS_GF2X_ELEMENT, A,
            NUM_DIGITS_GF2X_ELEMENT, B);
}
////////////////////////////////////////////////////////////////////////////////
void gf2x_mod_mul_dense_to_sparse(DIGIT Res[],
                                  CONST DIGIT dense[],
                                  CONST POSITION_T sparse[],
                                  unsigned int nPos)
{
   DIGIT resDouble[2*NUM_DIGITS_GF2X_ELEMENT] = {0};
   for (unsigned int i = 0; i < nPos; i++) {
      if (sparse[i] != INVALID_POS_VALUE) {
         gf2x_fmac(resDouble, dense,sparse[i]);
      }
   }
   gf2x_mod(Res, 2*NUM_DIGITS_GF2X_ELEMENT, resDouble);
}
////////////////////////////////////////////////////////////////////////////////
void densify_error(DIGIT dense[N0*NUM_DIGITS_GF2X_ELEMENT], POSITION_T sparse[NUM_ERRORS_T]) {
   memset(dense, 0x00, N0*NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_B);
   for (int j = 0; j < NUM_ERRORS_T; j++) {
      int polyIndex = (sparse[j] / P);
      int exponent = sparse[j] % P;
      gf2x_set_coeff( dense + NUM_DIGITS_GF2X_ELEMENT*polyIndex, exponent,
                      ( (DIGIT) 1));
   }
}
////////////////////////////////////////////////////////////////////////////////
// void compute_syndrome(DIGIT s[NUM_DIGITS_GF2X_ELEMENT], DIGIT *H_dense, POSITION_T e_sparse[NUM_ERRORS_T], DIGIT e_dense[N0*NUM_DIGITS_GF2X_ELEMENT]) {
//    // from encrypt_niederreiter()
//    int i;
//    DIGIT saux[NUM_DIGITS_GF2X_ELEMENT];
//    unsigned int filled;
//    memset(s, 0x00, NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_B);
//    POSITION_T blkErrorPos[NUM_ERRORS_T];
//    for (i = 0; i < N0-1; i++) {
//       filled=0;
//       for (int j = 0 ; j < NUM_ERRORS_T; j ++) {
//          if(e_sparse[j] / P == i) {
//             blkErrorPos[filled] =  e_sparse[j] % P;
//             filled++;
//          }
//       }
//       gf2x_mod_mul_dense_to_sparse(saux,
//                                    H_dense + i*NUM_DIGITS_GF2X_ELEMENT,
//                                    blkErrorPos,
//                                    filled);
//       gf2x_mod_add(s, s, saux);
//    }   // end for
//    gf2x_mod_add(s, s, e_dense+(N0-1)*NUM_DIGITS_GF2X_ELEMENT);
// }
// POSITION_T shift(POSITION_T h, POSITION_T i){
//     POSITION_T pos  = h + i;
//     POSITION_T mask = -(pos >= (POSITION_T)P);
//     return pos - ((POSITION_T)P & mask);
// }
// #define FLIP_BIT(arr, i) do { (arr)[(i) >> 6] ^=  (1ULL << ((i) & 63)); } while(0)
// #define S_WORDS ((P + 63) / 64) /* Number of words to represent the syndrome with an array of uint_64 */
// void compute_syndrome(DIGIT syndrome[], POSITION_T H[2][V], POSITION_T* error){
//     memset(syndrome,0, S_WORDS*DIGIT_SIZE_B);
//     // for each index inside error support
//     for (int i=0; i<NUM_ERRORS_T; i++) {
//         // set position
//         POSITION_T pos = error[i];
//         // determinate in with circulant we are
//         int b = (pos >= P);
//         POSITION_T local_pos = pos - (b * P);

//         for(int j = 0; j < V; j++){
//             POSITION_T pos_to_flip = shift(H[b][j], local_pos);
//             FLIP_BIT(syndrome, pos_to_flip);
//         }
//     }

// }
////////////////////////////////////////////////////////////////////////////////
void transposeHPosOnes(POSITION_T HtrPosOnes[N0][V], /* output*/
                       POSITION_T CONST HPosOnes[N0][V]
                      )
{
   for (int i = 0; i < N0; i++) {
      /* Obtain directly the sparse representation of the block of H */
      for (int k = 0; k < V; k++) {
         HtrPosOnes[i][k] = (P - HPosOnes[i][k])  % P; /* transposes indexes */
      }// end for k
   }
}