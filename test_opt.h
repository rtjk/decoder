#pragma once
#include "helpers.h"
#include "parameters.h"
#include "test_utils.h"
#include "test_bfmax.h"
////////////////////////////////////////////////////////////////////////////////
#define TH0 43
#define TH1 38
#define TH2 36
////////////////////////////////////////////////////////////////////////////////
#define OPT_BITSLICED_OPERAND_WIDTH (BITS_TO_REPRESENT(V)+1)
////////////////////////////////////////////////////////////////////////////////
#define BITSLICED_OPERAND_NUM_OF_SLICES (16)
////////////////////////////////////////////////////////////////////////////////
#define OPT_SHIFT_ROTATE_RIGHT OPT_shift_rotate_right_VT
#define OPT_SHIFT_ROTATE_RIGHT_XOR OPT_shift_rotate_right_xor_VT
////////////////////////////////////////////////////////////////////////////////
#define BITSLICED_OPERAND_WIDTH (BITS_TO_REPRESENT(V))
#define SLICE_TYPE __m256i
#define NUM_BITS_IN_BITSLICED_OP (256)
// TODO: reduce
#define NUM_SLICES_GF2X_ELEMENT ( (NUM_DIGITS_GF2X_ELEMENT+3)/ (NUM_BITS_IN_BITSLICED_OP/DIGIT_SIZE_b) )
////////////////////////////////////////////////////////////////////////////////
#define PADDED_BLOCK_DIGITS (NUM_SLICES_GF2X_ELEMENT * (NUM_BITS_IN_BITSLICED_OP / DIGIT_SIZE_b))
////////////////////////////////////////////////////////////////////////////////
typedef struct {
   SLICE_TYPE slice[BITSLICED_OPERAND_NUM_OF_SLICES];
} OPT_bs_operand_t;
////////////////////////////////////////////////////////////////////////////////
static INLINE void OPT_gf2x_copy(DIGIT dest[], CONST DIGIT in[])
{
    memcpy(dest, in, NUM_DIGITS_GF2X_ELEMENT * DIGIT_SIZE_B);
}
////////////////////////////////////////////////////////////////////////////////
void OPT_shift_rotate_right_VT(DIGIT Res[],
               CONST unsigned int shiftAmt,
               DIGIT operand[])
{
   unsigned int word_shift_amt = shiftAmt>>6;
   DIGIT * source = &operand[word_shift_amt];

   POS low_shift_amt = shiftAmt & (((POS)1 << LO_SHIFT_AMT_BITS)-1);

   DIGIT zeroshift_mask = (DIGIT)0 - (!!(low_shift_amt));
   DIGIT carryover = (source[NUM_DIGITS_GF2X_ELEMENT-1] << SLACK_SIZE) |
                     (source[NUM_DIGITS_GF2X_ELEMENT-2] >> (DIGIT_SIZE_b - SLACK_SIZE));
   /* and extract the bits which should be inserted back in the least signif. pos.*/
   carryover = carryover >> (DIGIT_SIZE_b - low_shift_amt);

   for (int j = 0; j < NUM_DIGITS_GF2X_ELEMENT; j++) {
         Res[j] = (source[j] >> low_shift_amt) |
                     ( zeroshift_mask &
                      (source[j+1]) << (DIGIT_SIZE_b - low_shift_amt) );
   }
   Res[NUM_DIGITS_GF2X_ELEMENT-1] &= SLACK_CLEAR_MASK;

}
////////////////////////////////////////////////////////////////////////////////
void OPT_shift_rotate_right_xor_VT(DIGIT Res[],
               CONST unsigned int shiftAmt,
               DIGIT operand[])
{
   unsigned int word_shift_amt = shiftAmt>>6;
   DIGIT * source = &operand[word_shift_amt];

   POS low_shift_amt = shiftAmt & (((POS)1 << LO_SHIFT_AMT_BITS)-1);

   DIGIT zeroshift_mask = (DIGIT)0 - (!!(low_shift_amt));
   DIGIT carryover = (source[NUM_DIGITS_GF2X_ELEMENT-1] << SLACK_SIZE) |
                     (source[NUM_DIGITS_GF2X_ELEMENT-2] >> (DIGIT_SIZE_b - SLACK_SIZE));
   /* and extract the bits which should be inserted back in the least signif. pos.*/
   carryover = carryover >> (DIGIT_SIZE_b - low_shift_amt);

   for (int j = 0; j < NUM_DIGITS_GF2X_ELEMENT; j++) {
         Res[j] ^= (source[j] >> low_shift_amt) |
                     ( zeroshift_mask &
                      (source[j+1]) << (DIGIT_SIZE_b - low_shift_amt) );
   }
   Res[NUM_DIGITS_GF2X_ELEMENT-1] &= SLACK_CLEAR_MASK;

}
////////////////////////////////////////////////////////////////////////////////
static INLINE OPT_bs_operand_t OPT_slice_constant(int32_t constant)
{
   OPT_bs_operand_t result;
   __m256i one = _mm256_set_epi64x(0xFFFFFFFFFFFFFFFF,
                                   0xFFFFFFFFFFFFFFFF,
                                   0xFFFFFFFFFFFFFFFF,
                                   0xFFFFFFFFFFFFFFFF);
   __m256i zero = _mm256_set_epi64x(0,0,0,0);
   for(int bit = 0; bit < OPT_BITSLICED_OPERAND_WIDTH; bit++) {
      if ((constant & 1) == 1) {
         result.slice[bit] = one;
      } else {
         result.slice[bit] = zero;
      }
      constant >>=1;
   }
   return result;
}
////////////////////////////////////////////////////////////////////////////////
static INLINE SLICE_TYPE OPT_bitslice_full_adder_carry(SLICE_TYPE  addend_a,
                         SLICE_TYPE  addend_b,
                         SLICE_TYPE  carry_in)
{
   SLICE_TYPE tmp = addend_a ^ addend_b;
   return (tmp & carry_in) | (addend_a & addend_b);
}
////////////////////////////////////////////////////////////////////////////////
static INLINE SLICE_TYPE OPT_bitslice_full_adder_result(SLICE_TYPE  addend_a,
                         SLICE_TYPE  addend_b,
                         SLICE_TYPE  carry_in)
{
   return addend_a ^ addend_b ^ carry_in;
}
////////////////////////////////////////////////////////////////////////////////
static INLINE SLICE_TYPE OPT_bitslice_half_adder_carry(SLICE_TYPE  addend_a,
                         SLICE_TYPE  addend_b)
{
   return addend_a & addend_b;
}
////////////////////////////////////////////////////////////////////////////////
static INLINE SLICE_TYPE OPT_bitslice_full_sum_positive_bit(OPT_bs_operand_t a, OPT_bs_operand_t b)
{

   a.slice[OPT_BITSLICED_OPERAND_WIDTH-1] = _mm256_set_epi64x(0,0,0,0);

   SLICE_TYPE carry = OPT_bitslice_half_adder_carry(a.slice[0],b.slice[0]);

   for(int i = 1; i<OPT_BITSLICED_OPERAND_WIDTH-1; i++) {
      carry = OPT_bitslice_full_adder_carry(a.slice[i],b.slice[i],carry);
   }

   carry = OPT_bitslice_full_adder_result(a.slice[OPT_BITSLICED_OPERAND_WIDTH-1],b.slice[OPT_BITSLICED_OPERAND_WIDTH-1],carry);
   
   return ~carry;
}
////////////////////////////////////////////////////////////////////////////////
static INLINE void OPT_bitslice_half_adder(SLICE_TYPE  addend_a,
                         SLICE_TYPE  addend_b,
                         SLICE_TYPE *result,
                         SLICE_TYPE *carry_out)
{
   *result    = addend_a ^ addend_b;
   *carry_out = addend_a & addend_b;
   return;
}
////////////////////////////////////////////////////////////////////////////////
static INLINE void OPT_bitslice_full_adder(SLICE_TYPE  addend_a,
                         SLICE_TYPE  addend_b,
                         SLICE_TYPE  carry_in,
                         SLICE_TYPE *result,
                         SLICE_TYPE *carry_out)
{
   SLICE_TYPE tmp = addend_a ^ addend_b;
   *result    = tmp ^ carry_in;
   *carry_out = (tmp & carry_in) | (addend_a & addend_b);
   return;
}
////////////////////////////////////////////////////////////////////////////////
static INLINE void OPT_bitslice_sum_in_place(OPT_bs_operand_t * a, SLICE_TYPE b, int dim_1, int dim_2, int offset)
{
   SLICE_TYPE carry = b;

   for(int i=0; i<dim_2; i++){
      OPT_bitslice_full_adder(a->slice[i+offset],
                           a->slice[i+offset+dim_1],
                           carry,
                           &(a->slice[i+offset]),
                           &carry);
   }
   for(int i=dim_2; i<dim_1; i++){
      OPT_bitslice_half_adder(a->slice[i+offset],
                          carry,
                          &(a->slice[i+offset]),
                          &carry);
   }

   if(offset+dim_1<BITSLICED_OPERAND_NUM_OF_SLICES) a->slice[offset+dim_1] = carry;
}
////////////////////////////////////////////////////////////////////////////////
/* python3 generate_all_bitsliced_parms.py list_of_all_vs.txt */
void OPT_update_error_vector_block(SLICE_TYPE err[], DIGIT orig_syndrome[], CONST POS HPosOnes[], OPT_bs_operand_t neg_threshold_bs)
{
   alignas(32) OPT_bs_operand_t upc[NUM_SLICES_GF2X_ELEMENT];
   alignas(32) SLICE_TYPE shifted[NUM_SLICES_GF2X_ELEMENT];
   alignas(32) DIGIT syndrome[3 * NUM_DIGITS_GF2X_ELEMENT];

   OPT_gf2x_copy(syndrome, orig_syndrome);
   DIGIT d_mask = ((DIGIT)1 << (P % DIGIT_SIZE_b)) - 1;
   int left_shift_amt = (P % DIGIT_SIZE_b);
   int right_shift_amt = DIGIT_SIZE_b - left_shift_amt;
   syndrome[NUM_DIGITS_GF2X_ELEMENT - 1] = (syndrome[0] << (P % DIGIT_SIZE_b)) | (syndrome[NUM_DIGITS_GF2X_ELEMENT - 1] & d_mask);
   for (int i = 0; i < 2 * NUM_DIGITS_GF2X_ELEMENT; i++) {
      syndrome[NUM_DIGITS_GF2X_ELEMENT + i] = (syndrome[i] >> right_shift_amt) | (syndrome[i + 1] << left_shift_amt);
   }

   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[0], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 0);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[1], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 1);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[2], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 1, 1, 0);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[3], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 2);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[4], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 3);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[5], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 1, 1, 2);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[6], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 2, 2, 0);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[7], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 3);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[8], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 4);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[9], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 1, 1, 3);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[10], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 5);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[11], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 6);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[12], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 1, 1, 5);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[13], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 2, 2, 3);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[14], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 3, 3, 0);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[15], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 4);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[16], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 5);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[17], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 1, 1, 4);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[18], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 6);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[19], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 7);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[20], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 1, 1, 6);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[21], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 2, 2, 4);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[22], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 7);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[23], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 8);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[24], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 1, 1, 7);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[25], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 9);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[26], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 10);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[27], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 1, 1, 9);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[28], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 2, 2, 7);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[29], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 3, 3, 4);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[30], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 4, 4, 0);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[31], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 5);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[32], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 6);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[33], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 1, 1, 5);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[34], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 7);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[35], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 8);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[36], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 1, 1, 7);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[37], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 2, 2, 5);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[38], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 8);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[39], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 9);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[40], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 1, 1, 8);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[41], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 10);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[42], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 11);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[43], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 1, 1, 10);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[44], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 2, 2, 8);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[45], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 3, 3, 5);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[46], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 9);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[47], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 10);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[48], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 1, 1, 9);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[49], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 11);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[50], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 12);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[51], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 1, 1, 11);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[52], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 2, 2, 9);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[53], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 12);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[54], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 13);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[55], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 1, 1, 12);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[56], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 14);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[57], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 15);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[58], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 1, 1, 14);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[59], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 2, 2, 12);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[60], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 3, 3, 9);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[61], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 4, 4, 5);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[62], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 5, 5, 0);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[63], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 6);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[64], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 7);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[65], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 1, 1, 6);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[66], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 8);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[67], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 0, 0, 9);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[68], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 1, 1, 8);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[69], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 2, 2, 6);
   OPT_SHIFT_ROTATE_RIGHT((DIGIT *) shifted, HPosOnes[70], syndrome);
   for(int i=0; i<NUM_SLICES_GF2X_ELEMENT; i++) OPT_bitslice_sum_in_place(&upc[i], shifted[i], 6, 3, 0);

   for (int i = 0; i < NUM_SLICES_GF2X_ELEMENT; i++) {
      // UPC >= threshold
      SLICE_TYPE delta = OPT_bitslice_full_sum_positive_bit(upc[i], neg_threshold_bs);
      err[i] ^= delta;
   }

}
////////////////////////////////////////////////////////////////////////////////
void OPT_gf2x_mod_fmac_dense_to_sparse(
   DIGIT Res[],
   CONST DIGIT dense[],
   CONST POS sparse[],
   unsigned int nPos)
{
   alignas(32) DIGIT operand[3 * NUM_DIGITS_GF2X_ELEMENT];
   OPT_gf2x_copy(operand, dense);
   DIGIT d_mask = ((DIGIT)1 << (P % DIGIT_SIZE_b)) - 1;
   int left_shift_amt = (P % DIGIT_SIZE_b);
   int right_shift_amt = DIGIT_SIZE_b - left_shift_amt;
   operand[NUM_DIGITS_GF2X_ELEMENT - 1] = (operand[0] << (P % DIGIT_SIZE_b)) | (operand[NUM_DIGITS_GF2X_ELEMENT - 1] & d_mask);
   for (int i = 0; i < 2 * NUM_DIGITS_GF2X_ELEMENT; i++) {
      operand[NUM_DIGITS_GF2X_ELEMENT + i] = (operand[i] >> right_shift_amt) | (operand[i + 1] << left_shift_amt);
   }
   for (unsigned int i = 0; i < nPos; i++) {
      if (sparse[i] != INVALID_POS_VALUE) {
         POS amount_right = (P - sparse[i]) & ((POS) - (!!(sparse[i])));
         OPT_SHIFT_ROTATE_RIGHT_XOR(Res, amount_right, operand);
      }
   }
   Res[NUM_DIGITS_GF2X_ELEMENT - 1] &= SLACK_CLEAR_MASK;
}
////////////////////////////////////////////////////////////////////////////////
void OPT_gf2x_mod_mul_dense_to_sparse(DIGIT Res[],
                                      CONST DIGIT dense[],
                                      CONST POS sparse[],
                                      unsigned int nPos)
{
    memset(Res, 0, NUM_DIGITS_GF2X_ELEMENT * DIGIT_SIZE_B);
    OPT_gf2x_mod_fmac_dense_to_sparse(Res, dense, sparse, nPos);
}
////////////////////////////////////////////////////////////////////////////////
int OPT_bf_decoder(
    OUT DIGIT error[N0 * NUM_DIGITS_GF2X_ELEMENT],
    IN POS H_sparse[N0][V],
    IO DIGIT syndrome[NUM_DIGITS_GF2X_ELEMENT])
{
   DEBUG_PRINT("i: F \t hw(s): %d \n", population_count(syndrome));

   int thresholds[ITER_MAX_OOP];
   thresholds[0] = TH0;
   thresholds[1] = TH1;
   thresholds[2] = TH2;
   for (int i = 3; i < ITER_MAX_OOP; i++) {
      thresholds[i] = (V + 1) / 2;
   }

   // ! pad each error block to 256 bits
   alignas(32) DIGIT estimate[N0 * NUM_SLICES_GF2X_ELEMENT * (NUM_BITS_IN_BITSLICED_OP / DIGIT_SIZE_b)] = {0};
   alignas(32) DIGIT currSyndrome[NUM_DIGITS_GF2X_ELEMENT];
   int syn_weight = P;
   OPT_gf2x_copy(currSyndrome, syndrome);

   for (int iteration = 0; iteration < ITER_MAX_OOP; iteration++) {

      /* Fixed threshold per iteration */
      OPT_bs_operand_t sliced_threshold;
      sliced_threshold = OPT_slice_constant((uint32_t)(-thresholds[iteration]));

      // Update error estimate
      for (int i = 0; i < N0; i++) {
         OPT_update_error_vector_block((SLICE_TYPE *)(estimate + i * NUM_SLICES_GF2X_ELEMENT * (NUM_BITS_IN_BITSLICED_OP / DIGIT_SIZE_b)), currSyndrome, H_sparse[i], sliced_threshold);
         estimate[i * NUM_SLICES_GF2X_ELEMENT * (NUM_BITS_IN_BITSLICED_OP / DIGIT_SIZE_b) + NUM_DIGITS_GF2X_ELEMENT - 1] &= SLACK_CLEAR_MASK;
      }
      if (iteration > 0)
         OPT_gf2x_copy(currSyndrome, syndrome);
      for (int i = 0; i < N0; i++) {
         OPT_gf2x_mod_fmac_dense_to_sparse(currSyndrome, estimate + i * NUM_SLICES_GF2X_ELEMENT * (NUM_BITS_IN_BITSLICED_OP / DIGIT_SIZE_b), H_sparse[i], V);
      }
      currSyndrome[NUM_DIGITS_GF2X_ELEMENT - 1] &= SLACK_CLEAR_MASK;

      // Check if the Hamming weight of the syndrome is 0
      syn_weight = population_count(currSyndrome);
      DEBUG_PRINT("i: %d \t hw(s): %d \t hw(e): %d\n", iteration, syn_weight, population_count(estimate) + population_count(estimate + NUM_SLICES_GF2X_ELEMENT * (NUM_BITS_IN_BITSLICED_OP / DIGIT_SIZE_b)));
      if (syn_weight == 0)
         break;
   }

   // Check if the Hamming weight of the found solution matches NUM_ERRORS_T
   int weight = 0;
   for (int i = 0; i < N0; i++) {
      for (int j = 0; j < NUM_DIGITS_GF2X_ELEMENT; j++) {
         error[(i * NUM_DIGITS_GF2X_ELEMENT) + j] = estimate[(i * NUM_SLICES_GF2X_ELEMENT * (NUM_BITS_IN_BITSLICED_OP / DIGIT_SIZE_b)) + j];
      }
      error[i * NUM_DIGITS_GF2X_ELEMENT + NUM_DIGITS_GF2X_ELEMENT - 1] &= SLACK_CLEAR_MASK;
      weight += population_count(&error[i * NUM_DIGITS_GF2X_ELEMENT]);
   }

   return (weight == NUM_ERRORS_T) && (syn_weight == 0);
}
////////////////////////////////////////////////////////////////////////////////
int OPT_bf_decoder_post(
    IO DIGIT error[N0 * NUM_DIGITS_GF2X_ELEMENT],
    IN POS H_sparse[N0][V],
    IN DIGIT syndrome[NUM_DIGITS_GF2X_ELEMENT],
    IN int number_of_iterations)
{
   /* set per-iteration thresholds */
   int thresholds[number_of_iterations];
   for (int i = 0; i < number_of_iterations; i++) {
      thresholds[i] = (V + 1) / 2; // TODO: try TH0 or ((V + 1)/2)+1
   }
   /* pad each error block to 256 bits */
   alignas(32) DIGIT error_add[N0 * NUM_SLICES_GF2X_ELEMENT * (NUM_BITS_IN_BITSLICED_OP / DIGIT_SIZE_b)] = {0};
   /* recompute the syndrome for each iteration */
   alignas(32) DIGIT syndrome_iter[NUM_DIGITS_GF2X_ELEMENT];
   OPT_gf2x_copy(syndrome_iter, syndrome);
   /* decoding iterations */
   for (int iteration = 0; iteration < number_of_iterations; iteration++) {
      /* early exit if the syndrome is zero */
      if (population_count(syndrome_iter) == 0)
         break;
      /* update error */
      OPT_bs_operand_t sliced_threshold = OPT_slice_constant((uint32_t)(-thresholds[iteration]));
      for (int i = 0; i < N0; i++) {
         OPT_update_error_vector_block((SLICE_TYPE *)(error_add + i * NUM_SLICES_GF2X_ELEMENT * (NUM_BITS_IN_BITSLICED_OP / DIGIT_SIZE_b)), syndrome_iter, H_sparse[i], sliced_threshold);
         error_add[i * NUM_SLICES_GF2X_ELEMENT * (NUM_BITS_IN_BITSLICED_OP / DIGIT_SIZE_b) + NUM_DIGITS_GF2X_ELEMENT - 1] &= SLACK_CLEAR_MASK;
      }
      /* compute syndrome */
      if (iteration > 0)
         OPT_gf2x_copy(syndrome_iter, syndrome);
      for (int i = 0; i < N0; i++) {
         OPT_gf2x_mod_fmac_dense_to_sparse(syndrome_iter, error_add + i * NUM_SLICES_GF2X_ELEMENT * (NUM_BITS_IN_BITSLICED_OP / DIGIT_SIZE_b), H_sparse[i], V);
      }
      syndrome_iter[NUM_DIGITS_GF2X_ELEMENT - 1] &= SLACK_CLEAR_MASK;
   }
   /* xor input error with the one found in the iterations */
   for (int block = 0; block < N0; block++) {
      int padded_block_digits = block * NUM_SLICES_GF2X_ELEMENT * (NUM_BITS_IN_BITSLICED_OP / DIGIT_SIZE_b);
      for (int j = 0; j < NUM_DIGITS_GF2X_ELEMENT; j++) {
         error[block * NUM_DIGITS_GF2X_ELEMENT + j] ^= error_add[padded_block_digits + j];
      }
   }
   return 1;
}
////////////////////////////////////////////////////////////////////////////////
