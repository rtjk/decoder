#pragma once
#include "helpers.h"
#include "parameters.h"
////////////////////////////////////////////////////////////////////////////////
static INLINE void gf2x_toggle_coeff(DIGIT poly[], CONST unsigned int exponent)
{
   int straightIdx = (NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_b -1) - exponent;
   int digitIdx = straightIdx / DIGIT_SIZE_b;
   unsigned int inDigitIdx = straightIdx % DIGIT_SIZE_b;

   /* clear given coefficient */
   DIGIT mask = ( ((DIGIT) 1) << (DIGIT_SIZE_b-1-inDigitIdx));
   poly[digitIdx] = poly[digitIdx] ^ mask;
}
////////////////////////////////////////////////////////////////////////////////
static INLINE int population_count(DIGIT upc[])
{
   int ret = 0;
   for(int i = NUM_DIGITS_GF2X_ELEMENT - 1; i >= 0; i--) {
      ret += __builtin_popcountll((unsigned long long int) (upc[i]));
   }
   return ret;
} // end population_count
////////////////////////////////////////////////////////////////////////////////
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
static INLINE void gf2x_mod_densify_VT(DIGIT dense[NUM_DIGITS_GF2X_ELEMENT],
                         CONST POSITION_T exponent[],
                         int num_exponents)
{
   for(int j=0; j<num_exponents; j++) {
      gf2x_set_coeff(dense, exponent[j], (DIGIT) 1);
   }
}
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
} // end transposeHPosOnes
////////////////////////////////////////////////////////////////////////////////
void util_densify_error(DIGIT dense[N0*NUM_DIGITS_GF2X_ELEMENT],
                        CONST POSITION_T positions[])
{
   for(int j=0; j<NUM_ERRORS_T; j++) {
      int block = positions[j] / P;
      gf2x_set_coeff(dense + block * NUM_DIGITS_GF2X_ELEMENT, positions[j] % P, 1);
   }
}
////////////////////////////////////////////////////////////////////////////////
void gf2x_add(CONST int nr, DIGIT Res[],
                            CONST int na, CONST DIGIT A[],
                            CONST int nb, CONST DIGIT B[])
{
   for (unsigned i = 0; i < nr; i++)
      Res[i] = A[i] ^ B[i];
} // end gf2x_add
////////////////////////////////////////////////////////////////////////////////
void gf2x_mod_add(DIGIT Res[], CONST DIGIT A[], CONST DIGIT B[])
{
   gf2x_add(NUM_DIGITS_GF2X_ELEMENT, Res,
            NUM_DIGITS_GF2X_ELEMENT, A,
            NUM_DIGITS_GF2X_ELEMENT, B);
} // end gf2x_mod_add
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
} // end right_bit_shift_n
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

} // end gf2x_mod
////////////////////////////////////////////////////////////////////////////////
void gf2x_fmac(DIGIT Res[],
               CONST DIGIT operand[],
               CONST unsigned int shiftAmt)
{
   unsigned int digitShift = shiftAmt / DIGIT_SIZE_b;
   unsigned int inDigitShift= shiftAmt % DIGIT_SIZE_b;
   DIGIT tmp,prevLo=0;
   int i;
   SIGNED_DIGIT inDigitShiftMask = ((SIGNED_DIGIT) (inDigitShift>0)  <<
                                    (DIGIT_SIZE_b-1)) >> (DIGIT_SIZE_b-1);
   for(i = NUM_DIGITS_GF2X_ELEMENT-1; i>=0 ; i--) {
      tmp = operand[i];
      Res[NUM_DIGITS_GF2X_ELEMENT+i-digitShift] ^= prevLo | (tmp << inDigitShift);
      prevLo = (tmp >> (DIGIT_SIZE_b - inDigitShift)) & inDigitShiftMask;
   }
   Res[NUM_DIGITS_GF2X_ELEMENT+i-digitShift] ^= prevLo;
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

} // end gf2x_mod_mul
////////////////////////////////////////////////////////////////////////////////
/* for r in rows
 *   for c in columns
 *     s[c] = s[c] xor (e[r] & H[r][c])
 */
void util_compute_syndrome(
   OUT DIGIT s_dense[NUM_DIGITS_GF2X_ELEMENT],
   IN  DIGIT Htr_dense[N0][NUM_DIGITS_GF2X_ELEMENT],
   IN  POSITION_T e_sparse[NUM_ERRORS_T])
{
   int i;
   DIGIT saux[NUM_DIGITS_GF2X_ELEMENT];
   unsigned int filled;
   memset(s_dense, 0x00, NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_B);
   POSITION_T blkErrorPos[NUM_ERRORS_T];
   for (i = 0; i < N0; i++) {
      filled=0;
      for (int j = 0 ; j < NUM_ERRORS_T; j++) {
         if(e_sparse[j] / P == i) {
            blkErrorPos[filled] =  e_sparse[j] % P;
            filled++;
         }
      }
      gf2x_mod_mul_dense_to_sparse(saux,
                                   Htr_dense[i],
                                   blkErrorPos,
                                   filled);
      gf2x_mod_add(s_dense, s_dense, saux);
   }   // end for
   // gf2x_mod_add(syndrome, syndrome, err+(N0-1)*NUM_DIGITS_GF2X_ELEMENT);
}
////////////////////////////////////////////////////////////////////////////////


