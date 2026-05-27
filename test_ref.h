#pragma once
#include "helpers.h"
#include "parameters.h"
#include "test_utils.h"
////////////////////////////////////////////////////////////////////////////////
// from test_utils:
// - population_count
// - gf2x_toggle_coeff
////////////////////////////////////////////////////////////////////////////////
void gf2x_copy(DIGIT dest[], CONST DIGIT in[])
{
   for (int i = NUM_DIGITS_GF2X_ELEMENT-1; i >= 0; i--)
      dest[i] = in[i];
} // end gf2x_copy
////////////////////////////////////////////////////////////////////////////////
DIGIT gf2x_get_coeff(CONST DIGIT poly[], CONST unsigned int exponent)
{
   unsigned int straightIdx = (NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_b -1) - exponent;
   unsigned int digitIdx = straightIdx / DIGIT_SIZE_b;
   unsigned int inDigitIdx = straightIdx % DIGIT_SIZE_b;
   return (poly[digitIdx] >> (DIGIT_SIZE_b-1-inDigitIdx)) & ((DIGIT) 1) ;
}
////////////////////////////////////////////////////////////////////////////////
int bf_decoder(DIGIT out[], // N0 polynomials
              CONST POS HtrPosOnes[N0][PAD32+V],
              DIGIT privateSyndrome[]  //  1 polynomial
              )
{
#if P < 64
#error The circulant block size should exceed 64
#endif

   uint8_t unsatParityChecks[N0*P];
   DIGIT currSyndrome[NUM_DIGITS_GF2X_ELEMENT];
   int check;
   int iteration = 0;

   do {
      gf2x_copy(currSyndrome, privateSyndrome);
      memset(unsatParityChecks,0x00,N0*P*sizeof(uint8_t));
      for (int i = 0; i < N0; i++) {
         for (int valueIdx = 0; valueIdx < P; valueIdx++) {
            for(int HtrOneIdx = 0; HtrOneIdx < V; HtrOneIdx++) {
               POS tmp = (HtrPosOnes[i][HtrOneIdx]+valueIdx) >= P ?
                                (HtrPosOnes[i][HtrOneIdx]+valueIdx) -P : (HtrPosOnes[i][HtrOneIdx]+valueIdx);
               if (gf2x_get_coeff(currSyndrome, tmp))
                  unsatParityChecks[i*P+valueIdx]++;
            }
         }
      }

      // computation of syndrome weight and threshold determination
      int syndrome_wt = population_count(currSyndrome);
      int min_idx=0;
      int max_idx;
      max_idx = sizeof(synd_corrt_vec)/(2*sizeof(unsigned int)) - 1;
      int thresh_table_idx = (min_idx + max_idx)/2;
      while(min_idx< max_idx) {
         if (synd_corrt_vec[thresh_table_idx][0] <= syndrome_wt) {
            min_idx = thresh_table_idx +1;
         } else {
            max_idx = thresh_table_idx -1;
         }
         thresh_table_idx = (min_idx +max_idx)/2;
      }
      int corrt_syndrome_based=synd_corrt_vec[thresh_table_idx][1];

      //Computation of correlation  with a full Q matrix
      for (int i = 0; i < N0; i++) {
         for (int j = 0; j < P; j++) {
            /* Correlation based flipping */
            if (unsatParityChecks[i*P+j] >= corrt_syndrome_based) {
               gf2x_toggle_coeff(out+NUM_DIGITS_GF2X_ELEMENT*i, j);
               for (int k = 0; k < V; k++) {
                  unsigned syndromePosToFlip;
                  for (int HtrOneIdx = 0; HtrOneIdx < V; HtrOneIdx++) {
                     syndromePosToFlip = (HtrPosOnes[i][k] + j) % P;
                     gf2x_toggle_coeff(privateSyndrome, syndromePosToFlip);
                  }
               } // end for v
            } // end if
         } // end for j
      } // end for i

      iteration = iteration + 1;
      check = 0;
      while (check < NUM_DIGITS_GF2X_ELEMENT && privateSyndrome[check++] == 0);

      DEBUG_PRINT("i: %d \t hw(s): %d \t UPCs: %d\n", iteration, syndrome_wt, check);

   } while (iteration < ITERATIONS_MAX && check < NUM_DIGITS_GF2X_ELEMENT);

   int hws = population_count(privateSyndrome);
   DEBUG_PRINT("i: L \t hw(s): %d \t UPCs: %d\n", hws, check);
   if(hws != 0) ERROR("hw(s)=%d decoding failure", hws);

   return (check == NUM_DIGITS_GF2X_ELEMENT);
}
////////////////////////////////////////////////////////////////////////////////