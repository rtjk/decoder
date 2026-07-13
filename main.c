#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdalign.h>
#include <assert.h>

////////////////////////////////////////////////////////////////////////////////

#include "helpers.h"        // printers and random sampling
#include "parameters.h"     // scheme parameters
#include "test_utils.h"     // functions to compute syndrome, transpose H, etc.
#include "test_ref.h"       // reference decoding function
#include "test_bfmax.h"     // optimized decoding function

////////////////////////////////////////////////////////////////////////////////

#define RUNS 10000
#define SEED 42

////////////////////////////////////////////////////////////////////////////////


int main() {

    srand(SEED);

    uint64_t count_1;
    uint64_t count_2;
    uint64_t sum = 0;

    uint8_t checksum = 0;

    for (int test = 0; test < TESTS + WARMUP; test++) {

        uint64_t s_dense[NUM_DIGITS_GF2X_ELEMENT] = {0};        // syndrome (dense)
        uint64_t e_out_dense[N0*NUM_DIGITS_GF2X_ELEMENT] = {0}; // output error vector (dense)
        uint64_t e_in_dense[N0*NUM_DIGITS_GF2X_ELEMENT] = {0};  // input error vector (dense)
        uint32_t e_in_sparse[NUM_ERRORS_T] = {0};               // input error vector (sparse)
        ALIGNED uint32_t H_sparse[N0][PAD32(V)] = {0};          // H (sparse)
        ALIGNED uint32_t Htr_sparse[N0][PAD32(V)] = {0};        // H^T (sparse)
        uint64_t H_dense[N0][NUM_DIGITS_GF2X_ELEMENT] = {0};    // H (dense)
        uint64_t Htr_dense[N0][NUM_DIGITS_GF2X_ELEMENT] = {0};  // H^T (dense)

        /* sample H */
        for(int block = 0; block < N0; block++) {
            u32_arr_rand_mod_unique(H_sparse[block], V, P);
            gf2x_mod_densify_VT(H_dense[block], H_sparse[block], V);
        }
        /* transpose H */
        transposeHPosOnes(Htr_sparse, H_sparse);
        for(int block = 0; block < N0; block++) {
            gf2x_mod_densify_VT(Htr_dense[block], Htr_sparse[block], V);
        }
        /* sample error vector */
        u32_arr_rand_mod_unique(e_in_sparse, NUM_ERRORS_T, N0*P);
        util_densify_error(e_in_dense, e_in_sparse);
        /* compute syndrome */
        util_compute_syndrome(s_dense, Htr_dense, e_in_sparse);

        uint8_t syndrome_bits[P];
        
        for (int i = 0; i < P; i++) {
            syndrome_bits[i] = gf2x_get_coeff(s_dense, i);
        }

        int hw = population_count(s_dense);

        /* decode */
        count_1 = CPUCYCLES(test);
        // uint8_t ret = bf_decoder(e_out_dense, Htr_sparse, s_dense);
        uint8_t ret = bfmax_decoder(e_out_dense, Htr_sparse, H_sparse, syndrome_bits, hw);
        count_2 = CPUCYCLES(test);

        /* compare error vectors */
        uint8_t cmp = memcmp(e_out_dense, e_in_dense, N0*NUM_DIGITS_GF2X_ELEMENT*sizeof(uint64_t));
        if (cmp != 0) ERROR("e_in != e_out");

        sum += count_2 - count_1;
        checksum += ret;
    }
    printf("[%d] %lu\n", checksum % 100, sum / RUNS);
}

////////////////////////////////////////////////////////////////////////////////

/*
#### benchmark
rm -f main; gcc -o main main.c -march=native -O3 -lcpucycles -DBENCH=1
taskset --cpu-list 0 ./main
#### test
rm -f main; gcc -o main main.c -march=native -O2 -g3 -fsanitize=address -Wall -pedantic -Wuninitialized -Wno-unused-function -Wno-unused-variable -DDEBUG=1
taskset --cpu-list 0 ./main
*/
