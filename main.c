#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdalign.h>
#include <assert.h>
#include <cpucycles.h>

////////////////////////////////////////////////////////////////////////////////

#include "helpers.h"        // printers and random sampling
#include "parameters.h"     // scheme parameters
#include "test_utils.h"     // functions to compute syndrome, transpose H, etc.
#include "test_ref.h"       // reference decoding function
#include "test_bfmax.h"     // optimized decoding function
#include "test_opt.h"       //
#include "stats.h"          //

////////////////////////////////////////////////////////////////////////////////

// #define RUNS 100000
#define SEED 42

#define BENCH_DFR (0)
#define BENCH_CC  (1)

////////////////////////////////////////////////////////////////////////////////


int main(int argc, char *argv[]) {

    /* argument parsing */
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <parameter> <runs>\n", argv[0]);
        return 1;
    }
    const int mode = atoi(argv[1]);
    const int tests = atoi(argv[2]);

    srand(SEED);

    stats_t stats = {0};

    /* clock cycles */
    uint64_t count_1;
    uint64_t count_2;
    uint64_t sum = 0;
    double avg = 0;

    /* decoding failure rate */
    uint64_t checksum = 0;
    uint64_t failures = 0;
    uint64_t successes = 0;
    double dfr = 0;

    for (int test = 0; test < tests; test++) {

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

        /* decode */
        count_1 = cpucycles();
        // uint8_t ret = bf_decoder(e_out_dense, Htr_sparse, s_dense);
        uint8_t ret = bfmax_decoder(e_out_dense, Htr_sparse, H_sparse, s_dense);
        // uint8_t ret = OPT_bf_decoder(e_out_dense, Htr_sparse, s_dense);
        count_2 = cpucycles();

        /* compare error vectors */
        uint8_t cmp = memcmp(e_out_dense, e_in_dense, N0*NUM_DIGITS_GF2X_ELEMENT*sizeof(uint64_t));
        if (DEBUG && cmp != 0) ERROR("e_in != e_out");

        /* update cycles and dfr */
        if (cmp == 0) {
            successes++;
            stats_update(&stats, (double)(count_2 - count_1));
        } else {
            failures++;
        }
    }

    assert(successes + failures == tests);
    assert(stats.n == successes);
    dfr = (double)failures / tests;

    // mode, P, tests, failures, dfr
    if(mode == BENCH_DFR) printf("dfr, %d, %d, %d, %.9f\n", P, tests, failures, dfr);
    // mode, P, tests, failures, cc_stddev, cc_mean
    if(mode == BENCH_CC)  printf("cc, %d, %d, %d, %.2f, %.2f\n", P, tests, failures, stats_stddev(&stats), stats_mean(&stats));
}

////////////////////////////////////////////////////////////////////////////////

/*

rm -f main.out; gcc -o main.out main.c -march=native -O3 -lcpucycles -lm -DP=10883
taskset --cpu-list 0 ./main.out 1 10000

*/
