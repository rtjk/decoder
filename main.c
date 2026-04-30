#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdalign.h>
#include <assert.h>

////////////////////////////////////////////////////////////////////////////////

#include "parameters.h"
#include "test.h"

////////////////////////////////////////////////////////////////////////////////

#define TESTS 1000
#define SEED 42

////////////////////////////////////////////////////////////////////////////////

/* benchmarking */
#ifdef BENCH
#include <cpucycles.h>
#define CPUCYCLES() cpucycles()
#define RUNS TESTS
/* profiling and testing */
#else
#define CPUCYCLES() 0
#define RUNS 1
#endif

////////////////////////////////////////////////////////////////////////////////

/* fill an array of n 32-bit words with random values modulo m */
void u32_arr_rand_mod(uint32_t *arr, size_t n, size_t m) {
    for (size_t i = 0; i < n; i++) arr[i] = rand() % m;
}

/* fill an array of n 32-bit words with unique random values modulo m */
void u32_arr_rand_unique(uint32_t *arr, size_t n, size_t m) {
    if (n > m) ERROR("n > m");
    size_t placed = 0;
    while (placed < n) {
        uint32_t r = rand() % m;
        uint8_t unique = 1;
        for (size_t i = 0; i < placed && unique; i++) {
            if (arr[i] == r) unique = 0;
        }
        if (unique) {
            arr[placed] = r;
            placed++;
        }
    }
}

/* print an array of n 32-bit words in decimal format */
void u32_arr_print_dec(const uint32_t *arr, size_t n) {
    for (size_t i = 0; i < n; i++) printf("%lu ", (uint64_t)arr[i]);
    printf("\n");
}

/* print an array of n 64-bit words in binary format
 * bits in a byte are BE
 * bytes in a word can be BE or LE
 */
void u64_arr_print_bin(const uint64_t *arr, size_t n, uint8_t endianness) {
    for (size_t i = 0; i < n; i++) {
        uint64_t value = arr[i];
        if (endianness == LE) value = __builtin_bswap64(value);
        for (int bit = 64 - 1; bit >= 0; bit--) {
            uint64_t b = (value >> bit) & 1;
            printf("%lu", (uint64_t)b);
        }
        if (endianness == LE) value = __builtin_bswap64(value);
        printf(" ");
    }
    printf("\n");
}

////////////////////////////////////////////////////////////////////////////////

int main() {

    srand(SEED);

    uint64_t count_1;
    uint64_t count_2;
    uint64_t sum = 0;

    uint8_t checksum = 0;

    for (int run = 0; run < RUNS; run++) {

        uint64_t s_dense[NUM_DIGITS_GF2X_ELEMENT] = {0};        // syndrome (dense)
        uint64_t e_out_dense[N0*NUM_DIGITS_GF2X_ELEMENT] = {0}; // output error vector (dense)
        uint64_t e_in_dense[N0*NUM_DIGITS_GF2X_ELEMENT] = {0};  // input error vector (dense)
        uint32_t e_in_sparse[NUM_ERRORS_T] = {0};               // input error vector (sparse)
        uint32_t H_sparse[N0][V] = {0};                         // H (sparse)
        uint32_t Htr_sparse[N0][V] = {0};                       // H^T (sparse)
        uint64_t H_dense[N0][NUM_DIGITS_GF2X_ELEMENT] = {0};    // H (dense)

        /* sample H */
        for(int block = 0; block < N0; block++) {
            u32_arr_rand_unique(H_sparse[block], V, P);
            gf2x_mod_densify_VT(H_dense[block], H_sparse[block], V);
        }
        /* transpose H */
        transposeHPosOnes(Htr_sparse, H_sparse);
        /* sample error vector */
        u32_arr_rand_unique(e_in_sparse, NUM_ERRORS_T, N0*P);
        densify_error(e_in_dense, e_in_sparse);
        /* compute syndrome */
        compute_syndrome(s_dense, H_dense[0], e_in_sparse, e_in_dense);
        u64_arr_print_bin(s_dense, NUM_DIGITS_GF2X_ELEMENT, BE);
        
        // s_dense[0] = 1;

        /* decode */
        count_1 = CPUCYCLES();
        uint8_t ret = bf_decoding_CT(e_out_dense, Htr_sparse, H_sparse, s_dense);
        count_2 = CPUCYCLES();

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
rm -f main.o; gcc -o main.o main.c -march=native -O3 -lcpucycles -DBENCH=1
taskset --cpu-list 0 ./main.o
#### test
rm -f main.o; gcc -o main.o main.c -march=native -O2 -lcpucycles -fsanitize=address -Wall -pedantic -Wuninitialized -DDEBUG=1
taskset --cpu-list 0 ./main.o
*/
