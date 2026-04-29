#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdalign.h>

////////////////////////////////////////////////////////////////////////////////

#include "parameters.h"
#include "test.h"

////////////////////////////////////////////////////////////////////////////////

#define TESTS 100

////////////////////////////////////////////////////////////////////////////////

/* benchmarking */
#ifdef BENCH
#include <cpucycles.h>
#define CPUCYCLES() cpucycles()
#define RUNS TESTS
/* profiling */
#else
#define CPUCYCLES() 0
#define RUNS 1
#endif

////////////////////////////////////////////////////////////////////////////////

/* fill an array of n bytes with random values */
void u8_arr_rand(uint8_t *arr, size_t n) {
    for (size_t i = 0; i < n; i++) arr[i] = rand() % 256;
}

/* fill an array of n 32-bit words with random values modulo m */
void u32_arr_rand_mod(uint32_t *arr, size_t n, size_t m) {
    for (size_t i = 0; i < n; i++) arr[i] = rand() % m;
}

/* fill an array of n 32-bit words with unique random values modulo m */
void u32_arr_rand_unique(uint32_t *arr, size_t n, size_t m) {
    if (n > m) return; // not enough unique values
    uint32_t *pool = malloc(m * sizeof(uint32_t));
    for (size_t i = 0; i < m; i++) pool[i] = i;
    for (size_t i = 0; i < n; i++) {
        size_t idx = rand() % (m - i);
        arr[i] = pool[idx];
        pool[idx] = pool[m - 1 - i]; // remove used value
    }
    free(pool); 
}

////////////////////////////////////////////////////////////////////////////////

int main() {

    uint64_t count_1;
    uint64_t count_2;
    uint64_t sum = 0;

    uint8_t checksum = 0;

    for (int run = 0; run < RUNS; run++) {

        uint64_t e[N0][NUM_DIGITS_GF2X_ELEMENT] = {0};
        uint64_t s[NUM_DIGITS_GF2X_ELEMENT] = {0};
        uint32_t HPosOnes[N0][V] = {0};
        uint32_t HtrPosOnes[N0][V] = {0};
        uint64_t H[N0][NUM_DIGITS_GF2X_ELEMENT] = {0};
        for(int block = 0; block < N0; block++) {
            u32_arr_rand_unique(HPosOnes[block], V, P);
        }
        transposeHPosOnes(HtrPosOnes, HPosOnes);
        u32_arr_rand_unique(e, NUM_ERRORS_T, N0*P);
        ////
        s[0] = 1;
        ////

        count_1 = CPUCYCLES();
        uint8_t ret = bf_decoding_CT(e[0], HtrPosOnes, HPosOnes, s);
        count_2 = CPUCYCLES();

        sum += count_2 - count_1;

        checksum += ret;
    }
    printf("[%d] %lu\n", checksum % 100, sum / RUNS);
}

////////////////////////////////////////////////////////////////////////////////

/*
rm -f main.o; gcc -o main.o main.c -march=native -O3 -lcpucycles -DBENCH=1
rm -f main.o; gcc -o main.o main.c -march=native -O2 -lcpucycles -fsanitize=address -Wall -pedantic -Wuninitialized
taskset --cpu-list 0 ./main.o
*/