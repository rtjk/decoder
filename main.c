#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdalign.h>

////////////////////////////////////////////////////////////////////////////////

#define TESTS 100
#define ARR_SIZE 99999

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

////////////////////////////////////////////////////////////////////////////////

int main() {

    uint64_t count_1;
    uint64_t count_2;
    uint64_t sum = 0;

    uint8_t checksum = 0;

    uint8_t arr[ARR_SIZE];

    for (int i = 0; i < RUNS; i++) {

        count_1 = CPUCYCLES();
        u8_arr_rand(arr, sizeof(arr));
        count_2 = CPUCYCLES();

        sum += count_2 - count_1;

        for(size_t j = 0; j < sizeof(arr); j++) {
            checksum += arr[j];
        }
    }
    printf("[%d] %lu\n", checksum % 100, sum / RUNS);
}

////////////////////////////////////////////////////////////////////////////////

/*
rm -f main.o; gcc -o main.o main.c -march=native -O3 -lcpucycles -DBENCH=1
taskset --cpu-list 0 ./main.o
*/