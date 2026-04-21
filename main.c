#include <cpucycles.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////

#define TESTS 50000

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

    uint8_t arr[1024];

    for (int i = 0; i < TESTS; i++) {

        count_1 = cpucycles();
        u8_arr_rand(arr, sizeof(arr));
        count_2 = cpucycles();

        sum += count_2 - count_1;

        checksum += arr[0];
    }
    printf("[%d] %llu\n", checksum % 100, sum / TESTS);
}

////////////////////////////////////////////////////////////////////////////////

/*
rm -f main.o; gcc -o main.o main.c -march=native -O3 -lcpucycles
taskset --cpu-list 0 ./main.o
*/