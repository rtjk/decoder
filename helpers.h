#pragma once

////////////////////////////////////////////////////////////////////////////////

/* benchmarking */
#ifdef BENCH
#include <cpucycles.h>
#define TESTS RUNS
#define WARMUP 1000
#define CPUCYCLES(test) ((test) >= WARMUP ? cpucycles() : 0)
/* profiling and testing */
#else
#define TESTS 1
#define WARMUP 0
#define CPUCYCLES(test) 0
#endif

#ifdef SKIP_INLINE
#define INLINE
#else
#define INLINE inline
#endif

#define RESTRICT
#define CONST

#ifdef DEBUG
#define DEBUG_PRINT(...)                            \
    do {                                            \
        printf("\033[0;33m");                       \
        printf(__VA_ARGS__);                        \
        printf("\033[0m");                          \
    } while(0)
#else
#define DEBUG_PRINT(...) ((void)0)
#endif

#define ALIGNED alignas(32)

#define ERROR(...)                                  \
    do {                                            \
        printf("\n\033[0;31m [!] %s: ", __func__);  \
        printf(__VA_ARGS__);                        \
        printf("\033[0m\n\n");                      \
        exit(1);                                    \
    } while(0)

#define BE 0                // big endian
#define LE 1                // little endian

#define IN                  // input parameter
#define OUT                 // output parameter

////////////////////////////////////////////////////////////////////////////////

/* fill an array of n 32-bit words with random values modulo m */
void u32_arr_rand_mod(uint32_t *arr, size_t n, size_t m) {
    for (size_t i = 0; i < n; i++) arr[i] = rand() % m;
}

int u32_cmp(const void *a, const void *b) {
    uint32_t ua = *(const uint32_t *)a;
    uint32_t ub = *(const uint32_t *)b;
    if (ua < ub) return -1;
    if (ua > ub) return 1;
    return 0;
}

/* fill an array of n 32-bit words with unique random values modulo m */
void u32_arr_rand_mod_unique(uint32_t *arr, size_t n, size_t m) {
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
    // qsort(arr, n, sizeof(uint32_t), u32_cmp);
}

////////////////////////////////////////////////////////////////////////////////

/* print an array of n 32-bit words in decimal format */
void u32_arr_print_dec(const uint32_t *arr, size_t n) {
    for (size_t i = 0; i < n; i++) printf("%lu ", (uint64_t)arr[i]);
    printf("\n");
}

/* print an array of n 8-bit words in decimal format */
void u8_arr_print_dec(const uint8_t *arr, size_t n) {
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