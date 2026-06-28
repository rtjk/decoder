#!/bin/bash

# exit on error
set -e

# inputs are in a text file, one per line
PRIMES="docs/primes_short.txt"
NPRIMES=$(wc -l < "$PRIMES")

# output directory
OUTPUT=$(date +"%Y-%m-%d-%H-%M-%S")
mkdir -p "$OUTPUT"
echo "*" > "$OUTPUT/.gitignore"

BENCH_DFR=0
BENCH_CC=1

TESTS_DFR=10000
TESTS_CC=10000

################################################################################

echo ">>>> compiling (serial)"

i=0
while IFS= read -r prime || [ -n "$prime" ]; do
    i=$((i + 1))
    echo -ne "$i/$NPRIMES\r"
    gcc -o "$OUTPUT/P-$prime.out" main.c -march=native -O3 -lcpucycles -lm -DP="$prime"
done < "$PRIMES"

################################################################################

echo ">>>> measuring clock cycles (serial)"

i=0
for f in "$OUTPUT"/*.out; do    
    i=$((i + 1))
    echo -ne "$i/$NPRIMES\r"
    taskset --cpu-list 0 "$f" $BENCH_CC $TESTS_CC >> "$OUTPUT/cc.csv"
done
sort -t, -k2,2n "$OUTPUT/cc.csv" -o "$OUTPUT/cc.csv"

################################################################################

echo ">>>> measuring decoding failure rate (parallel)"

FREE_CORES=1
TOTAL_CORES=$(nproc)
CORES=$((TOTAL_CORES - FREE_CORES))

find "$OUTPUT" -maxdepth 1 -type f -executable | \
    parallel -j "$CORES" --joblog $OUTPUT/log.txt \
    "taskset -c \$(( ({%} - 1) % $CORES )) {}" $BENCH_DFR $TESTS_DFR >> "$OUTPUT/dfr.csv"
sort -t, -k2,2n "$OUTPUT/dfr.csv" -o "$OUTPUT/dfr.csv"

find "$OUTPUT" -maxdepth 1 -type f -executable -delete

################################################################################

echo ">>>> done"
