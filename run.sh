#!/bin/bash

# exit on error
set -e

PRIMES="docs/short_primes.txt"
OUTPUT="log.txt"

rm -rf "$OUTPUT"

NPRIMES=$(wc -l < "$PRIMES")
i=0

while IFS= read -r prime || [ -n "$prime" ]; do
    i=$((i + 1))
    echo -ne "$i/$NPRIMES\r"
    rm -f main; gcc -o main main.c -march=native -O3 -lcpucycles -DBENCH=1 -DP=$prime
    taskset --cpu-list 0 ./main >> "$OUTPUT"
done < "$PRIMES"

echo ">>>> done"
