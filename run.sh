#!/bin/bash

# exit on error
set -e

PRIMES="docs/short_primes.txt"
OUTPUT=$(date +"%Y-%m-%d-%H-%M-%S")

mkdir -p "$OUTPUT"
echo "*" > "$OUTPUT/.gitignore"

################################################################################

echo ">>>> compiling"

NPRIMES=$(wc -l < "$PRIMES")
i=0

while IFS= read -r prime || [ -n "$prime" ]; do
    i=$((i + 1))
    echo -ne "$i/$NPRIMES\r"
    gcc -o "$OUTPUT/P-$prime" main.c -march=native -O3 -lcpucycles -DBENCH=1 -DP="$prime"
done < "$PRIMES"

################################################################################

echo ">>>> running"

FREE_CORES=1
TOTAL_CORES=$(nproc)
CORES=$((TOTAL_CORES - FREE_CORES))

find "$OUTPUT" -maxdepth 1 -type f -executable | \
    parallel -j "$CORES" --joblog $OUTPUT/log.txt \
    "taskset -c \$(( ({%} - 1) % $CORES )) {}" >> "$OUTPUT/out.csv"

find "$OUTPUT" -maxdepth 1 -type f -executable -delete

################################################################################

echo ">>>> done"
