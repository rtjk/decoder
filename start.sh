#!/bin/bash

case "$1" in
    bench)
	echo "========================================"
        echo "🚀 Running benchmark..."
        echo "========================================"
        rm -f main
        gcc -o main main.c -march=native -O3 -lcpucycles -DBENCH=1
        taskset --cpu-list 0 ./main
        ;;
    test)
	echo "========================================"
        echo "🧪 Running tests (AddressSanitizer + DEBUG)..."
        echo "========================================"
        rm -f main
        gcc -o main main.c \
            -march=native \
            -O2 \
            -g3 \
            -fsanitize=address \
            -Wall \
            -pedantic \
            -Wuninitialized \
            -Wno-unused-function \
            -Wno-unused-variable \
            -DDEBUG=1
        taskset --cpu-list 0 ./main
        ;;
    *)
        echo "Uso: $0 {bench|test}"
        exit 1
        ;;
esac
