#!/bin/bash

WARNING_EXCEPTIONS="-Wall -Wno-write-strings -Wno-deprecated-declarations -Wno-comment -Wno-switch -Wno-null-dereference -Wno-tautological-compare -Wno-unused-result -Wno-missing-declarations"
OUTPUT_DIR="bin"

COMPILER_FLAGS="-D_GNU_SOURCE $WARNING_EXCEPTIONS -fPIC -g -O0"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir $OUTPUT_DIR
fi

gcc $COMPILER_FLAGS -o $OUTPUT_DIR/cgrad ./src/cgrad.c -lm

