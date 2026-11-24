#!/bin/sh
# Builds variants.c and runs it over all 2^32 float inputs in 16 slices,
# JOBS at a time (default 4). Linux x86-64 with static glibc only.
#
# Output columns: routine, first input, end of slice, inputs whose results
# differ before rounding to float, inputs whose float results differ, first
# such input, fingerprint of the non-FMA results.
set -eu
cd "$(dirname "$0")"
cc -O2 -ffp-contract=off -static variants.c -lm -o /tmp/variants
jobs="${JOBS:-4}"
n=0
for i in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
  next="$(printf '%x' $((0x$i + 1)))"
  /tmp/variants "${i}0000000" "${next}0000000" > "/tmp/variants.$i" &
  n=$((n + 1))
  if [ $((n % jobs)) -eq 0 ]; then
    wait
  fi
done
wait
cat /tmp/variants.? | sort
