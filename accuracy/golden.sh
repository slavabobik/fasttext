#!/bin/sh
# Writes the golden files: reference dumps small enough to commit, so the Go
# tests can check the port without a C++ toolchain.
#
# usage: golden.sh <build dir> [<out dir>]
#
# Per model:
#   <model>.edge-cases.dump.gz      every id, k = 1, 5 and all labels
#   <model>.random-golden.dump.gz   every id, k = 1 and 5
#   <model>.tatoeba-golden.dump.gz  id count and hash, k = 1 and 5
#   <model>.threshold.dump.gz       the Tatoeba sample again with threshold
#                                   0.25, k = 1, 5 and all labels
#
# And once:
#   info.sha256   per model, the checksum of "refdump --info" without the
#                 header lines: every word and label with its count, and
#                 digests of the dense matrices. Checks the model reader.
#   heap.tsv      fingerprints of the C++ library's heap functions on scores
#                 with ties (heap/heapref.cc), if <build dir>/../heapref
#                 exists. Checks the port of those functions.
#
# Run it with the reference build (Linux, strict). The build description is
# copied to <out dir>/BUILD.
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
build="${1:?build dir}"
out="${2:-$here/golden}"
mkdir -p "$out"
cp "$build/BUILD" "$out/BUILD"
if command -v sha256sum >/dev/null; then
  sum="sha256sum"
else
  sum="shasum -a 256"
fi
: > "$out/info.sha256"
if [ -x "$build/../heapref" ]; then
  "$build/../heapref" > "$out/heap.tsv"
fi
for model in "$here"/models/*.bin "$here"/models/*.ftz; do
  m="$(basename "$model")"
  "$build/refdump" --ids "$model" 1,5,-1 < "$here/corpus/edge-cases.txt" 2>/dev/null |
    gzip -9n > "$out/$m.edge-cases.dump.gz"
  "$build/refdump" --ids "$model" 1,5 < "$here/corpus/random-golden.txt" 2>/dev/null |
    gzip -9n > "$out/$m.random-golden.dump.gz"
  "$build/refdump" "$model" 1,5 < "$here/corpus/tatoeba-golden.txt" 2>/dev/null |
    gzip -9n > "$out/$m.tatoeba-golden.dump.gz"
  "$build/refdump" "$model" 1,5,-1 0.25 < "$here/corpus/tatoeba-golden.txt" 2>/dev/null |
    gzip -9n > "$out/$m.threshold.dump.gz"
  printf '%s  %s\n' "$("$build/refdump" --info "$model" | grep -v '^#' | $sum | cut -d' ' -f1)" "$m" \
    >> "$out/info.sha256"
done
