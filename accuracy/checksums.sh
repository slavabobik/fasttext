#!/bin/sh
# Prints a checksum of the refdump output for every model and corpus, so that
# builds on different machines can be compared without moving the dumps.
#
# usage: checksums.sh <build dir>
#
# Environment: MODELS, CORPORA, KS, JOBS as in compare_builds.sh, and TOOL:
# the program in the build dir to run, "refdump" by default. With TOOL=ftdump
# the same script produces the checksums of the Go port.
# Output: model, corpus, k list, sha256, tab-separated.
set -eu

here="$(cd "$(dirname "$0")" && pwd)"
build="${1:?build dir}"
models="${MODELS:-$(ls "$here"/models/*.bin "$here"/models/*.ftz)}"
corpora="${CORPORA:-$here/corpus/edge-cases.txt $here/corpus/random.txt $here/corpus/tatoeba-test.txt}"
ks="${KS:-1,5,-1}"
jobs="${JOBS:-4}"
tool="${TOOL:-refdump}"

if command -v sha256sum >/dev/null; then
  sum="sha256sum"
else
  sum="shasum -a 256"
fi

n=0
for model in $models; do
  for corpus in $corpora; do
    (
      h="$("$build/$tool" "$model" "$ks" < "$corpus" 2>/dev/null | $sum | cut -d' ' -f1)"
      printf '%s\t%s\t%s\t%s\n' "$(basename "$model")" "$(basename "$corpus" .txt)" "$ks" "$h"
    ) &
    n=$((n + 1))
    if [ $((n % jobs)) -eq 0 ]; then
      wait
    fi
  done
done
wait
