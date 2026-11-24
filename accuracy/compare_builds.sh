#!/bin/sh
# Compares fastText builds with each other on every model and corpus.
#
# usage: compare_builds.sh <baseline build dir> <other build dir>...
#
# Environment:
#   MODELS   model files (default: every .bin and .ftz in models/)
#   CORPORA  corpus files (default: edge cases, random, Tatoeba sample)
#   KS       k list passed to refdump (default 1,5,-1)
#   JOBS     parallel comparisons (default 4)
#
# Writes one tab-separated line per model, corpus, build pair and k to standard
# output. The columns are listed in README.md. Nothing is stored: the
# dumps are streamed from refdump into dumpcmp.
set -eu

here="$(cd "$(dirname "$0")" && pwd)"
base="${1:?baseline build dir}"
shift

models="${MODELS:-$(ls "$here"/models/*.bin "$here"/models/*.ftz)}"
corpora="${CORPORA:-$here/corpus/edge-cases.txt $here/corpus/random.txt $here/corpus/tatoeba-test.txt}"
ks="${KS:-1,5,-1}"
jobs="${JOBS:-4}"
dumpcmp="${DUMPCMP:-$here/build/dumpcmp}"

one() {
  model="$1" corpus="$2" other="$3"
  label="$(basename "$model")	$(basename "$corpus" .txt)	$(basename "$base")	$(basename "$other")"
  a="$(mktemp -u)" b="$(mktemp -u)"
  mkfifo "$a" "$b"
  "$base/refdump" "$model" "$ks" < "$corpus" > "$a" 2>/dev/null &
  "$other/refdump" "$model" "$ks" < "$corpus" > "$b" 2>/dev/null &
  status=0
  "$dumpcmp" -tsv -show 0 -label "$label" "$a" "$b" || status=$?
  rm -f "$a" "$b"
  wait
  # 1 means "the dumps differ", which is a result and not a failure.
  [ "$status" -le 1 ]
}

n=0
for model in $models; do
  for corpus in $corpora; do
    for other in "$@"; do
      one "$model" "$corpus" "$other" &
      n=$((n + 1))
      if [ $((n % jobs)) -eq 0 ]; then
        wait
      fi
    done
  done
done
wait
