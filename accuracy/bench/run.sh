#!/bin/sh
# Times fastText and the Go port on every model: short texts and paragraphs,
# k = 1 and 5, one thread and many.
#
# usage: bench/run.sh <seconds per run> <threads> <program>...
#
# Run it from the accuracy directory. <threads> is the thread count of the
# "many" runs. Each program is an ftbench (bench/ftbench.cc built by
# refdump/build.sh, or fasttext/cmd/ftbench). MODELS overrides the model list.
#
# Output: the program's path, then its columns (see ftbench.cc), tab-separated.
# bench/summarize.py turns it into a table.
set -eu
secs="$1"
many="$2"
shift 2
models="${MODELS:-$(ls models/*.bin models/*.ftz)}"
for model in $models; do
  for corpus in corpus/tatoeba-golden.txt corpus/bench-paragraphs.txt; do
    for k in 1 5; do
      for threads in 1 "$many"; do
        for tool in "$@"; do
          printf '%s\t' "$tool"
          "$tool" "$model" "$corpus" "$k" "$threads" "$secs"
        done
      done
    done
  done
done
