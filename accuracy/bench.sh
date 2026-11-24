#!/bin/sh
# Times fastText's C++ library and the Go port on the same texts.
#
# usage: accuracy/bench.sh [<seconds per run>] [<threads>]
#
# Builds fastText (commit 1142dc4) with its own Makefile flags (-O3
# -funroll-loops -march=native), the fastest build, and the Go port's ftbench,
# plain and with GOEXPERIMENT=simd. Each program loads a model and predicts
# the texts of a corpus in a loop: one text per call, as fastText's Python
# package does. Prints a Markdown table per corpus, then one for texts of 16
# bytes to 64 KB (bench/lengths.sh); the raw numbers are in
# build/bench/bench.tsv.
#
# Environment: MODELS restricts the models (paths relative to accuracy/),
# SIMD=0 skips the SIMD build.
set -eu
cd "$(dirname "$0")"
secs="${1:-3}"
threads="${2:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
out=build/bench
mkdir -p "$out"
refdump/build.sh head shipped "$out/head-shipped" > /dev/null
(cd .. && CGO_ENABLED=0 go build -o "accuracy/$out/go/" ./cmd/ftbench)
tools="$out/head-shipped/ftbench $out/go/ftbench"
if [ "${SIMD:-1}" != 0 ]; then
  (cd .. && CGO_ENABLED=0 GOEXPERIMENT=simd go build -o "accuracy/$out/go-simd/" ./cmd/ftbench)
  tools="$tools $out/go-simd/ftbench"
fi
# shellcheck disable=SC2086
bench/run.sh "$secs" "$threads" $tools > "$out/bench.tsv"
python3 bench/summarize.py --brief "$out/bench.tsv"
echo "## lid.176.bin by text length, k = 1, one thread"
echo
# shellcheck disable=SC2086
bench/lengths.sh "$secs" $tools
