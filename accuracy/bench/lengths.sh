#!/bin/sh
# Times fastText and the Go port on single texts of 16 bytes to 64 KB.
#
# usage: bench/lengths.sh <seconds per run> <program>...
#
# Run it from the accuracy directory; bench.sh does. Each program is an
# ftbench, as for bench/run.sh, and the first is the baseline. The texts are
# cut at a space from the paragraphs of bench-paragraphs.txt joined together,
# the same texts as BenchmarkDetectLength in ../langid. Model: lid.176.bin
# (LENGTH_MODEL overrides it), k = 1, one thread. Prints a Markdown table of
# microseconds per text, with the baseline's time divided by each program's
# in brackets.
set -eu
secs="$1"
shift
model="${LENGTH_MODEL:-models/lid.176.bin}"
dir=build/bench/lengths
mkdir -p "$dir"
python3 -I - "$dir" <<'PY'
import sys
out = sys.argv[1]
with open("corpus/bench-paragraphs.txt", "rb") as f:
    text = b" ".join(f.read().split(b"\n"))
for n in [16, 64, 256, 1 << 10, 4 << 10, 16 << 10, 64 << 10]:
    s = text
    if len(s) > n:
        i = s.rfind(b" ", 0, n + 1)
        s = s[:i] if i > 0 else s[:n]
    with open(f"{out}/{n}.txt", "wb") as f:
        f.write(s + b"\n")
PY

name() {
  case "$1" in
    *head-shipped*) echo "fastText" ;;
    *head-strict*) echo "fastText strict" ;;
    *go-simd/*) echo "Go simd" ;;
    *go/*) echo "Go" ;;
    *) echo "$1" ;;
  esac
}

head="| Bytes |"
rule="|---:|"
for tool in "$@"; do
  head="$head $(name "$tool") |"
  rule="$rule---:|"
done
echo "$head"
echo "$rule"
for n in 16 64 256 1024 4096 16384 65536; do
  row="| $n |"
  base=""
  for tool in "$@"; do
    ns=$("$tool" "$model" "$dir/$n.txt" 1 1 "$secs" | cut -f10)
    us=$(awk -v ns="$ns" 'BEGIN { v = ns / 1000; printf(v < 100 ? "%.2f" : "%.0f", v) }')
    if [ -z "$base" ]; then
      base="$ns"
      row="$row $us |"
    else
      row="$row $us ($(awk -v a="$base" -v b="$ns" 'BEGIN { printf("%.1f×", a / b) }')) |"
    fi
  done
  echo "$row"
done
