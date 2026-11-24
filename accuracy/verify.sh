#!/bin/sh
# Checks that the Go port gives the same results as fastText's C++ code, bit
# for bit, on every test model and corpus.
#
# usage: accuracy/verify.sh            (from the repository root, on Linux)
#
# Steps:
#   1. Build fastText (commit 1142dc4) with the reference flags: refdump, the
#      C++ library's heap fingerprints, and the fastText CLI.
#   2. Check that this build reproduces the committed reference: the golden
#      dumps in golden/ and the checksums of the full corpora in
#      results/checksums-ft-refdump-n4-head-strict.tsv.
#   3. Build the Go port's ftdump, plain and with GOEXPERIMENT=simd, and
#      compare the checksums of its dumps with those of the C++ build.
#
# A dump holds, for each text, the input rows, the hidden vector and the k
# best labels with scores and probabilities as raw float bits, for k of 1, 5
# and all labels. Any difference in any bit fails the check.
#
# Run it in the reference environment (reference/Dockerfile) to compare with
# the committed results; "make accuracy" does that. Needs the Git LFS files
# ("git lfs pull") and lid.176.bin ("make testdata").
#
# Environment: JOBS (parallel dumps, default: number of CPUs), SIMD=0 to skip
# the SIMD build (it needs Go 1.27 or later).
set -eu
cd "$(dirname "$0")"
here="$(pwd)"
jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
out="$here/build/verify"
rm -rf "$out"
mkdir -p "$out"
fail=0

for f in models/*.bin models/*.ftz corpus/tatoeba-test.txt; do
  if head -c 23 "$f" | grep -q 'version https://git-lfs'; then
    echo "$f is a Git LFS pointer: run \"git lfs pull\"" >&2
    exit 1
  fi
done
if [ ! -f models/lid.176.bin ]; then
  echo "models/lid.176.bin is missing: run \"make testdata\"" >&2
  exit 1
fi
(cd models && sha256sum -c --quiet SHA256SUMS) || { echo "model checksums differ" >&2; exit 1; }
sha256sum -c --quiet corpus/SHA256SUMS || { echo "corpus checksums differ" >&2; exit 1; }

echo "== 1. Build fastText (head, strict flags)"
refdump/build.sh head strict "$out/ref" > /dev/null
cat "$out/ref/BUILD"
c++ -O2 -std=c++17 heap/heapref.cc -o "$out/heapref"
if [ "$(grep -v '^system:' golden/BUILD)" != "$(grep -v '^system:' "$out/ref/BUILD")" ]; then
  echo "note: this build differs from golden/BUILD (compiler or flags); results may differ" >&2
fi

echo "== 2. The C++ build reproduces the committed reference"
"$here/golden.sh" "$out/ref" "$out/golden" > /dev/null
# same <committed> <new>: the dumps, not the gzip bytes, which depend on the
# gzip version.
same() {
  case "$1" in
    *.gz)
      gzip -dc "$2" > "$out/new.dump" || return 1
      gzip -dc "$1" | cmp -s - "$out/new.dump"
      ;;
    *) cmp -s "$1" "$2" ;;
  esac
}
n=0 d=0
for f in golden/*.dump.gz golden/info.sha256 golden/heap.tsv; do
  n=$((n + 1))
  if ! same "$f" "$out/golden/$(basename "$f")"; then
    echo "  differs: $f"
    d=$((d + 1))
  fi
done
echo "  golden files: $((n - d)) of $n identical"
[ "$d" -eq 0 ] || fail=1

JOBS="$jobs" ./checksums.sh "$out/ref" | sort > "$out/checksums-ref.tsv"
r="$(./compare_checksums.sh results/checksums-ft-refdump-n4-head-strict.tsv "$out/checksums-ref.tsv")"
echo "  full corpora: $r"
case "$r" in *": 0 of "*) ;; *) fail=1 ;; esac

echo "== 3. The Go port matches the C++ build"
(cd .. && CGO_ENABLED=0 go build -o "$out/go/" ./cmd/ftdump)
variants="go"
if [ "${SIMD:-1}" != 0 ]; then
  (cd .. && CGO_ENABLED=0 GOEXPERIMENT=simd go build -o "$out/go-simd/" ./cmd/ftdump)
  variants="go go-simd"
fi
for v in $variants; do
  TOOL=ftdump JOBS="$jobs" ./checksums.sh "$out/$v" | sort > "$out/checksums-$v.tsv"
  r="$(./compare_checksums.sh "$out/checksums-ref.tsv" "$out/checksums-$v.tsv")"
  echo "  $v: $r"
  case "$r" in *": 0 of "*) ;; *) fail=1 ;; esac
done

if [ "$fail" -ne 0 ]; then
  echo "FAIL: results differ"
  exit 1
fi
echo "OK: the Go port gives the same bits as fastText on every model and corpus"
