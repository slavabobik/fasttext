#!/bin/sh
# Builds refdump, the speed test (bench/ftbench.cc) and the fastText CLI in one
# of the reference variants.
#
# usage: build.sh <version> <variant> [<out-dir>]
#
# Versions (downloaded by fetch.sh unless FASTTEXT_SRC points at a checkout):
#
#   head     commit 1142dc4, the last commit before the repository was archived
#   v0.9.2   the 0.9.2 release
#
# Variants:
#
#   strict   -O2 -ffp-contract=off, no -march. Every float operation is a
#            single IEEE operation. Static on Linux.
#   shipped  fastText's own Makefile flags: -O3 -funroll-loops -march=native.
#
# CXX picks the compiler (default c++).
set -eu

version="${1:?version: head or v0.9.2}"
variant="${2:?variant: strict or shipped}"
here="$(cd "$(dirname "$0")" && pwd)"
pre=""
case "$version" in
  head) ;;
  v0.9.2)
    # 0.9.2 uses uint64_t without including <cstdint>, which gcc 13 and later
    # reject. Force the include instead of changing the source.
    pre="-include cstdint"
    ;;
  *)
    echo "unknown version $version" >&2
    exit 2
    ;;
esac
if [ -n "${FASTTEXT_SRC:-}" ]; then
  src="$FASTTEXT_SRC/src"
else
  "$here/fetch.sh" "$version"
  src="$here/../build/src/$version/src"
fi
out="${3:-$here/../build/$version-$variant-$(uname -s | tr A-Z a-z)-$(uname -m)}"
cxx="${CXX:-c++}"

case "$variant" in
  strict)
    flags="-O2 -DNDEBUG -ffp-contract=off -fno-fast-math"
    if [ "$(uname -s)" = Linux ]; then
      flags="$flags -static"
    fi
    ;;
  shipped)
    flags="-O3 -funroll-loops -DNDEBUG -march=native"
    ;;
  *)
    echo "unknown variant $variant" >&2
    exit 2
    ;;
esac

mkdir -p "$out/obj"
objs=""
for f in args autotune matrix dictionary loss productquantizer densematrix \
  quantmatrix vector model utils meter fasttext; do
  $cxx -pthread -std=c++17 -w $pre $flags -I"$src" -c "$src/$f.cc" -o "$out/obj/$f.o"
  objs="$objs $out/obj/$f.o"
done
$cxx -pthread -std=c++17 -w $pre $flags -I"$src" $objs "$src/main.cc" -o "$out/fasttext"
$cxx -pthread -std=c++17 -w $pre $flags -I"$src" $objs "$here/refdump.cc" -o "$out/refdump"
$cxx -pthread -std=c++17 -w $pre $flags -I"$src" $objs "$here/../bench/ftbench.cc" -o "$out/ftbench"
# Which maths-library routines the prediction code calls, read from the object
# files. Whether softmax calls exp or expf depends on the C++ library.
libm="$(nm -u "$out/obj/loss.o" "$out/obj/model.o" "$out/obj/fasttext.o" |
  awk '{print $NF}' | sed 's/^_//' | grep -E '^(exp|log)[a-z0-9]*$' | sort -u | tr '\n' ' ')"
rm -r "$out/obj"

{
  echo "version: $version"
  echo "variant: $variant"
  echo "flags: -pthread -std=c++17 -w $pre $flags"
  echo "compiler: $($cxx --version | head -1)"
  echo "system: $(uname -srm)"
  echo "libm calls: $libm"
} > "$out/BUILD"
cat "$out/BUILD"
