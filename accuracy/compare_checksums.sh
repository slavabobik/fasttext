#!/bin/sh
# Compares two checksum files written by checksums.sh and prints how many
# model and corpus pairs differ, and for which models.
#
# usage: compare_checksums.sh <a.tsv> <b.tsv>
set -eu
a="$1" b="$2"
key() { awk -F'\t' '{print $1 "/" $2 "\t" $4}' "$1" | sort; }
ta="$(mktemp)" tb="$(mktemp)"
key "$a" > "$ta"
key "$b" > "$tb"
join -t "$(printf '\t')" "$ta" "$tb" | awk -F'\t' -v a="$(basename "$a" .tsv)" -v b="$(basename "$b" .tsv)" '
  { n++; if ($2 != $3) { d++; split($1, p, "/"); m[p[1]] = 1 } }
  END {
    l = ""
    for (k in m) l = l " " k
    printf "%s vs %s: %d of %d differ%s%s\n", a, b, d, n, (d ? ":" : ""), l
  }'
rm -f "$ta" "$tb"
