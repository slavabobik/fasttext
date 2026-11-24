#!/bin/sh
# Downloads the fastText sources the reference tools are built from.
#
# usage: fetch.sh <version> [<dir>]
#
#   head     commit 1142dc4c4ecbc19cc16eee5cdd28472e689267e6, the last commit
#            before facebookresearch/fastText was archived
#   v0.9.2   the 0.9.2 release, commit 5b5943c118b0ec5fb9cd8d20587de2b2d3966dfe
#
# The sources land in <dir> (default ../build/src/<version>).
set -eu
version="${1:?version: head or v0.9.2}"
here="$(cd "$(dirname "$0")" && pwd)"
dir="${2:-$here/../build/src/$version}"
case "$version" in
  head) commit=1142dc4c4ecbc19cc16eee5cdd28472e689267e6 ;;
  v0.9.2) commit=5b5943c118b0ec5fb9cd8d20587de2b2d3966dfe ;;
  *)
    echo "unknown version $version" >&2
    exit 2
    ;;
esac
if [ -f "$dir/src/fasttext.cc" ]; then
  exit 0
fi
mkdir -p "$dir"
git -C "$dir" init -q
git -C "$dir" fetch -q --depth 1 https://github.com/facebookresearch/fastText.git "$commit"
git -C "$dir" checkout -q FETCH_HEAD
echo "fastText $version ($commit) in $dir" >&2
