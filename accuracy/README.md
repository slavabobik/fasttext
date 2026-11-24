# accuracy: checking the Go port against C++ fastText

This directory holds the tools and data that check whether the Go port returns the same bits as fastText's C++ code, and that time the two. The results and what they mean are in [`docs/VERIFICATION.md`](../docs/VERIFICATION.md) and [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md).

## Quick start

```sh
git lfs install && git lfs pull   # the t-*.bin models and the Tatoeba corpora
make testdata                     # lid.176.bin, assembled from ../lid176/model
make accuracy-docker              # verify.sh in the reference environment
```

On a Linux machine with gcc, `make accuracy` runs `verify.sh` directly. Results equal to the committed ones need the toolchain of the reference: gcc 14.2 and glibc 2.41, which is what [`reference/Dockerfile`](reference/Dockerfile) provides.

## What is here

| Path | Content |
| --- | --- |
| `verify.sh` | The end-to-end check. It builds fastText, checks that the build reproduces `golden/` and the committed checksums, then checks that the Go port's dumps (plain and SIMD) have the same checksums. |
| `bench.sh` | Times fastText (its own release flags) against the Go port, plain and SIMD, and prints Markdown tables. |
| `reference/Dockerfile` | The reference environment: Debian trixie (gcc 14.2, glibc 2.41) plus Go. |
| `refdump/refdump.cc` | The reference tool. It links against the unmodified fastText sources and writes, per input line, the ids, the hidden vector and the top-k labels with scores and probabilities, floats as raw bits. |
| `refdump/build.sh`, `refdump/fetch.sh` | Download fastText (`head` = commit 1142dc4, or `v0.9.2`) and build `refdump`, `ftbench` and the fastText CLI with one set of flags (`strict` or `shipped`). |
| `dumpcmp/` | Compares two dumps layer by layer: which texts differ, in which layer, by how much. |
| `checksums.sh`, `compare_checksums.sh` | The checksum of each dump, so builds on different machines can be compared without moving dumps. With `TOOL=ftdump` the dumps come from the Go port (`../cmd/ftdump`). |
| `compare_builds.sh`, `summarize.py` | Run every model and corpus through two or more builds and summarise the differences. |
| `golden.sh` | Writes `golden/` from a reference build. |
| `golden/` | Reference dumps small enough to commit (20 MB), read by `go test` (`TestGolden`, `TestModelReader`, `TestHeapMatchesLibstdcxx`). |
| `models/` | Test models with `SHA256SUMS`. The `.ftz` files are in git, the `t-*.bin` files in Git LFS. `lid.176.bin` is assembled from `../lid176/model` by `make testdata`. |
| `corpus/` | Test inputs with `SHA256SUMS`. `make_corpora.py` and `make_bench_corpus.py` generate them; the large Tatoeba files are in Git LFS. |
| `heap/heapref.cc` | Runs sets of scores with ties through the C++ library's heap routines, to check the Go port of them. |
| `libm/` | `variants.c` compares the two builds of `exp` and `log` inside glibc on every float. `public.c` fingerprints the routines a program actually gets. |
| `bench/` | `ftbench.cc` (C++ timing program), `run.sh`, `summarize.py`. The Go counterpart is `../cmd/ftbench`. |
| `results/` | Raw output of the runs reported in the docs. Machine names: `n4` = Xeon Platinum 8581C (AVX-512), `e2` = older Xeon (AVX2), `arm` = Neoverse N3, `mac`/`darwin` = Apple M4 Pro. `libm-variants-n4.tsv` is read by `TestLibmExhaustive`. |

## By hand

```sh
# One dump from each side
refdump/build.sh head strict build/ref
printf 'hello world\n' | build/ref/refdump --ids models/lid.176.ftz 1,5
(cd .. && go build -o accuracy/build/go/ ./cmd/ftdump)
printf 'hello world\n' | build/go/ftdump --ids models/lid.176.ftz 1,5

# Full corpora: Go against the committed reference checksums
TOOL=ftdump ./checksums.sh build/go | sort > build/checksums-go.tsv
./compare_checksums.sh results/checksums-ft-refdump-n4-head-strict.tsv build/checksums-go.tsv

# Layer-by-layer comparison of two builds
(cd .. && go build -o accuracy/build/dumpcmp ./accuracy/dumpcmp)
./compare_builds.sh build/ref build/other > build/compare.tsv
python3 summarize.py build/compare.tsv
```

## Result columns of `compare_builds.sh`

| # | Column |
| --- | --- |
| 1 to 5 | model, corpus, baseline build, other build, k |
| 6 | texts |
| 7 | header lines that differ |
| 8 | texts whose ids differ |
| 9, 10 | texts whose hidden vector differs; largest distance in representable values |
| 11 | texts where the number of predictions differs |
| 12 | texts whose best label differs |
| 13 | texts with the same number of predictions and a different label somewhere |
| 14, 15 | texts with a differing score; largest distance |
| 16, 17 | texts with a differing probability; largest distance |
| 18 | largest probability difference for the same label |
| 19 | for texts whose best label differs, the largest gap between the baseline's two best probabilities |
| 20 | texts whose two best probabilities are bit-equal in the baseline |
