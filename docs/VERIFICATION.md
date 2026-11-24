# Verification: the Go port against C++ fastText

This document answers two questions:

1. **If fastText itself is built in different ways and run on different machines, how much do its results differ?** That fixes which build the Go port has to match bit for bit. See [fastText against fastText](#fasttext-against-fasttext).
2. **Does the Go port return the same results as that build?** See [The Go port against the reference](#the-go-port-against-the-reference).

The measurements were made on 2026-11-20. The tools are in [`accuracy/`](../accuracy). [`accuracy/verify.sh`](../accuracy/verify.sh) repeats the central check on any Linux machine, and the [Accuracy workflow](../.github/workflows/accuracy.yml) runs it in CI on x86-64 and arm64.

## Summary

- **The Go port matches the reference bit for bit.** This covers 11 test models and two large published models (GlotLID, OpenLID-v2), with 233,381 texts per model and k of 1, 5 and all labels. Every input row, hidden vector, score and probability was compared as raw float bits.
- It was checked on macOS arm64, Linux arm64, and two kinds of Linux x86-64 (AVX-512 and AVX2), each with plain Go and with the SIMD kernels.
- **The reference** is fastText at its last commit (`1142dc4`), compiled by gcc 14.2 against glibc 2.41 with strict floating-point flags. On Linux x86-64 and Linux arm64 it gives the same bits. For `lid.176.bin` it also gives the same bits as fastText's own release build.
- **Other builds of fastText differ from each other** in the last bits of some probabilities: a Mac build, or a build with `-march=native` on quantized models. The best label never changed in any comparison.

## How the comparison works

- **`refdump`** ([`accuracy/refdump/refdump.cc`](../accuracy/refdump/refdump.cc)) links against the unmodified fastText sources. For each input line it writes:
  - the input row ids (words, character n-grams, word n-grams);
  - the hidden vector;
  - the top-k labels with their scores and probabilities.

  Floats are written as raw bits. It checks every line against fastText's public `predictLine`.
- **`cmd/ftdump`** writes the same format from the Go port.
- Two dumps of the same model and text must be **byte-identical**. [`accuracy/dumpcmp`](../accuracy/dumpcmp) compares dumps layer by layer when they are not.

Dump format, one line per text, tab-separated:

| Field | Content |
| --- | --- |
| `ids=<n>:<hash>` | Number of input rows and an FNV-1a hash of them; with `--ids`, the ids themselves |
| `h=<hex>` | Hidden vector, 8 hex digits per dimension |
| `k<k>=<label>:<score>:<prob>,…` | One field per requested k, best first: label index, bits of the log-probability, bits of the probability |

## Builds of fastText compared

| Name | Flags | Meaning |
| --- | --- | --- |
| `strict` | `-O2 -ffp-contract=off`, static on Linux | Every float operation is one IEEE operation. Reproducible in Go. |
| `shipped` | `-O3 -funroll-loops -march=native` | fastText's own Makefile flags. |

Each flag set was built from two source versions:

- `head`: commit `1142dc4`, the last before the repository was archived.
- `v0.9.2`: the 2020 release.

| Machine | Compiler and libraries |
| --- | --- |
| Apple M4 Pro, macOS | Apple clang 17, libc++, Apple's maths library |
| Linux x86-64, Intel Xeon Platinum 8581C (AVX-512) | gcc 14.2 (Debian), libstdc++, glibc 2.41 |
| Linux x86-64, older Intel Xeon (AVX2) | the same |
| Linux arm64, Arm Neoverse N3 | the same |

In [`accuracy/results`](../accuracy/results) the Linux machines are named after their cloud machine families: `n4` (Xeon Platinum 8581C), `e2` (older Xeon) and `arm` (Neoverse N3).

## Models

| Model | Output layer | Vector size | Notes |
| --- | --- | --- | --- |
| `lid.176.bin` | hierarchical softmax | 16 | The published model, 176 languages |
| `lid.176.ftz` | hierarchical softmax | 16 | Published, quantized and pruned |
| `t-softmax-d32.bin` | softmax | 32 | |
| `t-softmax-d32.ftz` | softmax | 32 | Quantized with `-qnorm`, pruned to 20,000 |
| `t-softmax-d256.bin` | softmax | 256 | Character n-grams 2 to 5 |
| `t-softmax-d10-wn2.bin` | softmax | 10 | Word 2-grams, no character n-grams |
| `t-hs-d20-wn3.bin` | hierarchical softmax | 20 | Word 3-grams |
| `t-hs-d20-wn3.ftz` | hierarchical softmax | 20 | Output matrix quantized too (`-qout`) |
| `t-ova-d64.bin` | one-vs-all | 64 | |
| `t-ova-d64.ftz` | one-vs-all | 64 | `-qnorm -qout -dsub 4`, pruned to 5,000 |
| `t-ns-d16.bin` | negative sampling | 16 | |

The `t-` models were trained on a Tatoeba slice (323 labels) only to exercise every code path; they are not good classifiers. Training is not repeatable bit for bit, so the files themselves are the fixture. They are stored in the repository with Git LFS.

Two large published models were also checked:

- **GlotLID** (`cis-lmu/glotlid`, 783 MB, vector size 96, 2,060 labels, softmax).
- **OpenLID-v2** (`laurievb/OpenLID-v2`, about 1.2 GB, vector size 256, 200 labels, softmax).

Both were downloaded from Hugging Face, and their SHA-256 matched the published one.

## Corpora

| File | Texts | Content |
| --- | --- | --- |
| `edge-cases.txt` | 156 | Hand-written: empty and separator-only lines, `__label__` tokens, a literal `</s>`, invalid UTF-8, emoji, 1,022 to 5,000 tokens around the 1,024-token limit, short chat-style messages |
| `random.txt` | 20,000 | Seeded: raw bytes, separator-heavy bytes, random Unicode, damaged sentences |
| `tatoeba-test.txt` | 213,225 | Every 64th sentence of the Tatoeba export, 323 languages |
| `random-golden.txt`, `tatoeba-golden.txt` | 1,000 and 2,012 | Every 20th and every 106th line of the two above, for the golden files |
| `bench-paragraphs.txt` | 144 | Timing only: for each of 144 languages, its first 20 test sentences joined into one line (855 bytes on average) |

Each text is passed to fastText as `line + "\n"`, which is what the Python package does. Tatoeba sentences are © their contributors, under CC-BY 2.0 FR.

## fastText against fastText

Every comparison ran all 11 models over the 233,381 texts with k of 1, 5 and all labels.

### What never differed

- **Input rows:** identical in every comparison.
- **Best label:** identical in every comparison, on every text.
- **Source version:** `head` and `v0.9.2` give bit-identical dumps, with both flag sets, on the Mac and on Linux.
- **x86 CPU type:** the AVX-512 and AVX2 machines give bit-identical dumps, with both flag sets.
- **CPU architecture:** a strict build on Linux arm64 gives the same bits as on Linux x86-64, for all 13 models.

### Linux: strict against shipped

Six of the seven dense models are bit-identical, including `lid.176.bin`. The other five:

| Model | Hidden vector differs | Best probability differs | Largest probability difference | Full ranking differs |
| --- | --- | --- | --- | --- |
| `lid.176.ftz` | 230,369 | 41,657 (17.8%) | 8.3e-7 | 57 |
| `t-softmax-d32.ftz` | 232,757 | 49,254 (21.1%) | 1.9e-6 | 5,699 |
| `t-hs-d20-wn3.ftz` | 0 | 48,215 (20.7%) | 5.4e-7 | 122 |
| `t-softmax-d10-wn2.bin` | 0 | 20,762 (8.9%) | 7.7e-7 | 1,479 |
| `t-ova-d64.ftz` | 232,725 | 2 | 0.0078 | 80 |

- **Cause:** fused multiply-add. With `-march=native` the compiler may merge a multiply and an add into one instruction with one rounding.
- **The 0.0078 outlier:** the one-vs-all and negative-sampling layers read the sigmoid from a 512-entry table, and a last-bit change in the input can move to the next entry.

### Mac against Linux, both strict

The hidden vectors match on every model. The probabilities differ by at most 1.8e-7, because Apple's `exp` and `log` round differently from glibc's. There is a second difference: the order of labels with exactly equal probabilities (thousands of texts for the one-vs-all and negative-sampling models). fastText's heap compares scores only, and libc++ and libstdc++ order equal scores differently.

### glibc on CPUs with and without FMA

glibc contains two builds of `exp`, `expf`, `log` and `logf`, and picks one when the program starts. Both were run on every `float` bit pattern ([`accuracy/libm/variants.c`](../accuracy/libm/variants.c)). They differ, after rounding to `float`, only in `expf`, and only on 2 of the 4,294,967,296 inputs. The Go port follows the non-FMA build, which is also what glibc uses on arm64. The port and fastText on an FMA machine could differ only if a hierarchical-softmax node's dot product were exactly -32.5646. That did not happen on any text.

### Conclusions

1. **The reference build** is fastText `head`, Linux, gcc 14.2, static glibc 2.41, `strict` flags. It is what [`accuracy/reference/Dockerfile`](../accuracy/reference/Dockerfile) builds, and what wrote [`accuracy/golden`](../accuracy/golden).
2. **For `lid.176.bin` the reference is not a special case.** Strict and shipped, `head` and `v0.9.2`, and both x86 CPUs all give the same bits.
3. **For quantized models and some vector sizes that are not a multiple of 16,** a shipped build differs from the reference in the last bits of the probability. The best label never changes.
4. **To match the reference, the port needs three things beyond plain arithmetic:**
   - glibc's `expf`, `exp` and `log` algorithms (non-FMA build);
   - libstdc++'s heap routines, for the order of ties;
   - the sigmoid and log tables of the one-vs-all and negative-sampling layers.

## The Go port against the reference

### Whole corpora

The table counts model and corpus pairs whose complete dump has the same SHA-256 as the dump of the reference build (k of 1, 5 and all labels):

| Program | Machine | Inner loops | 11 test models, 3 corpora | GlotLID and OpenLID-v2, 3 corpora |
| --- | --- | --- | --- | --- |
| Go | Apple M4 Pro, macOS | plain | 33 of 33 | not run |
| Go simd | Apple M4 Pro, macOS | NEON | 33 of 33 | not run |
| Go | Linux x86-64, Xeon 8581C | plain | 33 of 33 | 6 of 6 |
| Go simd | Linux x86-64, Xeon 8581C | AVX-512 | 33 of 33 | 6 of 6 |
| Go | Linux x86-64, older Xeon | plain | 33 of 33 | not run |
| Go simd | Linux x86-64, older Xeon | AVX2 | 33 of 33 | not run |
| Go | Linux arm64, Neoverse N3 | plain | 33 of 33 | 6 of 6 |
| Go simd | Linux arm64, Neoverse N3 | NEON | 33 of 33 | 6 of 6 |

- GlotLID was run on the Tatoeba sample with k of 1 and 5 only; with all 2,060 labels its dump would be 11 GB.
- "Not run" means the model did not fit on that machine.

Files: `accuracy/results/checksums-*.tsv`, `port-checksums-*.tsv`, `port-large-*.tsv`.

### Parts checked on their own

`go test` runs these:

| Part | Test | Compared with |
| --- | --- | --- |
| Model file reader | `TestModelReader` | A hash of every header field, dictionary entry and matrix from `refdump --info`, 11 models (`golden/info.sha256`) |
| Ids, hidden vector, scores, probabilities | `TestGolden` | The dumps in `golden/`: 3,168 texts per model, 2,012 of them a second time with a threshold; run once per kernel |
| `expf`, `exp`, `log(x + 1e-5)` | `TestLibmExhaustive` | glibc's non-FMA build on all 2^32 `float` values, through fingerprints in `results/libm-variants-n4.tsv` |
| Order of tied labels | `TestHeapMatchesLibstdcxx` | libstdc++ on 350 sets of scores with ties (`golden/heap.tsv`) |
| SIMD and unrolled loops | `TestKernels` | Plain loops with the same order of operations, bit for bit |
| Use from many goroutines | `TestConcurrent`, also under `-race` | Single-threaded results |
| Memory | `TestNoAllocation` | No allocation per prediction with `AppendPredict` |
| Damaged files | `TestDamagedModel` | Truncated files must fail to load. Files with changed bytes may load, but nothing may crash, in loading or in predicting |

### Against fastText's Python package

On a separate set of 39,196 real-world chat messages (157 bytes on average), the port was compared with fastText's Python package (`fasttext` 0.9.3 on macOS) for four models: `lid.176.bin`, GlotLID v3, OpenLID v1 and OpenLID v2. The data itself is not public.

- **The detected language was the same for every message and every model.**
- Every difference in the order of the second and third label was an exact tie in probability, ordered differently by libc++ (see [Mac against Linux](#mac-against-linux-both-strict)).
- The probabilities differed by at most 1.2e-7, the difference between Apple's maths library and glibc's.

## Not covered

- **Other glibc versions, and musl.** Only glibc 2.41 was tested. This matters only when comparing against a fastText binary: the Go port does not use the C library.
- **Model files older than format version 12.** The port reads them the way fastText does, but no such file was tested. All models named here are version 12.
- **Machines:** one CPU model per kind.

## Found on the way

- **fastText `head` fails to train with fewer than 10 threads.** Training stopped with `Encountered NaN` with 1 and with 8 threads. Reading the code, `DenseMatrix::uniform` splits the input matrix into 10 blocks and fills one block per thread, so with fewer threads part of the matrix is never initialised. Prediction is not affected. The `t-` models were trained with `v0.9.2`.
- **`v0.9.2` does not compile with gcc 14** without `-include cstdint`. `head` compiles as is.
- **Go on x86-64: a number conversion in a loop can serialise the loop.** The softmax loop converted a `float32` to `float64`, called `exp`, and converted back. On amd64 such a conversion waits for the old content of the register it writes, so every `exp` waited for the one before. The port now does the conversions and the `exp` calls in separate loops, with the same results.
- **Go 1.27 SIMD on x86-64 needs `archsimd.ClearAVXUpperBits()` after each kernel.** Without it each kernel call was about 160 ns slower.

## Reproducing

```sh
git lfs install && git lfs pull   # test models and corpora
make testdata                     # assembles lid.176.bin from lid176/model
make golden                       # every Go test, including the golden dumps
make accuracy-docker              # fastText vs Go on the full corpora (needs Docker)
```

See [`accuracy/README.md`](../accuracy/README.md) for the individual tools.
