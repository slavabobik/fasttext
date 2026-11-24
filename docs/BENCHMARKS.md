# Benchmarks: the Go port against C++ fastText

[`accuracy/bench/ftbench.cc`](../accuracy/bench/ftbench.cc) calls fastText's C++ library the way its Python package does: one text, one call to `predictLine`. [`cmd/ftbench`](../cmd/ftbench) does the same with the Go port's `AppendPredict`. Both programs load a model, then predict the texts of a corpus in a loop for 3 seconds.

- **fastText** is built with its own Makefile flags (`-O3 -funroll-loops -march=native`), the fastest of the builds tried.
- **Go** is the default build. **Go simd** is built with `GOEXPERIMENT=simd` (Go 1.27), which uses NEON on arm64 and AVX2 or AVX-512 on x86-64 for the inner loops. Both builds return the same bits.
- **Sentences** are the 2,012 lines of `tatoeba-golden.txt`, 41 bytes on average. **Paragraphs** are the 144 lines of `bench-paragraphs.txt`, 855 bytes on average.
- **Numbers** are microseconds per text with k = 1 on one thread. The factor in brackets is fastText's time divided by the port's, so above 1 means the port is faster.
- **The last two columns** show how many times the throughput on sentences grows when every CPU runs a thread. One model is shared by all threads in both programs.

Measured on 2026-11-20. Raw files: [`accuracy/results/bench-*.tsv`](../accuracy/results), which also hold k = 5, fastText's strict build, load time and peak memory.

## Language identification models

| Model | Machine | fastText | Go | Go simd |
|---|---|---:|---:|---:|
| `lid.176.bin` | Intel Xeon Platinum 8581C | 3.2 | 2.6 (1.3×) | 2.2 (1.4×) |
| | Arm Neoverse N3 | 4.4 | 2.8 (1.5×) | 2.7 (1.6×) |
| | Apple M4 Pro | 2.3 | 1.2 (2.0×) | 1.0 (2.2×) |
| `lid.176.ftz` | Intel Xeon Platinum 8581C | 5.8 | 3.4 (1.7×) | 3.4 (1.7×) |
| | Arm Neoverse N3 | 6.1 | 2.7 (2.2×) | 2.8 (2.2×) |
| | Apple M4 Pro | 3.8 | 1.7 (2.3×) | 1.7 (2.2×) |
| GlotLID | Intel Xeon Platinum 8581C | 105 | 90.9 (1.2×) | 32.6 (3.2×) |
| | Arm Neoverse N3 | 131 | 96.4 (1.4×) | 52.0 (2.5×) |
| OpenLID-v2 | Intel Xeon Platinum 8581C | 34.6 | 33.4 (1.0×) | 11.8 (2.9×) |
| | Arm Neoverse N3 | 57.7 | 35.0 (1.6×) | 21.9 (2.6×) |

Loading and memory are about the same as fastText's:

- `lid.176.bin` loads in 95 ms against fastText's 97 ms on the Xeon, and 29 ms against 41 ms on the Mac.
- Peak memory: 140 MB against 137 MB for `lid.176.bin`, 759 MB against 758 MB for GlotLID, and 1,238 MB against 1,204 MB for OpenLID-v2.

## All models, per machine

### Apple M4 Pro (macOS arm64), 10 threads in the last two columns

| Model | fastText, sentences | Go | Go simd | fastText, paragraphs | Go | Go simd | Go, 10 threads | fastText, 10 threads |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lid.176.bin` | 2.3 | 1.2 (2.0×) | 1.0 (2.2×) | 26.2 | 14.4 (1.8×) | 11.8 (2.2×) | 9.5× | 9.4× |
| `lid.176.ftz` | 3.8 | 1.7 (2.3×) | 1.7 (2.2×) | 54.6 | 23.9 (2.3×) | 24.0 (2.3×) | 9.3× | 9.5× |
| `t-hs-d20-wn3.bin` | 2.0 | 1.1 (1.8×) | 1.1 (1.9×) | 18.8 | 10.3 (1.8×) | 9.4 (2.0×) | 9.4× | 9.3× |
| `t-hs-d20-wn3.ftz` | 2.7 | 2.0 (1.4×) | 1.9 (1.4×) | 28.9 | 21.4 (1.4×) | 21.2 (1.4×) | 9.1× | 9.3× |
| `t-ns-d16.bin` | 3.2 | 1.9 (1.7×) | 1.5 (2.2×) | 21.0 | 11.1 (1.9×) | 9.6 (2.2×) | 9.2× | 9.1× |
| `t-ova-d64.bin` | 6.1 | 4.5 (1.4×) | 2.6 (2.4×) | 24.9 | 21.1 (1.2×) | 12.5 (2.0×) | 9.3× | 9.5× |
| `t-ova-d64.ftz` | 19.0 | 14.2 (1.3×) | 14.1 (1.3×) | 70.8 | 59.9 (1.2×) | 59.3 (1.2×) | 8.9× | 9.5× |
| `t-softmax-d10-wn2.bin` | 3.5 | 2.1 (1.6×) | 1.9 (1.8×) | 11.7 | 4.6 (2.5×) | 4.7 (2.5×) | 9.3× | 9.4× |
| `t-softmax-d256.bin` | 32.3 | 19.3 (1.7×) | 9.9 (3.3×) | 97.0 | 106 (0.9×) | 52.6 (1.8×) | 9.0× | 9.3× |
| `t-softmax-d32.bin` | 4.7 | 3.8 (1.2×) | 2.8 (1.7×) | 27.9 | 20.9 (1.3×) | 15.2 (1.8×) | 9.3× | 9.6× |
| `t-softmax-d32.ftz` | 7.3 | 5.7 (1.3×) | 5.1 (1.4×) | 73.8 | 48.7 (1.5×) | 48.3 (1.5×) | 9.2× | 9.4× |

### Intel Xeon Platinum 8581C (Linux x86-64, AVX-512), 4 vCPUs

| Model | fastText, sentences | Go | Go simd | fastText, paragraphs | Go | Go simd | Go, 4 threads | fastText, 4 threads |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lid.176.bin` | 3.2 | 2.6 (1.3×) | 2.2 (1.4×) | 40.4 | 32.0 (1.3×) | 26.3 (1.5×) | 3.9× | 3.9× |
| `lid.176.ftz` | 5.8 | 3.4 (1.7×) | 3.4 (1.7×) | 87.0 | 47.1 (1.8×) | 46.5 (1.9×) | 3.9× | 3.8× |
| `t-hs-d20-wn3.bin` | 3.0 | 2.4 (1.2×) | 2.3 (1.3×) | 28.8 | 22.5 (1.3×) | 19.8 (1.5×) | 4.2× | 3.7× |
| `t-hs-d20-wn3.ftz` | 4.7 | 4.0 (1.2×) | 4.1 (1.2×) | 49.4 | 45.0 (1.1×) | 44.9 (1.1×) | 3.1× | 4.2× |
| `t-ns-d16.bin` | 6.1 | 3.9 (1.5×) | 2.6 (2.3×) | 30.6 | 24.4 (1.3×) | 18.2 (1.7×) | 3.9× | 4.0× |
| `t-ova-d64.bin` | 11.1 | 9.3 (1.2×) | 3.7 (3.0×) | 35.7 | 37.5 (1.0×) | 20.9 (1.7×) | 3.9× | 3.9× |
| `t-ova-d64.ftz` | 40.5 | 34.3 (1.2×) | 34.2 (1.2×) | 115 | 133 (0.9×) | 140 (0.8×) | 4.2× | 4.2× |
| `t-softmax-d10-wn2.bin` | 6.7 | 4.3 (1.5×) | 3.5 (1.9×) | 21.2 | 11.8 (1.8×) | 12.1 (1.8×) | 3.9× | 4.2× |
| `t-softmax-d256.bin` | 46.3 | 36.4 (1.3×) | 12.7 (3.6×) | 137 | 166 (0.8×) | 90.0 (1.5×) | 3.9× | 4.0× |
| `t-softmax-d32.bin` | 9.1 | 7.9 (1.2×) | 5.1 (1.8×) | 43.3 | 39.6 (1.1×) | 29.5 (1.5×) | 4.0× | 3.8× |
| `t-softmax-d32.ftz` | 14.1 | 12.3 (1.2×) | 9.4 (1.5×) | 127 | 102 (1.2×) | 95.9 (1.3×) | 4.2× | 3.8× |
| GlotLID | 105 | 90.9 (1.2×) | 32.6 (3.2×) | 191 | 186 (1.0×) | 89.8 (2.1×) | 4.3× | 4.2× |
| OpenLID-v2 | 34.6 | 33.4 (1.0×) | 11.8 (2.9×) | 152 | 287 (0.5×) | 124 (1.2×) | 4.4× | 4.1× |

### Older Intel Xeon (Linux x86-64, AVX2), 2 vCPUs

| Model | fastText, sentences | Go | Go simd | fastText, paragraphs | Go | Go simd | Go, 2 threads | fastText, 2 threads |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lid.176.bin` | 9.0 | 7.3 (1.2×) | 6.6 (1.4×) | 107 | 101 (1.1×) | 84.6 (1.3×) | 2.0× | 2.1× |
| `lid.176.ftz` | 14.6 | 6.9 (2.1×) | 6.9 (2.1×) | 206 | 90.5 (2.3×) | 91.3 (2.3×) | 1.9× | 1.9× |
| `t-hs-d20-wn3.bin` | 8.1 | 6.2 (1.3×) | 5.5 (1.5×) | 84.3 | 65.4 (1.3×) | 58.5 (1.4×) | 2.0× | 2.1× |
| `t-hs-d20-wn3.ftz` | 11.8 | 9.2 (1.3×) | 9.7 (1.2×) | 138 | 112 (1.2×) | 107 (1.3×) | 2.0× | 2.0× |
| `t-ns-d16.bin` | 13.5 | 8.4 (1.6×) | 6.6 (2.0×) | 69.8 | 52.7 (1.3×) | 48.4 (1.4×) | 1.7× | 1.9× |
| `t-ova-d64.bin` | 35.6 | 18.9 (1.9×) | 10.8 (3.3×) | 105 | 83.2 (1.3×) | 62.7 (1.7×) | 1.9× | 1.9× |
| `t-ova-d64.ftz` | 80.0 | 68.3 (1.2×) | 67.5 (1.2×) | 277 | 253 (1.1×) | 258 (1.1×) | 2.0× | 1.9× |
| `t-softmax-d10-wn2.bin` | 15.2 | 10.0 (1.5×) | 9.5 (1.6×) | 53.1 | 25.8 (2.1×) | 26.3 (2.0×) | 1.9× | 1.8× |
| `t-softmax-d256.bin` | 164 | 79.0 (2.1×) | 42.1 (3.9×) | 490 | 487 (1.0×) | 301 (1.6×) | 1.8× | 2.0× |
| `t-softmax-d32.bin` | 26.6 | 19.8 (1.3×) | 15.0 (1.8×) | 125 | 132 (0.9×) | 110 (1.1×) | 1.5× | 2.0× |
| `t-softmax-d32.ftz` | 38.8 | 25.0 (1.6×) | 23.1 (1.7×) | 313 | 194 (1.6×) | 237 (1.3×) | 1.8× | 1.8× |

### Arm Neoverse N3 (Linux arm64), 4 vCPUs

| Model | fastText, sentences | Go | Go simd | fastText, paragraphs | Go | Go simd | Go, 4 threads | fastText, 4 threads |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lid.176.bin` | 4.4 | 2.8 (1.5×) | 2.7 (1.6×) | 63.2 | 39.2 (1.6×) | 34.2 (1.8×) | 4.1× | 4.4× |
| `lid.176.ftz` | 6.1 | 2.7 (2.2×) | 2.8 (2.2×) | 94.0 | 37.7 (2.5×) | 38.0 (2.5×) | 4.1× | 4.3× |
| `t-hs-d20-wn3.bin` | 3.4 | 2.4 (1.4×) | 2.4 (1.4×) | 43.2 | 30.4 (1.4×) | 28.9 (1.5×) | 4.3× | 4.6× |
| `t-hs-d20-wn3.ftz` | 4.8 | 3.4 (1.4×) | 3.5 (1.4×) | 65.4 | 41.1 (1.6×) | 43.1 (1.5×) | 4.0× | 4.1× |
| `t-ns-d16.bin` | 5.9 | 4.0 (1.5×) | 2.9 (2.0×) | 41.0 | 24.1 (1.7×) | 18.3 (2.2×) | 4.2× | 3.9× |
| `t-ova-d64.bin` | 13.8 | 9.8 (1.4×) | 5.0 (2.8×) | 50.5 | 42.9 (1.2×) | 26.2 (1.9×) | 4.0× | 4.0× |
| `t-ova-d64.ftz` | 31.5 | 26.4 (1.2×) | 26.4 (1.2×) | 104 | 107 (1.0×) | 107 (1.0×) | 4.0× | 4.0× |
| `t-softmax-d10-wn2.bin` | 5.9 | 4.0 (1.5×) | 3.4 (1.7×) | 18.6 | 9.2 (2.0×) | 8.3 (2.2×) | 4.0× | 4.0× |
| `t-softmax-d256.bin` | 68.1 | 43.0 (1.6×) | 24.5 (2.8×) | 291 | 245 (1.2×) | 159 (1.8×) | 4.1× | 4.1× |
| `t-softmax-d32.bin` | 11.6 | 8.5 (1.4×) | 5.9 (1.9×) | 69.9 | 52.5 (1.3×) | 42.3 (1.7×) | 4.1× | 4.2× |
| `t-softmax-d32.ftz` | 16.4 | 10.9 (1.5×) | 8.9 (1.8×) | 146 | 80.1 (1.8×) | 77.9 (1.9×) | 4.0× | 4.0× |
| GlotLID | 131 | 96.4 (1.4×) | 52.0 (2.5×) | 362 | 307 (1.2×) | 201 (1.8×) | 4.1× | 4.1× |
| OpenLID-v2 | 57.7 | 35.0 (1.6×) | 21.9 (2.6×) | 485 | 360 (1.3×) | 257 (1.9×) | 4.2× | 4.2× |

## What the tables show

- **`lid.176`.** Plain Go is 1.1 to 2.5 times faster than fastText on every machine, for both files and both text lengths. SIMD adds little, because this model's vectors have 16 numbers.
- **GlotLID and OpenLID-v2.** With SIMD the port is 2.5 to 3.2 times faster on sentences, and 1.2 to 2.1 times faster on paragraphs. Plain Go is 1.0 to 1.6 times faster on sentences.
- **Where plain Go is slower than fastText:** paragraphs with 256-number vectors on the Xeon and on the Mac (0.5 to 0.9).
  - Most of the time there goes into adding up rows of the input matrix. fastText's build does that with vector instructions, and plain Go adds one number at a time.
  - fastText's strict build, compiled without `-march=native`, takes about as long as plain Go (302 µs against 287 µs for OpenLID-v2 paragraphs on the Xeon).
  - The SIMD build is faster than fastText there too.
- **Threads.** Both programs scale with the number of CPUs.

## How far to trust the numbers

- **One run of 3 seconds per number,** on cloud machines shared with other workloads. fastText's own times moved by up to 13% between two rounds, so differences below about 15% mean nothing. The OpenLID-v2 paragraph row on the Xeon is the least reliable.
- **Every run returned the same number of predictions** in all programs, so they did the same work.

## Reproducing

C++ fastText against the Go port, in the reference environment:

```sh
make testdata
docker build -t fasttext-reference accuracy/reference
docker run --rm -v "$PWD:/src" -w /src fasttext-reference accuracy/bench.sh 3
```

On a machine with gcc and Go, run `accuracy/bench.sh 3` directly. The [Accuracy workflow](../.github/workflows/accuracy.yml) runs it on GitHub's x86-64 and arm64 runners and writes the table to the job summary.

Go benchmarks only:

```sh
make bench                                    # every Benchmark* in both modules
go test -run '^$' -bench 'Predict$' -count 6 . | benchstat -   # with golang.org/x/perf/cmd/benchstat
```
