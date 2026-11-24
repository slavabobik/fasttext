# fasttext

**fastText inference and language identification in pure Go. No cgo. Bit-identical to C++ fastText.**

[![CI](https://github.com/slavabobik/fasttext/actions/workflows/ci.yml/badge.svg)](https://github.com/slavabobik/fasttext/actions/workflows/ci.yml)
[![Accuracy vs C++](https://github.com/slavabobik/fasttext/actions/workflows/accuracy.yml/badge.svg)](https://github.com/slavabobik/fasttext/actions/workflows/accuracy.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/slavabobik/fasttext.svg)](https://pkg.go.dev/github.com/slavabobik/fasttext)
![Go version](https://img.shields.io/badge/go-%E2%89%A51.24-00ADD8?logo=go)
![cgo](https://img.shields.io/badge/cgo-not%20required-success)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

The package reads the model files that [fastText](https://fasttext.cc) writes (`.bin` and quantized `.ftz`) and predicts their labels. It works with any fastText classifier and was built for language identification: `lid.176`, GlotLID and OpenLID. It returns what fastText returns: the same labels, in the same order, with the same probabilities, **bit for bit**.

```go
import "github.com/slavabobik/fasttext/lid176"

lang, err := lid176.Detect("Où est ma commande ? Elle devait arriver hier.")
// lang.Code == "fr", lang.Prob == 0.99…
```

The model is built in, so there is nothing to download, nothing to install, and no C compiler involved.

## Why

- **Pure Go, no cgo.**
  - `CGO_ENABLED=0` works everywhere, and cross-compiling is one `GOOS=… GOARCH=…` away.
  - Binaries are static and run in `scratch` or distroless images.
  - No C++ toolchain is needed in CI or Docker builds.
  - It builds for WebAssembly too.
  - CI cross-compiles it for 17 platforms and runs the tests on big-endian s390x.
- **Exact.**
  - Tested layer by layer against a reference build of fastText's C++ code: input rows, hidden vector, scores and probabilities, compared as raw float bits.
  - The tests cover every loss (softmax, hierarchical softmax, one-vs-all, negative sampling), dense and quantized models, and 233,381 texts per model.
  - Results don't depend on CPU, OS or build flags. See [docs/VERIFICATION.md](docs/VERIFICATION.md).
- **Fast.**
  - 1.2 to 2.5× faster than fastText's C++ library on `lid.176`, and up to 3.9× faster with the optional SIMD build ([benchmarks](#performance)).
  - No allocation per prediction.
- **Safe for concurrent use.** One model can be shared by any number of goroutines.
- **No dependencies** beyond the Go standard library.

## Install

```sh
go get github.com/slavabobik/fasttext/lid176   # language ID with the embedded lid.176.bin
go get github.com/slavabobik/fasttext          # the library only, bring your own model
```

The command-line tool:

```sh
go install github.com/slavabobik/fasttext/lid176/cmd/lid@latest
```

Go 1.24 or later is required.

> **Repository is private?** Then tell Go not to use the public proxy for it, and make sure `git` can authenticate to GitHub:
> `go env -w GOPRIVATE=github.com/slavabobik/*`

## Quick start

### Detect a language with the embedded model

```go
package main

import (
	"fmt"
	"log"

	"github.com/slavabobik/fasttext/lid176"
)

func main() {
	d, err := lid176.Detector() // loads lid.176.bin once (10–100 ms); safe to share
	if err != nil {
		log.Fatal(err)
	}

	lang, err := d.Detect("Wo ist meine Bestellung? Sie sollte gestern ankommen.")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(lang.Code, lang.Prob) // de 0.9999

	top, _ := d.DetectTop("Bonjour tout le monde", 3, 0) // the 3 best, any probability
	for _, l := range top {
		fmt.Printf("%s %.4f\n", l.Code, l.Prob)
	}
}
```

`lid176` adds about 131 MB (the model) to your binary. If size matters more than the last bit of accuracy, use the 917 KB `lid.176.ftz` from a file instead (see below).

### Use another model

Any fastText language-identification model works through package `langid`:

```go
import "github.com/slavabobik/fasttext/langid"

d, err := langid.Open("lid.176.ftz") // or GlotLID's model.bin, OpenLID's model.bin, …
lang, err := d.Detect("Где мой заказ?")
// lang.Code == "ru"
```

| Model | Languages | Size | Codes | Download |
| --- | ---: | ---: | --- | --- |
| `lid.176.bin` (embedded in `lid176`) | 176 | 131 MB | `en`, `zh`, `als` | [fasttext.cc](https://fasttext.cc/docs/en/language-identification.html) |
| `lid.176.ftz` | 176 | 917 KB | `en`, `zh`, `als` | [fasttext.cc](https://fasttext.cc/docs/en/language-identification.html) |
| GlotLID | 2,060 | 783 MB | `eng_Latn` | [cis-lmu/glotlid](https://huggingface.co/cis-lmu/glotlid) |
| OpenLID-v2 | 200 | 1.2 GB | `eng_Latn` | [laurievb/OpenLID-v2](https://huggingface.co/laurievb/OpenLID-v2) |

### Any fastText classifier

The core package works with any supervised fastText model and returns labels exactly as fastText does:

```go
import "github.com/slavabobik/fasttext"

m, err := fasttext.Open("model.bin")
preds, err := m.Predict("some text", 5, 0.1) // k best labels with probability ≥ 0.1
for _, p := range preds {
	fmt.Println(p.Label, p.Prob) // "__label__positive" 0.93
}
```

### Command line

```console
$ echo "Bonjour tout le monde" | lid -k 2
fr	0.9946	en	0.0009
$ lid -json Hola que tal
{"text":"Hola que tal","languages":[{"code":"es","prob":0.9247749}]}
$ lid -model lid.176.ftz < messages.txt     # one result line per input line
```

### Static binary in an empty container

```dockerfile
FROM golang:1.27 AS build
RUN CGO_ENABLED=0 go install github.com/slavabobik/fasttext/lid176/cmd/lid@latest

FROM scratch
COPY --from=build /go/bin/lid /lid
ENTRYPOINT ["/lid"]
```

## API

| Package | Call | Does |
| --- | --- | --- |
| `lid176` | `Detect(text)` | Most likely language with the shared embedded model |
| | `Detector()` | The shared `*langid.Detector` (loaded once) |
| | `Model()`, `Reader()` | A fresh `*fasttext.Model`, or the raw model bytes |
| `langid` | `Open(path)`, `Load(r)`, `New(model)` | A detector for any fastText LID model |
| | `(*Detector).Detect(text)` | Most likely `Language{Code, Prob}` |
| | `(*Detector).DetectTop(text, k, threshold)` | Up to `k` languages (`-1` for all) at or above `threshold` |
| | `(*Detector).AppendDetect(dst, text, k, threshold)` | The same, appending to `dst`; no allocation |
| | `(*Detector).Languages()` | Every code the model knows |
| `fasttext` | `Open(path)`, `Load(r)` | Read a `.bin` or `.ftz` model |
| | `(*Model).Predict(text, k, threshold)` | The `k` best labels with probabilities, as fastText's Python `predict` |
| | `(*Model).AppendPredict(dst, …)` | The same, appending to `dst`; no allocation |
| | `(*Model).Labels()`, `Dim()`, `Info()` | Labels and the settings stored in the file |
| | `(*Model).Trace(…)` | Every intermediate value of a prediction, for comparison with fastText |
| | `Kernel()` | Which inner-loop implementation is in use (`go`, `neon`, `avx2`, `avx512`) |

Full documentation: [pkg.go.dev/github.com/slavabobik/fasttext](https://pkg.go.dev/github.com/slavabobik/fasttext).

## Accuracy: identical to C++ fastText

The goal is not "close to fastText" but **the same bits**. A port that is only close gives other probabilities, and sometimes another ranking of near-equal labels. Every check below is in this repository and runs in CI.

| Check | Compared with | Result |
| --- | --- | --- |
| Input rows, hidden vector, scores, probabilities of 3,168 texts × 11 models (`go test`) | Dumps written by C++ fastText ([`accuracy/golden`](accuracy/golden)) | Identical |
| Full corpora: 233,381 texts × 11 models, k = 1, 5, all ([`accuracy/verify.sh`](accuracy/verify.sh)) | A fresh build of fastText in Docker | 33 of 33 dumps identical, plain Go and SIMD |
| GlotLID and OpenLID-v2 on the same corpora | C++ fastText | Identical |
| `expf`, `exp`, `log` on all 2³² float inputs | glibc 2.41 | Identical |
| Order of labels with tied probabilities, 350 cases | libstdc++ heap routines | Identical |
| Platforms | macOS arm64, Linux arm64, Linux x86-64 (AVX2, AVX-512), with and without SIMD | Identical everywhere |

How it gets there:

- **Every floating-point operation is rounded separately,** in fastText's order. Go never fuses a multiply and an add unless asked to.
- **It carries its own ports of the maths functions fastText calls** (`expf`, `exp` and `log` of glibc, which come from ARM Optimized Routines).
- **It ports the C++ library's heap routines,** which decide the order of labels with equal probability.

So the results don't depend on the CPU, the OS or the build. [docs/VERIFICATION.md](docs/VERIFICATION.md) has the details. It includes a finding that matters if you compare against your own fastText build: fastText's own builds differ from each other (a Mac build against a Linux build, `-march=native` on quantized models). The reference here is fastText's last commit built with strict floating-point flags on Linux. For `lid.176.bin` this is also what fastText's release flags give.

Reproduce it yourself:

```sh
git lfs install && git lfs pull && make testdata
make golden           # all Go tests with every fixture
make accuracy-docker  # C++ fastText vs Go on the full corpora (Docker)
```

## Performance

Microseconds per text, one thread, k = 1, sentences of 41 bytes on average. fastText's C++ library is built with its own release flags (`-O3 -march=native`). Brackets: how many times faster than fastText.

| Model | Machine | fastText (C++) | Go | Go + SIMD |
| --- | --- | ---: | ---: | ---: |
| `lid.176.bin` | Intel Xeon Platinum 8581C | 3.2 | 2.6 (1.3×) | 2.2 (1.4×) |
| | Arm Neoverse N3 | 4.4 | 2.8 (1.5×) | 2.7 (1.6×) |
| | Apple M4 Pro | 2.3 | 1.2 (2.0×) | 1.0 (2.2×) |
| `lid.176.ftz` | Intel Xeon Platinum 8581C | 5.8 | 3.4 (1.7×) | 3.4 (1.7×) |
| | Apple M4 Pro | 3.8 | 1.7 (2.3×) | 1.7 (2.2×) |
| GlotLID | Intel Xeon Platinum 8581C | 105 | 90.9 (1.2×) | 32.6 (3.2×) |
| OpenLID-v2 | Intel Xeon Platinum 8581C | 34.6 | 33.4 (1.0×) | 11.8 (2.9×) |

- **Allocation.** `langid.Detect` with `lid.176.bin` on an Apple M4 Pro takes about 1.1 µs, with 0 allocations.
- **Loading.** Loading the embedded model takes 8 ms on an M4 Pro and up to about 100 ms on slower servers, and as much memory as the model.
- **Concurrency.** Throughput scales with the number of cores.

**SIMD build (optional).** With Go 1.27 or later, build with `GOEXPERIMENT=simd` to use NEON on arm64 and AVX2 or AVX-512 on amd64. It needs no assembly and no cgo, and the results are the same bits. It matters for models with large vectors (GlotLID, OpenLID); for `lid.176` plain Go is already as fast.

All machines, models and paragraph-length texts: [docs/BENCHMARKS.md](docs/BENCHMARKS.md). To measure on your own machine: `make bench` (Go only), or `accuracy/bench.sh` (C++ against Go).

## Behaviour notes

These follow fastText. They matter mainly if you compare numbers with it:

- **Probabilities can exceed 1 by about 1e-5,** because fastText adds 1e-5 before taking the logarithm.
- **Hierarchical softmax models** (such as `lid.176`) never return labels with a probability below about 1e-5, whatever `k` is.
- **Newlines.** `fasttext.Model.Predict` rejects text with `\n` (`ErrNewline`), as fastText's Python package does. `langid` replaces newlines with spaces instead.
- **Empty text.** fastText still returns its most frequent label for an empty text, with a low probability. `langid` returns `ErrNoText` instead.
- **Prefixes.** Labels keep the `__label__` prefix in package `fasttext`; `langid` strips it.
- **Damaged models.** A truncated or damaged model file gives an error from `Open`, not a crash. A model whose weights produce NaN gives `ErrNaN`.

## Supported platforms

Every platform Go supports.

- **Tested in CI:** Linux amd64 and arm64, macOS arm64, Windows amd64, Linux s390x (big-endian, under QEMU).
- **Cross-compiled in CI:** the 17 platforms listed in the [Makefile](Makefile), including `js/wasm` and `wasip1/wasm`.
- **SIMD build:** amd64 (AVX2, AVX-512) and arm64 (NEON), with Go 1.27+. Every other platform uses the plain-Go loops.

## Papers and references

fastText and the models this package runs are described in:

1. Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov. [**Bag of Tricks for Efficient Text Classification**](https://arxiv.org/abs/1607.01759). EACL 2017. The fastText classifier.
2. Armand Joulin, Edouard Grave, Piotr Bojanowski, Matthijs Douze, Hérve Jégou, Tomas Mikolov. [**FastText.zip: Compressing text classification models**](https://arxiv.org/abs/1612.03651). arXiv:1612.03651, 2016. Product quantization and pruning, the `.ftz` format.
3. Piotr Bojanowski, Edouard Grave, Armand Joulin, Tomas Mikolov. [**Enriching Word Vectors with Subword Information**](https://arxiv.org/abs/1607.04606). TACL 2017. Character n-grams.
4. Amir Hossein Kargaran, Ayyoob Imani, François Yvon, Hinrich Schütze. [**GlotLID: Language Identification for Low-Resource Languages**](https://arxiv.org/abs/2310.16248). Findings of EMNLP 2023.
5. Laurie Burchell, Alexandra Birch, Nikolay Bogoychev, Kenneth Heafield. [**An Open Dataset and Model for Language Identification**](https://arxiv.org/abs/2305.13820). ACL 2023. OpenLID.

Also useful:

- The [fastText language-identification page](https://fasttext.cc/docs/en/language-identification.html), with the `lid.176` models and their language list.
- The [fastText repository](https://github.com/facebookresearch/fastText), archived in 2024. This port follows its last commit, `1142dc4`.

If you use `lid.176` in research, cite [1] and [2], as the fastText authors ask:

```bibtex
@inproceedings{joulin2017bag,
  title     = {Bag of Tricks for Efficient Text Classification},
  author    = {Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  booktitle = {Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers},
  pages     = {427--431},
  year      = {2017}
}

@article{joulin2016fasttext,
  title   = {FastText.zip: Compressing text classification models},
  author  = {Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Douze, Matthijs and J{\'e}gou, H{\'e}rve and Mikolov, Tomas},
  journal = {arXiv preprint arXiv:1612.03651},
  year    = {2016}
}
```

To cite this package, see [CITATION.cff](CITATION.cff).

## Project layout

| Path | Contents |
| --- | --- |
| [`/`](.) | Package `fasttext`: model reader, prediction, maths ports, kernels |
| [`langid/`](langid) | Package `langid`: language codes, newline handling, `ErrNoText` |
| [`lid176/`](lid176) | Separate module with the embedded `lid.176.bin` and the `lid` CLI, so the core module stays small |
| [`cmd/ftdump`](cmd/ftdump), [`cmd/ftbench`](cmd/ftbench) | Dump and timing tools used for the comparison with C++ |
| [`accuracy/`](accuracy) | The reference harness: C++ tools, Dockerfile, models, corpora, golden dumps, results |
| [`docs/`](docs) | [Verification](docs/VERIFICATION.md) and [benchmarks](docs/BENCHMARKS.md) |

## Contributing

Bug reports and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). The one hard rule: a change to the prediction code must keep the golden tests passing. Every result must stay bit-identical to fastText.

## License

- **The code** is MIT-licensed; see [LICENSE](LICENSE).
- **Ported parts** keep their own MIT licences: fastText (© Facebook, Inc.), see [LICENSE-fastText](LICENSE-fastText); and ARM Optimized Routines, see [LICENSE-arm-optimized-routines](LICENSE-arm-optimized-routines).
- **The embedded `lid.176.bin` model** is © Facebook, Inc., under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/); see [lid176/LICENSE-lid176](lid176/LICENSE-lid176).
- **Test corpora** are drawn from [Tatoeba](https://tatoeba.org), under CC BY 2.0 FR.

[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists all of it.
