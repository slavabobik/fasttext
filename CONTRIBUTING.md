# Contributing

Thanks for helping. Bug reports, questions and pull requests are welcome.

## Reporting a difference from fastText

The package promises the same bits as fastText's C++ code. If you see a different label, ranking or probability, open an issue with:

- the model (a link, or its name and SHA-256);
- the exact text, `k` and threshold;
- what fastText returned, and which build of fastText that was (Python package version, OS, CPU).

Builds of fastText differ from each other in the last bits (see [docs/VERIFICATION.md](docs/VERIFICATION.md)). The reference here is a strict Linux build, so a difference from a Mac build of fastText may be expected.

## Development setup

You need Go 1.24 or later (Go 1.27+ for the SIMD build) and [Git LFS](https://git-lfs.com) for the large test fixtures.

```sh
git clone https://github.com/slavabobik/fasttext && cd fasttext
git lfs install && git lfs pull   # the t-*.bin test models and the Tatoeba corpora
make testdata                     # assembles lid.176.bin from lid176/model
```

Without Git LFS everything still builds, and the tests that need the large files are skipped.

The repository holds two modules: the root module (`github.com/slavabobik/fasttext`) and `lid176/`, which embeds the model. `make` creates a `go.work` so that both build against your working copy. Don't commit it.

## Common tasks

| Command | What it does |
| --- | --- |
| `make test` | Quick tests of both modules (`-short`) |
| `make test-simd` | The same with `GOEXPERIMENT=simd` |
| `make golden` | Every test with every fixture: golden dumps of 11 models, the exhaustive libm check, plain and SIMD |
| `make race` | Tests under the race detector |
| `make lint` | `go vet`, gofmt, golangci-lint, govulncheck |
| `make nocgo` | Cross-compiles for 17 platforms with `CGO_ENABLED=0` |
| `make bench` | Go benchmarks |
| `make accuracy-docker` | Builds C++ fastText in Docker and compares it with the Go port on 233,381 texts per model |

## Rules for changes to prediction code

1. **Results must stay bit-identical.** `make golden` must pass with plain Go and with `GOEXPERIMENT=simd`. A change of operation order, a fused multiply-add, or a different `exp` changes bits, even when it looks harmless.
2. **Speed changes need numbers.** Run the benchmarks before and after, with [benchstat](https://pkg.go.dev/golang.org/x/perf/cmd/benchstat):
   ```sh
   go test -run '^$' -bench 'Predict' -count 10 . > old.txt   # on main
   go test -run '^$' -bench 'Predict' -count 10 . > new.txt   # on your branch
   benchstat old.txt new.txt
   ```
3. **No cgo, no assembly files, no dependencies.** The SIMD kernels use `simd/archsimd` behind `//go:build goexperiment.simd`.
4. Run `go vet`, `gofmt` and the tests. Document exported identifiers, and add a line to `CHANGELOG.md` for any change users will notice.

## Regenerating reference data

The golden files in `accuracy/golden` are written by C++ fastText, not by this package. Regenerate them only on purpose, with the reference build, as described in [accuracy/README.md](accuracy/README.md). Never regenerate them from the Go port.

## Releasing

1. Update `CHANGELOG.md`, then tag the root module: `git tag vX.Y.Z`.
2. If `lid176` changes, point its `go.mod` at the new root version, then tag it: `git tag lid176/vX.Y.Z`.
3. Push the tags: `git push origin --tags`.
