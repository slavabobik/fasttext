# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). The two modules are versioned
separately: tags `vX.Y.Z` for `github.com/slavabobik/fasttext`, and
`lid176/vX.Y.Z` for `github.com/slavabobik/fasttext/lid176`.

## [Unreleased]

## [0.1.0] - 2026-11-24

First release.

### fasttext

- `Open`, `Load`: read fastText `.bin` and `.ftz` models (format version 12),
  with every loss: softmax, hierarchical softmax, one-vs-all and negative
  sampling.
- `Predict`, `AppendPredict`: the k best labels and probabilities, bit for bit
  those of fastText's C++ code. `AppendPredict` does not allocate.
- `Info`, `Labels`, `Dim`, `Trace`, `Kernel`.
- Optional SIMD kernels (NEON, AVX2, AVX-512) with `GOEXPERIMENT=simd` on Go
  1.27 or later, with identical results.

### langid

- `Detector` with `Detect`, `DetectTop`, `AppendDetect`, `Languages`: language
  codes for any fastText language-identification model. Newlines read as
  spaces; text without words gives `ErrNoText`.

### lid176 (lid176/v0.1.0)

- The `lid.176.bin` model embedded with `go:embed`: `Detect`, `Detector`,
  `Model`, `Reader`.
- `cmd/lid`: command-line language detection.

[Unreleased]: https://github.com/slavabobik/fasttext/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/slavabobik/fasttext/releases/tag/v0.1.0
