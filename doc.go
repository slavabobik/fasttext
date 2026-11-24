// Package fasttext predicts labels with fastText classification models, such
// as the language identification models lid.176.bin and lid.176.ftz, in pure
// Go: no cgo, no C++ toolchain, no dependencies.
//
// It reads the model files fastText writes (.bin, and .ftz with product
// quantization) and returns what fastText returns: the same labels in the
// same order with the same probabilities, bit for bit. Only prediction is
// implemented; there is no training and no access to word vectors.
//
//	m, err := fasttext.Open("lid.176.ftz")
//	if err != nil { ... }
//	preds, err := m.Predict("Bonjour tout le monde", 1, 0)
//	// preds[0].Label == "__label__fr"
//
// For language identification, package
// github.com/slavabobik/fasttext/langid returns language codes instead of
// labels, and module github.com/slavabobik/fasttext/lid176 embeds lid.176.bin
// so that no model file is needed.
//
// A Model is read-only after loading and may be used from any number of
// goroutines. AppendPredict does not allocate.
//
// # No cgo
//
// The package is written in Go only. It builds with CGO_ENABLED=0 for every
// platform Go supports, including cross-compilation, static binaries for
// scratch or distroless containers, and WebAssembly. It carries its own ports
// of the parts of the C and C++ libraries that fastText's results depend on,
// so it needs neither.
//
// # What "the same" means
//
// The reference is fastText at its last commit (1142dc4), built for Linux with
// gcc and glibc and without flags that let the compiler reorder floating-point
// arithmetic; docs/VERIFICATION.md in the repository describes the build and
// how other builds of fastText differ from it. Predict behaves like predict in
// fastText's Python package:
//
//   - The text is one line. A text with a newline is an error (ErrNewline),
//     as in Python.
//   - k is the number of labels wanted, -1 for all. Labels with a
//     probability below threshold are left out.
//   - Labels keep their prefix ("__label__en").
//   - Probabilities are float32 and can exceed 1 by about 1e-5, because
//     fastText adds 1e-5 before taking the logarithm.
//   - With hierarchical softmax (lid.176), labels with a probability below
//     about 1e-5 are never returned, whatever k and threshold are.
//   - Labels with exactly equal probabilities come out in the order of
//     fastText on Linux, which is an accident of the C++ library's heap
//     functions. The package repeats them move by move.
//
// To get there the package carries its own ports of the three C library
// functions fastText calls (expf, exp and log of glibc, which come from ARM
// Optimized Routines; see LICENSE-arm-optimized-routines) and rounds every
// floating-point operation separately. The same holds on every platform Go
// supports: results do not depend on the processor, the operating system or
// the build.
//
// Where fastText's own behaviour is undefined, the package returns an error
// instead: a model whose weights produce a NaN score gives ErrNaN (fastText
// throws for some models and reads outside a table for others), and a
// damaged model file gives an error from Open.
//
// # Speed
//
// The inner loops are plain Go arranged so that the processor can work on
// several sums at once without changing any of them. Built with
// GOEXPERIMENT=simd (Go 1.27 or later) the package uses vector instructions on
// amd64 (AVX2, AVX-512) and arm64 (NEON) for the same loops; results are
// identical and the tests check that. Kernel reports which is in use.
//
// # Verification
//
// The tests compare every layer of a prediction (input rows, hidden vector,
// scores, probabilities) with dumps written by the reference build, for
// eleven models covering every loss and file format, and check the maths
// functions against glibc on every float32 value. cmd/ftdump writes the same
// dumps for any model and text, and accuracy/verify.sh compares them with a
// fresh build of fastText on 233,381 texts per model.
package fasttext
