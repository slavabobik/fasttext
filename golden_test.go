package fasttext_test

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/slavabobik/fasttext"
	"github.com/slavabobik/fasttext/internal/refdump"
	"github.com/slavabobik/fasttext/internal/testdata"
)

// The golden files in accuracy/golden were written by the reference build of
// fastText (Linux x86-64, see accuracy/golden/BUILD) with accuracy/golden.sh.
// Each holds, for one model and one text file, the input rows, the hidden
// vector and the predictions of every record, floats as raw bits. The port
// has to reproduce them byte for byte.

func accuracyDir() string { return testdata.Dir() }

var testModels = []string{
	"lid.176.bin", "lid.176.ftz",
	"t-softmax-d32.bin", "t-softmax-d32.ftz", "t-softmax-d256.bin", "t-softmax-d10-wn2.bin",
	"t-hs-d20-wn3.bin", "t-hs-d20-wn3.ftz",
	"t-ova-d64.bin", "t-ova-d64.ftz",
	"t-ns-d16.bin",
}

// openModel loads a test model, or skips the test if the file is not there
// (the .bin test models are stored in Git LFS).
func openModel(tb testing.TB, name string) *fasttext.Model {
	tb.Helper()
	path := testdata.Model(tb, name)
	m, err := fasttext.Open(path)
	if err != nil {
		tb.Fatal(err)
	}
	return m
}

func TestGolden(t *testing.T) {
	kinds := []struct {
		name, corpus string
		fullIDs      bool
	}{
		{"edge-cases", "edge-cases.txt", true},
		{"random-golden", "random-golden.txt", true},
		{"tatoeba-golden", "tatoeba-golden.txt", false},
		{"threshold", "tatoeba-golden.txt", false},
	}
	// Every implementation of the inner loops this build can run has to
	// reproduce the files: plain Go and, with GOEXPERIMENT=simd, the vector
	// kernels.
	for _, kernel := range fasttext.KernelNames() {
		t.Run(kernel, func(t *testing.T) {
			defer fasttext.UseKernel(kernel)()
			for _, name := range testModels {
				t.Run(name, func(t *testing.T) {
					m := openModel(t, name)
					for _, kind := range kinds {
						t.Run(kind.name, func(t *testing.T) {
							want := readGzip(t, filepath.Join(accuracyDir(), "golden", name+"."+kind.name+".dump.gz"))
							ks, threshold := runSettings(t, want)
							text, err := os.ReadFile(filepath.Join(accuracyDir(), "corpus", kind.corpus))
							if err != nil {
								t.Fatal(err)
							}
							var got bytes.Buffer
							if _, err := refdump.Write(&got, m, bytes.NewReader(text), ks, threshold, kind.fullIDs); err != nil {
								t.Fatal(err)
							}
							compareDumps(t, got.Bytes(), want)
						})
					}
				})
			}
		})
	}
}

func readGzip(t *testing.T, path string) []byte {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(z)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// runSettings reads the k values and the threshold from the "#run" line of a
// dump, so the test repeats exactly the run that wrote the golden file.
func runSettings(t *testing.T, dump []byte) (ks []int, threshold float32) {
	t.Helper()
	sc := bufio.NewScanner(bytes.NewReader(dump))
	sc.Buffer(nil, 1<<26)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "#run ") {
			continue
		}
		for _, f := range strings.Fields(line)[1:] {
			key, val, _ := strings.Cut(f, "=")
			switch key {
			case "threshold":
				v, err := strconv.ParseFloat(val, 32)
				if err != nil {
					t.Fatal(err)
				}
				threshold = float32(v)
			case "k":
				for _, s := range strings.Split(val, ",") {
					k, err := strconv.Atoi(s)
					if err != nil {
						t.Fatal(err)
					}
					ks = append(ks, k)
				}
			}
		}
		return ks, threshold
	}
	t.Fatal("golden file has no #run line")
	return nil, 0
}

func compareDumps(t *testing.T, got, want []byte) {
	t.Helper()
	if bytes.Equal(got, want) {
		return
	}
	g := bytes.Split(got, []byte("\n"))
	w := bytes.Split(want, []byte("\n"))
	if len(g) != len(w) {
		t.Errorf("%d lines, want %d", len(g), len(w))
	}
	shown := 0
	for i := 0; i < len(g) && i < len(w) && shown < 3; i++ {
		if bytes.Equal(g[i], w[i]) {
			continue
		}
		gf := bytes.Split(g[i], []byte("\t"))
		wf := bytes.Split(w[i], []byte("\t"))
		for j := 0; j < len(gf) && j < len(wf); j++ {
			if !bytes.Equal(gf[j], wf[j]) {
				t.Errorf("line %d, field %d:\n got %.200s\nwant %.200s", i+1, j+1, gf[j], wf[j])
				break
			}
		}
		shown++
	}
}
