package fasttext_test

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/slavabobik/fasttext"
	"github.com/slavabobik/fasttext/internal/testdata"
)

func TestPredict(t *testing.T) {
	m := openModel(t, "lid.176.ftz")

	// Labels and probabilities as printed by fastText's own command line
	// tool (predict-prob, six significant digits) for the same model.
	for _, c := range []struct {
		text string
		want string
	}{
		{"Bonjour tout le monde, comment allez-vous ?", "__label__fr 0.986674 __label__en 0.00388813 __label__it 0.00149437"},
		{"The quick brown fox jumps over the lazy dog.", "__label__en 0.697451 __label__es 0.046211 __label__pt 0.0252001"},
		{"Это предложение написано по-русски.", "__label__ru 0.985661 __label__bg 0.00491949 __label__ba 0.00190375"},
		{"", "__label__en 0.124504 __label__ca 0.0859483 __label__de 0.0802881"},
	} {
		preds, err := m.Predict(c.text, 3, 0)
		if err != nil {
			t.Fatalf("Predict(%q): %v", c.text, err)
		}
		var got []string
		for _, p := range preds {
			got = append(got, fmt.Sprintf("%s %.6g", p.Label, p.Prob))
		}
		if s := strings.Join(got, " "); s != c.want {
			t.Errorf("Predict(%q) = %s, want %s", c.text, s, c.want)
		}
	}

	preds, err := m.Predict("Bonjour tout le monde, comment allez-vous ?", 2, 0.5)
	if err != nil || len(preds) != 1 || preds[0].Label != "__label__fr" {
		t.Errorf("with threshold 0.5: %v, %v; want only __label__fr", preds, err)
	}

	// With k = -1 a softmax model returns every label. A hierarchical
	// softmax model, like fastText, leaves out labels below about 1e-5.
	all, err := m.Predict("Bonjour", -1, 0)
	if err != nil || len(all) == 0 || len(all) > len(m.Labels()) {
		t.Errorf("k = -1: %d predictions, %v", len(all), err)
	}
	for i := 1; i < len(all); i++ {
		if all[i].Prob > all[i-1].Prob {
			t.Fatalf("k = -1: predictions not ordered at %d", i)
		}
	}
	sm := openModel(t, "t-softmax-d32.ftz")
	all, err = sm.Predict("Bonjour", -1, 0)
	if err != nil || len(all) != len(sm.Labels()) || len(all) != 323 {
		t.Errorf("softmax, k = -1: %d predictions, %v; want %d", len(all), err, len(sm.Labels()))
	}

	if _, err := m.Predict("two\nlines", 1, 0); !errors.Is(err, fasttext.ErrNewline) {
		t.Errorf("newline: err = %v, want ErrNewline", err)
	}
	for _, k := range []int{0, -2} {
		if _, err := m.Predict("text", k, 0); !errors.Is(err, fasttext.ErrK) {
			t.Errorf("k = %d: err = %v, want ErrK", k, err)
		}
	}

	if got := len(m.Labels()); got != 176 {
		t.Errorf("%d labels, want 176", got)
	}
	info := m.Info()
	if info.Dim != 16 || info.Loss != "hs" || !info.Quantized || !info.Pruned || info.Labels != 176 {
		t.Errorf("Info() = %+v", info)
	}
	if m.Dim() != 16 {
		t.Errorf("Dim() = %d", m.Dim())
	}
}

func TestAppendPredict(t *testing.T) {
	m := openModel(t, "lid.176.ftz")
	dst := make([]fasttext.Prediction, 1, 8)
	dst[0] = fasttext.Prediction{Label: "kept"}
	dst, err := m.AppendPredict(dst, "Guten Morgen, wie geht es Ihnen?", 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(dst) != 3 || dst[0].Label != "kept" || dst[1].Label != "__label__de" {
		t.Errorf("AppendPredict = %v", dst)
	}
}

// TestNoAllocation checks that a prediction into a buffer of enough capacity
// does not allocate, for every kind of model.
func TestNoAllocation(t *testing.T) {
	if raceEnabled {
		t.Skip("allocation counts are not meaningful under the race detector")
	}
	for _, name := range testModels {
		t.Run(name, func(t *testing.T) {
			m := openModel(t, name)
			dst := make([]fasttext.Prediction, 0, 8)
			text := "The quick brown fox jumps over the lazy dog, and then some more words follow."
			// Warm up: the first calls size the working memory.
			for i := 0; i < 4; i++ {
				m.AppendPredict(dst, text, 5, 0)
			}
			allocs := testing.AllocsPerRun(200, func() {
				if _, err := m.AppendPredict(dst, text, 5, 0); err != nil {
					t.Fatal(err)
				}
			})
			if allocs != 0 {
				t.Errorf("%v allocations per prediction, want 0", allocs)
			}
		})
	}
}

func corpusLines(tb testing.TB, name string, max int) []string {
	tb.Helper()
	f, err := os.Open(testdata.Path(tb, "corpus/"+name))
	if err != nil {
		tb.Skipf("corpus not available: %v", err)
	}
	defer f.Close()
	var lines []string
	sc := bufio.NewScanner(f)
	sc.Buffer(nil, 1<<24)
	for sc.Scan() && len(lines) < max {
		lines = append(lines, sc.Text())
	}
	return lines
}

// TestConcurrent runs predictions from many goroutines on one model and
// compares them with the results of a single goroutine. Run it with -race.
func TestConcurrent(t *testing.T) {
	lines := corpusLines(t, "tatoeba-golden.txt", 2012)
	for _, name := range []string{"lid.176.ftz", "t-softmax-d32.ftz", "t-ova-d64.ftz", "t-softmax-d32.bin"} {
		t.Run(name, func(t *testing.T) {
			m := openModel(t, name)
			want := make([][]fasttext.Prediction, len(lines))
			for i, l := range lines {
				p, err := m.Predict(l, 3, 0)
				if err != nil {
					t.Fatal(err)
				}
				want[i] = p
			}
			var wg sync.WaitGroup
			errs := make(chan string, 16)
			for g := 0; g < 16; g++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					var buf []fasttext.Prediction
					for n := 0; n < len(lines); n++ {
						i := (n*7 + g*131) % len(lines)
						var err error
						buf, err = m.AppendPredict(buf[:0], lines[i], 3, 0)
						if err != nil || len(buf) != len(want[i]) {
							errs <- fmt.Sprintf("line %d: %v, %v", i, buf, err)
							return
						}
						for j := range buf {
							if buf[j] != want[i][j] {
								errs <- fmt.Sprintf("line %d: got %v, want %v", i, buf, want[i])
								return
							}
						}
					}
				}()
			}
			wg.Wait()
			close(errs)
			for e := range errs {
				t.Error(e)
			}
		})
	}
}

func TestLoadFromReader(t *testing.T) {
	path := testdata.Model(t, "lid.176.ftz")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("model not available: %v", err)
	}
	// A reader that hides its length.
	m, err := fasttext.Load(struct{ io.Reader }{bytes.NewReader(data)})
	if err != nil {
		t.Fatal(err)
	}
	p, err := m.Predict("Bonjour tout le monde", 1, 0)
	if err != nil || len(p) != 1 || p[0].Label != "__label__fr" {
		t.Errorf("Predict = %v, %v", p, err)
	}

	if _, err := fasttext.Load(strings.NewReader("this is not a model file")); err == nil {
		t.Error("Load of text: no error")
	}
	if _, err := fasttext.Open(filepath.Join(t.TempDir(), "missing.bin")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("Open of a missing file: %v", err)
	}
}
