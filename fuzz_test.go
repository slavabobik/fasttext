package fasttext_test

import (
	"bytes"
	"errors"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/slavabobik/fasttext"
	"github.com/slavabobik/fasttext/internal/testdata"
)

// FuzzPredict feeds arbitrary text to lid.176.ftz and checks the invariants
// of a prediction: no panic, labels best first, probabilities in range, and
// AppendPredict equal to Predict.
//
//	go test -run '^$' -fuzz FuzzPredict
func FuzzPredict(f *testing.F) {
	for _, s := range []string{"", " ", "hello world", "Bonjour tout le monde", "</s>", "__label__en x",
		"\xff\xfe\x00", strings.Repeat("ab ", 2000), "日本語のテキスト", "a\tb\vc\fd\re"} {
		f.Add(s, 3)
	}
	m, err := fasttext.Open(testdata.Model(f, "lid.176.ftz"))
	if err != nil {
		f.Fatal(err)
	}
	f.Fuzz(func(t *testing.T, text string, k int) {
		if k < -1 || k > 200 {
			k = 3
		}
		preds, err := m.Predict(text, k, 0)
		if strings.Contains(text, "\n") {
			if !errors.Is(err, fasttext.ErrNewline) {
				t.Fatalf("text with newline: err = %v", err)
			}
			return
		}
		if k == 0 {
			if !errors.Is(err, fasttext.ErrK) {
				t.Fatalf("k = 0: err = %v", err)
			}
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		if k > 0 && len(preds) > k {
			t.Fatalf("%d predictions for k = %d", len(preds), k)
		}
		for i, p := range preds {
			if !(p.Prob >= 0 && p.Prob <= 1.0001) {
				t.Fatalf("probability %v out of range", p.Prob)
			}
			if i > 0 && p.Prob > preds[i-1].Prob {
				t.Fatalf("not sorted: %v", preds)
			}
		}
		again, err := m.AppendPredict(make([]fasttext.Prediction, 0, 4), text, k, 0)
		if err != nil || len(again) != len(preds) {
			t.Fatalf("AppendPredict: %v, %d predictions, Predict %d", err, len(again), len(preds))
		}
		for i := range again {
			if again[i] != preds[i] {
				t.Fatalf("AppendPredict differs at %d: %v, Predict %v", i, again[i], preds[i])
			}
		}
	})
}

// FuzzLoad loads mutated model files twice: from a reader that tells its
// length and from one that does not, which takes the path of an HTTP body or
// a gzip stream. Loading may fail, and a model that loads may predict
// anything, but nothing may panic or exhaust memory, and the two loads must
// agree. The seeds are whole model files, which the fuzzer is slow to
// minimize: without -fuzzminimizetime 0 it stops for a minute after each new
// input.
//
//	go test -run '^$' -fuzz FuzzLoad -fuzzminimizetime 0
func FuzzLoad(f *testing.F) {
	for _, name := range []string{"t-ova-d64.ftz", "t-softmax-d32.ftz"} {
		data, err := os.ReadFile(testdata.Model(f, name))
		if err != nil {
			f.Fatal(err)
		}
		f.Add(data)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		m, err := fasttext.Load(bytes.NewReader(data))
		stream, streamErr := fasttext.Load(struct{ io.Reader }{bytes.NewReader(data)})
		if (err == nil) != (streamErr == nil) {
			t.Fatalf("known size: %v; unknown size: %v", err, streamErr)
		}
		if err != nil {
			return
		}
		for _, text := range []string{"", "hello world", "Это предложение"} {
			p, err := m.Predict(text, 2, 0)
			q, streamErr := stream.Predict(text, 2, 0)
			if (err == nil) != (streamErr == nil) || len(p) != len(q) {
				t.Fatalf("%q: known size %v, %v; unknown size %v, %v", text, p, err, q, streamErr)
			}
			for i := range p {
				if p[i] != q[i] && !(p[i].Label == q[i].Label && p[i].Prob != p[i].Prob && q[i].Prob != q[i].Prob) {
					t.Fatalf("%q: known size %v; unknown size %v", text, p, q)
				}
			}
		}
	})
}
