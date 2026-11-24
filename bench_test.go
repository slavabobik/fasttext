package fasttext_test

import (
	"testing"

	"github.com/slavabobik/fasttext"
	"github.com/slavabobik/fasttext/internal/testdata"
)

// The benchmarks predict the texts of a corpus in turn. One operation is one
// text, so ns/op is the time per text. For the comparison with fastText
// itself see cmd/ftbench and accuracy/bench.

func benchPredict(b *testing.B, model, corpus string, k int) {
	m := openModel(b, model)
	texts := corpusLines(b, corpus, 1<<30)
	var bytes int64
	for _, t := range texts {
		bytes += int64(len(t))
	}
	var preds []fasttext.Prediction
	for _, t := range texts {
		preds, _ = m.AppendPredict(preds[:0], t, k, 0)
	}
	b.SetBytes(bytes / int64(len(texts)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		preds, _ = m.AppendPredict(preds[:0], texts[i%len(texts)], k, 0)
	}
}

func BenchmarkPredict(b *testing.B) {
	for _, model := range testModels {
		b.Run(model, func(b *testing.B) { benchPredict(b, model, "tatoeba-golden.txt", 1) })
	}
}

func BenchmarkPredictTop5(b *testing.B) {
	for _, model := range []string{"lid.176.bin", "lid.176.ftz", "t-softmax-d256.bin"} {
		b.Run(model, func(b *testing.B) { benchPredict(b, model, "tatoeba-golden.txt", 5) })
	}
}

func BenchmarkPredictParagraph(b *testing.B) {
	for _, model := range []string{"lid.176.bin", "lid.176.ftz", "t-softmax-d256.bin"} {
		b.Run(model, func(b *testing.B) { benchPredict(b, model, "bench-paragraphs.txt", 1) })
	}
}

// BenchmarkPredictParallel runs predictions on one model from all
// processors.
func BenchmarkPredictParallel(b *testing.B) {
	for _, model := range []string{"lid.176.bin", "lid.176.ftz", "t-softmax-d256.bin"} {
		b.Run(model, func(b *testing.B) {
			m := openModel(b, model)
			texts := corpusLines(b, "tatoeba-golden.txt", 1<<30)
			b.ReportAllocs()
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				var preds []fasttext.Prediction
				i := 0
				for pb.Next() {
					preds, _ = m.AppendPredict(preds[:0], texts[i%len(texts)], 1, 0)
					i++
				}
			})
		})
	}
}

func BenchmarkOpen(b *testing.B) {
	for _, model := range []string{"lid.176.bin", "lid.176.ftz"} {
		b.Run(model, func(b *testing.B) {
			openModel(b, model)
			path := testdata.Model(b, model)
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := fasttext.Open(path); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
