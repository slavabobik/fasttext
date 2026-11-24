package langid_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/slavabobik/fasttext/langid"
)

// The benchmarks below use lid.176.bin, the model of package lid176. One
// operation is one text, so ns/op is the time per text, and MB/s is the
// throughput in bytes of UTF-8.

// BenchmarkDetectScript times the samples of each writing system. Scripts
// outside Latin take 2 to 3 bytes per character, so the same sentence costs
// more bytes, and more character n-grams, to look up.
func BenchmarkDetectScript(b *testing.B) {
	d := open(b, "lid.176.bin")
	var order []string
	byScript := map[string][]string{}
	for _, s := range samples {
		if byScript[s.script] == nil {
			order = append(order, s.script)
		}
		byScript[s.script] = append(byScript[s.script], s.text)
	}
	for _, script := range order {
		texts := byScript[script]
		b.Run(script, func(b *testing.B) { benchDetect(b, d, texts) })
	}
}

// BenchmarkDetectLength times texts from 16 bytes to 64 KB, cut from the
// paragraphs of 144 languages. The time grows with the number of words and
// character n-grams: a supervised model reads the whole text.
func BenchmarkDetectLength(b *testing.B) {
	d := open(b, "lid.176.bin")
	long := strings.Join(corpus(b, "bench-paragraphs.txt"), " ")
	for _, n := range []int{16, 64, 256, 1 << 10, 4 << 10, 16 << 10, 64 << 10} {
		text := cut(long, n)
		b.Run(fmt.Sprintf("%dB", n), func(b *testing.B) { benchDetect(b, d, []string{text}) })
	}
}

// BenchmarkAppendDetectTop times AppendDetect for the top 1, the top 5 and
// all 176 languages.
func BenchmarkAppendDetectTop(b *testing.B) {
	d := open(b, "lid.176.bin")
	texts := corpus(b, "tatoeba-golden.txt")
	for _, k := range []int{1, 5, -1} {
		name := fmt.Sprintf("k=%d", k)
		if k < 0 {
			name = "k=all"
		}
		b.Run(name, func(b *testing.B) {
			dst := make([]langid.Language, 0, 176)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				dst, _ = d.AppendDetect(dst[:0], texts[i%len(texts)], k, 0)
			}
		})
	}
}

func benchDetect(b *testing.B, d *langid.Detector, texts []string) {
	n := 0
	for _, t := range texts {
		n += len(t)
	}
	b.SetBytes(int64(n / len(texts)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := d.Detect(texts[i%len(texts)]); err != nil {
			b.Fatal(err)
		}
	}
}

// cut returns the first n bytes of s or less, ending at a space.
func cut(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if i := strings.LastIndexByte(s[:n+1], ' '); i > 0 {
		return s[:i]
	}
	return s[:n]
}
