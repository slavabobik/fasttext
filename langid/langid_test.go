package langid_test

import (
	"bufio"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/slavabobik/fasttext"
	"github.com/slavabobik/fasttext/internal/testdata"
	"github.com/slavabobik/fasttext/langid"
)

func open(tb testing.TB, model string) *langid.Detector {
	tb.Helper()
	d, err := langid.Open(testdata.Model(tb, model))
	if err != nil {
		tb.Fatal(err)
	}
	return d
}

// closeNeighbours are the languages of samples that lid.176 takes for a close
// neighbour on a single short sentence, the neighbours named. For these
// TestDetect accepts the language among the top three answers.
// docs/LANGUAGES.md shows the same pairs on Tatoeba.
var closeNeighbours = map[string]string{
	"gl": "pt",       // Galician, Portuguese
	"no": "da",       // Norwegian, Danish
	"sl": "sr hr",    // Slovene, Serbian, Croatian
	"hr": "bs sr sl", // Croatian, Bosnian, Serbian, Slovene
	"ms": "id",       // Malay, Indonesian
}

func TestDetect(t *testing.T) {
	for _, model := range []string{"lid.176.bin", "lid.176.ftz"} {
		t.Run(model, func(t *testing.T) {
			d := open(t, model)
			right := 0
			for _, s := range samples {
				top, err := d.DetectTop(s.text, 3, 0)
				if err != nil {
					t.Fatalf("DetectTop(%q): %v", s.text, err)
				}
				if top[0].Prob <= 0 || top[0].Prob > 1.0001 {
					t.Errorf("%s: probability %v out of range", s.code, top[0].Prob)
				}
				if top[0].Code == s.code {
					right++
					continue
				}
				if _, ok := closeNeighbours[s.code]; ok && (top[1].Code == s.code || top[2].Code == s.code) {
					continue
				}
				t.Errorf("%s %q: got %v", s.code, s.text, top)
			}
			t.Logf("%d of %d samples right at the top", right, len(samples))
		})
	}
}

// TestSameAsModel checks that the detector returns the model's own
// predictions, with the prefix removed and newlines read as spaces.
func TestSameAsModel(t *testing.T) {
	d := open(t, "lid.176.ftz")
	m := d.Model()
	texts := corpus(t, "tatoeba-golden.txt")
	texts = append(texts, "line one\nline two", "\nleading", "trailing\n", "a\n\nb")
	for _, text := range texts {
		for _, k := range []int{1, 3, -1} {
			got, gotErr := d.DetectTop(text, k, 0)
			want, wantErr := m.Predict(strings.ReplaceAll(text, "\n", " "), k, 0)
			if (gotErr != nil) != (wantErr != nil) {
				t.Fatalf("%q k=%d: error %v, model %v", text, k, gotErr, wantErr)
			}
			if len(got) != len(want) {
				t.Fatalf("%q k=%d: %d languages, model %d", text, k, len(got), len(want))
			}
			for i := range got {
				if "__label__"+got[i].Code != want[i].Label || got[i].Prob != want[i].Prob {
					t.Fatalf("%q k=%d #%d: %+v, model %+v", text, k, i, got[i], want[i])
				}
			}
		}
	}
}

func TestNoText(t *testing.T) {
	d := open(t, "lid.176.ftz")
	for _, text := range []string{"", " ", "\n", " \t\r\n\v\f ", "\x00"} {
		if _, err := d.Detect(text); !errors.Is(err, langid.ErrNoText) {
			t.Errorf("Detect(%q): err = %v, want ErrNoText", text, err)
		}
		if langs, err := d.DetectTop(text, 5, 0); !errors.Is(err, langid.ErrNoText) || len(langs) != 0 {
			t.Errorf("DetectTop(%q) = %v, %v; want ErrNoText", text, langs, err)
		}
	}
}

func TestDetectTop(t *testing.T) {
	d := open(t, "lid.176.ftz")
	langs, err := d.DetectTop("Bonjour tout le monde", 3, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(langs) != 3 || langs[0].Code != "fr" {
		t.Fatalf("DetectTop = %v", langs)
	}
	for i := 1; i < len(langs); i++ {
		if langs[i].Prob > langs[i-1].Prob {
			t.Errorf("not sorted: %v", langs)
		}
	}
	all, err := d.DetectTop("Bonjour tout le monde", -1, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Hierarchical softmax drops labels below about 1e-5.
	if len(all) < 3 || len(all) > len(d.Languages()) {
		t.Errorf("DetectTop(-1) returned %d of %d languages", len(all), len(d.Languages()))
	}
	high, err := d.DetectTop("Bonjour tout le monde", -1, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if len(high) != 1 || high[0].Code != "fr" {
		t.Errorf("DetectTop with threshold 0.5 = %v", high)
	}
	if _, err := d.DetectTop("hello", 0, 0); !errors.Is(err, fasttext.ErrK) {
		t.Errorf("k=0: err = %v, want ErrK", err)
	}
}

func TestLanguages(t *testing.T) {
	d := open(t, "lid.176.ftz")
	langs := d.Languages()
	if len(langs) != 176 {
		t.Fatalf("%d languages, want 176", len(langs))
	}
	seen := map[string]bool{}
	for _, l := range langs {
		if strings.HasPrefix(l, "__label__") || l == "" || seen[l] {
			t.Errorf("bad or duplicate code %q", l)
		}
		seen[l] = true
	}
	for _, want := range []string{"en", "fr", "de", "zh", "ja", "ru", "ar"} {
		if !seen[want] {
			t.Errorf("missing %q", want)
		}
	}
}

func TestNoAllocation(t *testing.T) {
	if raceEnabled {
		t.Skip("allocation counts are not meaningful under the race detector")
	}
	d := open(t, "lid.176.ftz")
	dst := make([]langid.Language, 0, 8)
	text := "Wo ist meine Bestellung? Sie sollte gestern ankommen."
	d.AppendDetect(dst, text, 5, 0) // warm the pool
	allocs := testing.AllocsPerRun(100, func() {
		dst, _ = d.AppendDetect(dst[:0], text, 5, 0)
	})
	if allocs != 0 {
		t.Errorf("AppendDetect allocates %v times per call", allocs)
	}
	allocs = testing.AllocsPerRun(100, func() {
		d.Detect(text)
	})
	if allocs != 0 {
		t.Errorf("Detect allocates %v times per call", allocs)
	}
}

func TestConcurrent(t *testing.T) {
	d := open(t, "lid.176.ftz")
	want := make([]langid.Language, len(samples))
	for i, s := range samples {
		want[i], _ = d.Detect(s.text)
	}
	var wg sync.WaitGroup
	errs := make(chan string, 64)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for n := 0; n < 200; n++ {
				i := (n + g) % len(samples)
				got, err := d.Detect(samples[i].text)
				if err != nil || got != want[i] {
					select {
					case errs <- samples[i].text:
					default:
					}
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Errorf("concurrent result differs for %q", e)
	}
}

func corpus(tb testing.TB, name string) []string {
	tb.Helper()
	f, err := os.Open(testdata.Path(tb, "corpus/"+name))
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()
	var lines []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		lines = append(lines, sc.Text())
	}
	if err := sc.Err(); err != nil {
		tb.Fatal(err)
	}
	return lines
}

func BenchmarkDetect(b *testing.B) {
	for _, model := range []string{"lid.176.bin", "lid.176.ftz"} {
		b.Run(model, func(b *testing.B) {
			d := open(b, model)
			texts := corpus(b, "tatoeba-golden.txt")
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				d.Detect(texts[i%len(texts)])
			}
		})
	}
}

func BenchmarkDetectParallel(b *testing.B) {
	d := open(b, "lid.176.bin")
	texts := corpus(b, "tatoeba-golden.txt")
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			d.Detect(texts[i%len(texts)])
			i++
		}
	})
}
