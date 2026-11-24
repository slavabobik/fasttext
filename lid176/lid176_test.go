package lid176_test

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"testing"

	"github.com/slavabobik/fasttext/lid176"
)

func TestEmbeddedChecksum(t *testing.T) {
	h := sha256.New()
	n, err := io.Copy(h, lid176.Reader())
	if err != nil {
		t.Fatal(err)
	}
	if n != lid176.Size {
		t.Errorf("embedded model is %d bytes, want %d", n, lid176.Size)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != lid176.SHA256 {
		t.Errorf("embedded model has SHA-256 %s, want %s", got, lid176.SHA256)
	}
}

func TestDetector(t *testing.T) {
	d, err := lid176.Detector()
	if err != nil {
		t.Fatal(err)
	}
	d2, err := lid176.Detector()
	if err != nil {
		t.Fatal(err)
	}
	if d != d2 {
		t.Error("Detector returned two different detectors")
	}
	if n := len(d.Languages()); n != 176 {
		t.Errorf("%d languages, want 176", n)
	}
	if info := d.Model().Info(); info.Dim != 16 || info.Loss != "hs" {
		t.Errorf("unexpected model settings %+v", info)
	}
	for text, want := range map[string]string{
		"Bonjour tout le monde":                  "fr",
		"The quick brown fox jumps over the dog": "en",
		"Съешь же ещё этих мягких французских булок": "ru",
		"今日はいい天気ですね": "ja",
	} {
		got, err := lid176.Detect(text)
		if err != nil {
			t.Fatal(err)
		}
		if got.Code != want {
			t.Errorf("Detect(%q) = %v, want %s", text, got, want)
		}
	}
}

func Example() {
	lang, err := lid176.Detect("Où est ma commande ? Elle devait arriver hier.")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%s %.2f\n", lang.Code, lang.Prob)
	// Output:
	// fr 0.99
}

func ExampleDetector() {
	d, err := lid176.Detector() // loads the model once
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, text := range []string{"Where is my order?", "Wo ist meine Bestellung?", "Где мой заказ?"} {
		lang, err := d.Detect(text)
		if err != nil {
			fmt.Println(err)
			return
		}
		fmt.Println(lang.Code)
	}
	// Output:
	// en
	// de
	// ru
}

func BenchmarkModel(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := lid176.Model(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDetect(b *testing.B) {
	d, err := lid176.Detector()
	if err != nil {
		b.Fatal(err)
	}
	text := "Where is my order? It was supposed to arrive yesterday."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		d.Detect(text)
	}
}

func BenchmarkDetectParallel(b *testing.B) {
	d, err := lid176.Detector()
	if err != nil {
		b.Fatal(err)
	}
	text := "Where is my order? It was supposed to arrive yesterday."
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			d.Detect(text)
		}
	})
}
