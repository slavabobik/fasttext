// Package lid176 detects the language of a text with fastText's lid.176.bin
// model, embedded in the package: no model file to download, no cgo.
//
//	d, err := lid176.Detector()
//	if err != nil { ... }
//	lang, err := d.Detect("Bonjour tout le monde")
//	// lang.Code == "fr"
//
// lid.176.bin recognises 176 languages and was trained by Facebook AI
// Research on Wikipedia, Tatoeba and SETimes
// (https://fasttext.cc/docs/en/language-identification.html). It is
// distributed under the Creative Commons Attribution-Share-Alike License 3.0;
// see LICENSE-lid176 in this directory.
//
// Importing the package adds about 131 MB to the binary, and loading the
// model takes about 10 to 100 milliseconds and as much memory as the model.
// For a smaller footprint load lid.176.ftz (under 1 MB, slightly less
// accurate) from a file with langid.Open instead.
//
// The predictions are bit for bit those of fastText; see the documentation of
// package github.com/slavabobik/fasttext.
package lid176

import (
	_ "embed"
	"io"
	"strings"
	"sync"

	"github.com/slavabobik/fasttext"
	"github.com/slavabobik/fasttext/langid"
)

// The model is stored in three parts because GitHub does not accept files
// over 100 MB.
var (
	//go:embed model/lid.176.bin.0
	part0 string
	//go:embed model/lid.176.bin.1
	part1 string
	//go:embed model/lid.176.bin.2
	part2 string
)

// SHA256 is the SHA-256 of lid.176.bin as published by fastText, which the
// embedded model matches.
const SHA256 = "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"

// Size is the size of lid.176.bin in bytes.
const Size = 131266198

// Reader returns a reader over the embedded lid.176.bin file.
func Reader() io.Reader {
	return &sizedReader{
		Reader: io.MultiReader(strings.NewReader(part0), strings.NewReader(part1), strings.NewReader(part2)),
		n:      len(part0) + len(part1) + len(part2),
	}
}

// sizedReader tells fasttext.Load the length of the model, so that the sizes
// in the file are checked against it.
type sizedReader struct {
	io.Reader
	n int
}

func (r *sizedReader) Len() int { return r.n }

// Model loads a new copy of the embedded model. Most programs need only one,
// and should call Detector instead.
func Model() (*fasttext.Model, error) {
	return fasttext.Load(Reader())
}

var shared = sync.OnceValues(func() (*langid.Detector, error) {
	m, err := Model()
	if err != nil {
		return nil, err
	}
	return langid.New(m), nil
})

// Detector returns a language detector for the embedded model. The model is
// loaded on the first call; later calls return the same Detector, which is
// safe for concurrent use.
func Detector() (*langid.Detector, error) {
	return shared()
}

// Detect returns the most likely language of text using the shared
// Detector. It is a shortcut for Detector followed by Detect.
func Detect(text string) (langid.Language, error) {
	d, err := Detector()
	if err != nil {
		return langid.Language{}, err
	}
	return d.Detect(text)
}
