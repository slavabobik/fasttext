// Package langid detects the language of a text with a fastText
// language-identification model, in pure Go (no cgo).
//
// It is a thin layer over package fasttext that speaks in language codes
// rather than fastText labels:
//
//	d, err := langid.Open("lid.176.bin")
//	if err != nil { ... }
//	lang, err := d.Detect("Bonjour tout le monde")
//	// lang.Code == "fr", lang.Prob ≈ 0.98
//
// For a detector that needs no model file, use the lid176 module
// (github.com/slavabobik/fasttext/lid176), which embeds lid.176.bin.
//
// Any fastText language-identification model works: lid.176.bin and
// lid.176.ftz from fastText (codes such as "en", "zh", "als"), GlotLID and
// OpenLID (codes such as "eng_Latn"). The code is the model's label without
// the "__label__" prefix.
//
// # Differences from calling the model directly
//
// The probabilities are those of fasttext.Model.Predict, bit for bit, on the
// same text with two conveniences:
//
//   - Line breaks are replaced by spaces, so a multi-line text is classified
//     as one text. (fastText's Python package rejects a text with "\n".)
//   - A text without any word, that is empty or made only of whitespace and
//     NUL bytes, gives ErrNoText. fastText would still return its most
//     frequent label with a low probability.
//
// A Detector is safe for concurrent use by any number of goroutines.
package langid

import (
	"errors"
	"io"
	"strings"
	"sync"

	"github.com/slavabobik/fasttext"
)

// ErrNoText is returned for a text with nothing to classify: empty, or only
// whitespace.
var ErrNoText = errors.New("langid: text has no words")

const labelPrefix = "__label__"

// Language is one detected language.
type Language struct {
	// Code is the model's label without "__label__": an ISO 639 code such
	// as "en" for lid.176, or a code with script such as "eng_Latn" for
	// GlotLID and OpenLID.
	Code string
	// Prob is the probability the model gives this language, between 0
	// and 1. Like fastText's, it can exceed 1 by about 1e-5.
	Prob float32
}

// Detector detects languages with one fastText model.
type Detector struct {
	m     *fasttext.Model
	codes []string
	pool  sync.Pool // *[]fasttext.Prediction
}

// New returns a Detector that uses m.
func New(m *fasttext.Model) *Detector {
	labels := m.Labels()
	codes := make([]string, len(labels))
	for i, l := range labels {
		codes[i] = strings.TrimPrefix(l, labelPrefix)
	}
	return &Detector{m: m, codes: codes}
}

// Open loads the fastText model file at path (.bin or .ftz) and returns a
// Detector for it.
func Open(path string) (*Detector, error) {
	m, err := fasttext.Open(path)
	if err != nil {
		return nil, err
	}
	return New(m), nil
}

// Load reads a fastText model from r and returns a Detector for it.
func Load(r io.Reader) (*Detector, error) {
	m, err := fasttext.Load(r)
	if err != nil {
		return nil, err
	}
	return New(m), nil
}

// Detect returns the most likely language of text.
func (d *Detector) Detect(text string) (Language, error) {
	var buf [1]Language
	langs, err := d.AppendDetect(buf[:0], text, 1, 0)
	if err != nil {
		return Language{}, err
	}
	if len(langs) == 0 {
		return Language{}, ErrNoText
	}
	return langs[0], nil
}

// DetectTop returns up to k languages, most likely first, leaving out those
// with a probability below threshold. A k of -1 means all languages.
//
// A text with words can still give an empty result if every language is
// below threshold.
func (d *Detector) DetectTop(text string, k int, threshold float32) ([]Language, error) {
	return d.AppendDetect(nil, text, k, threshold)
}

// AppendDetect is DetectTop appending to dst. For a text without line
// breaks and a dst with enough capacity it does not allocate.
func (d *Detector) AppendDetect(dst []Language, text string, k int, threshold float32) ([]Language, error) {
	if !hasWord(text) {
		return dst, ErrNoText
	}
	if strings.IndexByte(text, '\n') >= 0 {
		text = strings.ReplaceAll(text, "\n", " ")
	}
	bp, _ := d.pool.Get().(*[]fasttext.Prediction)
	if bp == nil {
		bp = new([]fasttext.Prediction)
	}
	preds, err := d.m.AppendPredict((*bp)[:0], text, k, threshold)
	if err == nil {
		for _, p := range preds {
			dst = append(dst, Language{Code: strings.TrimPrefix(p.Label, labelPrefix), Prob: p.Prob})
		}
	}
	*bp = preds
	d.pool.Put(bp)
	return dst, err
}

// Languages returns the codes of every language the model knows, in the
// order of the model file. The slice must not be modified.
func (d *Detector) Languages() []string { return d.codes }

// Model returns the underlying fastText model.
func (d *Detector) Model() *fasttext.Model { return d.m }

// hasWord reports whether text has a byte that fastText does not treat as a
// separator, that is, at least one token.
func hasWord(text string) bool {
	for i := 0; i < len(text); i++ {
		switch text[i] {
		case ' ', '\n', '\r', '\t', '\v', '\f', 0:
		default:
			return true
		}
	}
	return false
}
