package fasttext_test

import (
	"errors"
	"fmt"
	"strings"

	"github.com/slavabobik/fasttext"
)

// The examples use lid.176.ftz, the compressed language-identification model
// published by fastText (https://fasttext.cc/docs/en/language-identification.html).
// For language detection without managing a model file, see the langid
// package and the lid176 module, which embeds lid.176.bin.

func Example() {
	m, err := fasttext.Open("accuracy/models/lid.176.ftz")
	if err != nil {
		fmt.Println(err)
		return
	}
	preds, err := m.Predict("Bonjour tout le monde, comment allez-vous ?", 2, 0)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, p := range preds {
		fmt.Printf("%s %.4f\n", strings.TrimPrefix(p.Label, "__label__"), p.Prob)
	}
	// Output:
	// fr 0.9867
	// en 0.0039
}

// Predict with a threshold returns only the labels at or above it, best
// first.
func ExampleModel_Predict_threshold() {
	m, err := fasttext.Open("accuracy/models/lid.176.ftz")
	if err != nil {
		fmt.Println(err)
		return
	}
	preds, err := m.Predict("Wo ist meine Bestellung? Sie sollte gestern ankommen.", -1, 0.5)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, p := range preds {
		fmt.Println(p.Label)
	}
	// Output:
	// __label__de
}

// AppendPredict reuses the slice it is given, so a loop over many texts
// does not allocate.
func ExampleModel_AppendPredict() {
	m, err := fasttext.Open("accuracy/models/lid.176.ftz")
	if err != nil {
		fmt.Println(err)
		return
	}
	var preds []fasttext.Prediction
	for _, text := range []string{"Where is my order?", "¿Dónde está mi pedido?", "Где мой заказ?"} {
		preds, err = m.AppendPredict(preds[:0], text, 1, 0)
		if err != nil {
			fmt.Println(err)
			return
		}
		fmt.Println(strings.TrimPrefix(preds[0].Label, "__label__"))
	}
	// Output:
	// en
	// es
	// ru
}

// A text must be a single line, as in fastText's Python package.
func ExampleModel_Predict_newline() {
	m, err := fasttext.Open("accuracy/models/lid.176.ftz")
	if err != nil {
		fmt.Println(err)
		return
	}
	_, err = m.Predict("first line\nsecond line", 1, 0)
	fmt.Println(errors.Is(err, fasttext.ErrNewline))
	// Output:
	// true
}
