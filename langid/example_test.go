package langid_test

import (
	"fmt"

	"github.com/slavabobik/fasttext/langid"
)

func Example() {
	d, err := langid.Open("../accuracy/models/lid.176.ftz")
	if err != nil {
		fmt.Println(err)
		return
	}
	lang, err := d.Detect("Où est ma commande ? Elle devait arriver hier.")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(lang.Code)
	// Output:
	// fr
}

func ExampleDetector_DetectTop() {
	d, err := langid.Open("../accuracy/models/lid.176.ftz")
	if err != nil {
		fmt.Println(err)
		return
	}
	langs, err := d.DetectTop("Bonjour tout le monde, comment allez-vous ?", 2, 0)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, l := range langs {
		fmt.Printf("%s %.4f\n", l.Code, l.Prob)
	}
	// Output:
	// fr 0.9867
	// en 0.0039
}

func ExampleDetector_Detect_noText() {
	d, err := langid.Open("../accuracy/models/lid.176.ftz")
	if err != nil {
		fmt.Println(err)
		return
	}
	_, err = d.Detect("   ")
	fmt.Println(err)
	// Output:
	// langid: text has no words
}
