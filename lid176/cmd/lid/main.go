// Command lid detects the language of text with fastText's lid.176.bin
// model, which is built into the binary. It needs no model file and no cgo.
//
// Install:
//
//	go install github.com/slavabobik/fasttext/lid176/cmd/lid@latest
//
// Usage:
//
//	lid [flags] [text ...]
//
// With arguments, the arguments joined by spaces are one text. Without, every
// line of standard input is one text. For each text lid prints one line: the
// language code and its probability, tab-separated, for each of the k best
// languages. A text with no words gives an empty line.
//
//	$ echo "Bonjour tout le monde" | lid -k 2
//	fr	0.9946	en	0.0009
//
// Flags:
//
//	-k n          print the n best languages (default 1; -1 for all)
//	-threshold p  leave out languages with a probability below p
//	-json         print one JSON object per text instead
//	-model path   use this fastText model file instead of the built-in one
//	              (lid.176.ftz, GlotLID, OpenLID, ...)
package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/slavabobik/fasttext/langid"
	"github.com/slavabobik/fasttext/lid176"
)

func main() {
	k := flag.Int("k", 1, "number of languages to print, -1 for all")
	threshold := flag.Float64("threshold", 0, "leave out languages with a probability below this")
	asJSON := flag.Bool("json", false, "print one JSON object per text")
	modelPath := flag.String("model", "", "fastText model file to use instead of the built-in lid.176.bin")
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "usage: lid [flags] [text ...]\n\nDetects the language of the text given as arguments, or of each line of standard input.\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	if *k == 0 || *k < -1 {
		fatal(errors.New("-k must be 1 or more, or -1 for all"))
	}

	var d *langid.Detector
	var err error
	if *modelPath != "" {
		d, err = langid.Open(*modelPath)
	} else {
		d, err = lid176.Detector()
	}
	if err != nil {
		fatal(err)
	}

	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()
	p := printer{d: d, k: *k, threshold: float32(*threshold), json: *asJSON, out: out}

	if flag.NArg() > 0 {
		if err := p.print(strings.Join(flag.Args(), " ")); err != nil {
			fatal(err)
		}
		return
	}
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 64*1024), 64*1024*1024)
	for sc.Scan() {
		if err := p.print(sc.Text()); err != nil {
			out.Flush()
			fatal(err)
		}
	}
	if err := sc.Err(); err != nil {
		out.Flush()
		fatal(err)
	}
}

type printer struct {
	d         *langid.Detector
	k         int
	threshold float32
	json      bool
	out       io.Writer
	buf       []langid.Language
}

type jsonLanguage struct {
	Code string  `json:"code"`
	Prob float32 `json:"prob"`
}

type jsonResult struct {
	Text      string         `json:"text"`
	Languages []jsonLanguage `json:"languages"`
}

func (p *printer) print(text string) error {
	langs, err := p.d.AppendDetect(p.buf[:0], text, p.k, p.threshold)
	p.buf = langs
	if err != nil && !errors.Is(err, langid.ErrNoText) {
		return err
	}
	if p.json {
		r := jsonResult{Text: text, Languages: make([]jsonLanguage, len(langs))}
		for i, l := range langs {
			r.Languages[i] = jsonLanguage(l)
		}
		b, err := json.Marshal(r)
		if err != nil {
			return err
		}
		b = append(b, '\n')
		_, err = p.out.Write(b)
		return err
	}
	var line []byte
	for i, l := range langs {
		if i > 0 {
			line = append(line, '\t')
		}
		line = append(line, l.Code...)
		line = append(line, '\t')
		line = strconv.AppendFloat(line, float64(l.Prob), 'f', 4, 32)
	}
	line = append(line, '\n')
	_, err = p.out.Write(line)
	return err
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "lid:", err)
	os.Exit(1)
}
