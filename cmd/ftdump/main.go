// Command ftdump writes predictions of the Go port layer by layer, in the
// format of the reference tool built from the fastText sources
// (accuracy/refdump). The two outputs must be identical byte for byte.
//
//	ftdump [--ids] <model> [<k>[,<k>...]] [<threshold>] < text > dump
//
// Every input line is one record. See accuracy/README.md for the format.
package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/slavabobik/fasttext"
	"github.com/slavabobik/fasttext/internal/refdump"
)

func main() {
	fullIDs := false
	var pos []string
	for _, a := range os.Args[1:] {
		if a == "--ids" {
			fullIDs = true
		} else {
			pos = append(pos, a)
		}
	}
	if len(pos) < 1 || len(pos) > 3 {
		fmt.Fprintln(os.Stderr, "usage: ftdump [--ids] <model> [<k>[,<k>...]] [<threshold>]")
		os.Exit(2)
	}
	ks := []int{1}
	if len(pos) > 1 {
		ks = ks[:0]
		for _, f := range strings.Split(pos[1], ",") {
			k, err := strconv.Atoi(f)
			if err != nil {
				fatal(err)
			}
			ks = append(ks, k)
		}
	}
	var threshold float32
	if len(pos) > 2 {
		t, err := strconv.ParseFloat(pos[2], 32)
		if err != nil {
			fatal(err)
		}
		threshold = float32(t)
	}

	m, err := fasttext.Open(pos[0])
	if err != nil {
		fatal(err)
	}
	n, err := refdump.Write(os.Stdout, m, os.Stdin, ks, threshold, fullIDs)
	if err != nil {
		fatal(err)
	}
	fmt.Fprintf(os.Stderr, "ftdump: %d records\n", n)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "ftdump:", err)
	os.Exit(1)
}
