// Command ftbench measures the prediction speed of the Go port. It is the
// counterpart of accuracy/bench/ftbench.cc, which measures fastText itself,
// and prints the same columns.
//
//	ftbench <model> <corpus> <k> <threads> <seconds>
//
// Output, tab-separated: tool, model, corpus, k, threads, texts per pass,
// bytes per pass, predictions returned in one pass, load time in ms, ns per
// text (per thread), texts per second (all threads), peak memory in MB.
package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/slavabobik/fasttext"
)

func main() {
	if len(os.Args) != 6 {
		fmt.Fprintln(os.Stderr, "usage: ftbench <model> <corpus> <k> <threads> <seconds>")
		os.Exit(2)
	}
	k, err1 := strconv.Atoi(os.Args[3])
	threads, err2 := strconv.Atoi(os.Args[4])
	budget, err3 := strconv.ParseFloat(os.Args[5], 64)
	if err1 != nil || err2 != nil || err3 != nil || threads < 1 {
		fmt.Fprintln(os.Stderr, "ftbench: invalid arguments")
		os.Exit(2)
	}
	runtime.GOMAXPROCS(threads)

	t0 := time.Now()
	m, err := fasttext.Open(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "ftbench:", err)
		os.Exit(1)
	}
	loadMs := float64(time.Since(t0)) / 1e6

	f, err := os.Open(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, "ftbench:", err)
		os.Exit(1)
	}
	var texts []string
	var bytes int64
	sc := bufio.NewScanner(f)
	sc.Buffer(nil, 1<<26)
	for sc.Scan() {
		texts = append(texts, sc.Text())
		bytes += int64(len(sc.Text()))
	}
	f.Close()
	if len(texts) == 0 {
		fmt.Fprintln(os.Stderr, "ftbench: empty corpus")
		os.Exit(1)
	}

	// One pass to warm up and to count the predictions.
	var returned int64
	var preds []fasttext.Prediction
	for _, text := range texts {
		preds, err = m.AppendPredict(preds[:0], text, k, 0)
		if err != nil {
			fmt.Fprintln(os.Stderr, "ftbench:", err)
			os.Exit(1)
		}
		returned += int64(len(preds))
	}

	rate := make([]float64, threads)
	var ready, done sync.WaitGroup
	start := make(chan struct{})
	for t := 0; t < threads; t++ {
		ready.Add(1)
		done.Add(1)
		go func() {
			defer done.Done()
			var preds []fasttext.Prediction
			ready.Done()
			<-start
			begin := time.Now()
			var n int64
			var elapsed float64
			for {
				for _, text := range texts {
					preds, _ = m.AppendPredict(preds[:0], text, k, 0)
				}
				n += int64(len(texts))
				elapsed = time.Since(begin).Seconds()
				if elapsed >= budget {
					break
				}
			}
			rate[t] = float64(n) / elapsed
		}()
	}
	ready.Wait()
	close(start)
	done.Wait()

	var total float64
	for _, r := range rate {
		total += r
	}
	rssMb := peakRSSMegabytes()
	fmt.Printf("%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%.1f\t%.0f\t%.0f\t%.1f\n",
		"go", filepath.Base(os.Args[1]), filepath.Base(os.Args[2]), k, threads, len(texts), bytes, returned,
		loadMs, 1e9*float64(threads)/total, total, rssMb)
}
