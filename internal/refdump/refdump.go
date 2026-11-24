// Package refdump writes predictions layer by layer in the format of the
// reference tool built from the fastText sources (accuracy/refdump/refdump.cc).
// Run on the same text, the two must produce the same bytes: that is how the
// port is checked against fastText.
package refdump

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"

	"github.com/slavabobik/fasttext"
)

// hexFloat formats f as C's printf("%a") does.
func hexFloat(f float64) string {
	s := strconv.FormatFloat(f, 'x', -1, 64)
	// Go writes at least two exponent digits, C as few as possible.
	i := strings.IndexByte(s, 'p') + 2
	j := i
	for j < len(s)-1 && s[j] == '0' {
		j++
	}
	return s[:i] + s[j:]
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// Write reads records from in, one per line, and writes the dump for the
// given k values and threshold to out. With fullIDs every input row is
// written instead of a count and a hash. It returns the number of records.
func Write(out io.Writer, m *fasttext.Model, in io.Reader, ks []int, threshold float32, fullIDs bool) (int64, error) {
	w := bufio.NewWriterSize(out, 1<<20)
	n, err := write(w, m, in, ks, threshold, fullIDs)
	if err == nil {
		err = w.Flush()
	}
	return n, err
}

func write(w *bufio.Writer, m *fasttext.Model, in io.Reader, ks []int, threshold float32, fullIDs bool) (int64, error) {
	info := m.Info()
	fmt.Fprintf(w, "#args dim=%d ws=%d epoch=%d minCount=%d neg=%d wordNgrams=%d loss=%s model=3 bucket=%d minn=%d maxn=%d lrUpdateRate=%d t=%s qout=%d\n",
		info.Dim, info.WS, info.Epoch, info.MinCount, info.Neg, info.WordNgrams, info.Loss,
		info.Bucket, info.Minn, info.Maxn, info.LrUpdateRate, hexFloat(info.T), b2i(info.QuantizedOutput))
	fmt.Fprintf(w, "#dict nwords=%d nlabels=%d ntokens=%d quant=%d\n", info.Words, info.Labels, info.Tokens, b2i(info.Quantized))
	w.WriteString("#labels")
	for _, l := range m.Labels() {
		w.WriteByte(' ')
		w.WriteString(l)
	}
	w.WriteByte('\n')
	fmt.Fprintf(w, "#run threshold=%s mode=line k=", hexFloat(float64(threshold)))
	for i, k := range ks {
		if i > 0 {
			w.WriteByte(',')
		}
		w.WriteString(strconv.Itoa(k))
	}
	w.WriteByte('\n')

	const hexDigits = "0123456789abcdef"
	hex32 := func(u uint32) {
		var b [8]byte
		for i := 7; i >= 0; i-- {
			b[i] = hexDigits[u&15]
			u >>= 4
		}
		w.Write(b[:])
	}

	r := bufio.NewReaderSize(in, 1<<20)
	var t fasttext.Trace
	var num []byte
	var n int64
	for {
		line, err := r.ReadString('\n')
		if err != nil && err != io.EOF {
			return n, err
		}
		if err == io.EOF && line == "" {
			return n, nil
		}
		line = strings.TrimSuffix(line, "\n")
		for i, k := range ks {
			if err := m.Trace(&t, line, k, threshold); err != nil {
				return n, fmt.Errorf("record %d: %w", n, err)
			}
			if i == 0 {
				w.WriteString("ids=")
				w.WriteString(strconv.Itoa(len(t.IDs)))
				w.WriteByte(':')
				if fullIDs {
					for j, id := range t.IDs {
						if j > 0 {
							w.WriteByte(',')
						}
						num = strconv.AppendInt(num[:0], int64(id), 10)
						w.Write(num)
					}
				} else {
					h := uint64(14695981039346656037)
					for _, id := range t.IDs {
						u := uint32(id)
						for s := 0; s < 32; s += 8 {
							h = (h ^ uint64(byte(u>>s))) * 1099511628211
						}
					}
					fmt.Fprintf(w, "%016x", h)
				}
				w.WriteString("\th=")
				for _, v := range t.Hidden {
					hex32(math.Float32bits(v))
				}
			}
			w.WriteString("\tk")
			w.WriteString(strconv.Itoa(k))
			w.WriteByte('=')
			for j, l := range t.Labels {
				if j > 0 {
					w.WriteByte(',')
				}
				num = strconv.AppendInt(num[:0], int64(l), 10)
				w.Write(num)
				w.WriteByte(':')
				hex32(math.Float32bits(t.Scores[j]))
				w.WriteByte(':')
				hex32(math.Float32bits(t.Probs[j]))
			}
		}
		w.WriteByte('\n')
		n++
		if err == io.EOF {
			return n, nil
		}
	}
}
