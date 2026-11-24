package fasttext

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"strings"
	"unsafe"
)

// Info describes a loaded model: the settings stored in the model file and
// the size of its dictionary.
type Info struct {
	Version      int    // model file version
	Dim          int    // size of the vectors
	WS           int    // context window (unused by classifiers)
	Epoch        int    // training epochs
	MinCount     int    // minimal number of word occurrences
	Neg          int    // negatives sampled
	WordNgrams   int    // longest word n-gram
	Loss         string // "hs", "ns", "softmax" or "ova"
	Bucket       int    // number of n-gram buckets
	Minn         int    // shortest character n-gram
	Maxn         int    // longest character n-gram
	LrUpdateRate int
	T            float64 // sampling threshold

	Quantized       bool // input matrix is product-quantized (.ftz)
	QuantizedOutput bool // the qout flag of the file
	Pruned          bool // n-gram rows were pruned

	Words  int   // words in the vocabulary
	Labels int   // labels
	Tokens int64 // tokens seen in training
}

var lossNames = map[int32]string{lossHS: "hs", lossNS: "ns", lossSoftmax: "softmax", lossOVA: "ova"}

// Info returns the description of the model.
func (m *Model) Info() Info {
	a := &m.args
	return Info{
		Version:         int(m.version),
		Dim:             int(a.dim),
		WS:              int(a.ws),
		Epoch:           int(a.epoch),
		MinCount:        int(a.minCount),
		Neg:             int(a.neg),
		WordNgrams:      int(a.wordNgrams),
		Loss:            lossNames[a.loss],
		Bucket:          int(a.bucket),
		Minn:            int(a.minn),
		Maxn:            int(a.maxn),
		LrUpdateRate:    int(a.lrUpdateRate),
		T:               a.t,
		Quantized:       m.quant,
		QuantizedOutput: m.qout,
		Pruned:          m.dict.pruneidxSize >= 0,
		Words:           int(m.dict.nwords),
		Labels:          int(m.dict.nlabels),
		Tokens:          m.dict.ntokens,
	}
}

// Trace holds the intermediate values of one prediction. It exists to compare
// this package with fastText layer by layer; see cmd/ftdump.
type Trace struct {
	// IDs are the rows of the input matrix the text maps to: words and
	// n-grams, in fastText's order.
	IDs []int32
	// Hidden is the average of those rows. Empty when IDs is empty.
	Hidden []float32
	// Labels, Scores and Probs describe the predictions, best first: the
	// index of the label in Labels(), its log-probability and the
	// probability Predict reports.
	Labels []int32
	Scores []float32
	Probs  []float32
}

// Trace runs Predict on text and stores the intermediate values in t, whose
// slices are reused.
func (m *Model) Trace(t *Trace, text string, k int, threshold float32) error {
	t.IDs, t.Hidden = t.IDs[:0], t.Hidden[:0]
	t.Labels, t.Scores, t.Probs = t.Labels[:0], t.Scores[:0], t.Probs[:0]
	if strings.IndexByte(text, '\n') >= 0 {
		return ErrNewline
	}
	s := m.pool.Get().(*scratch)
	defer m.pool.Put(s)
	err := m.run(s, text, k, threshold)
	t.IDs = append(t.IDs, s.ids...)
	if err != nil {
		return err
	}
	if len(s.ids) > 0 {
		t.Hidden = append(t.Hidden, s.hidden...)
	}
	for _, p := range s.heap {
		t.Labels = append(t.Labels, p.label)
		t.Scores = append(t.Scores, p.score)
		t.Probs = append(t.Probs, expf(p.score))
	}
	return nil
}

// writeInfo writes the dictionary and digests of the dense matrices in the
// format of the reference tool (accuracy/refdump --info), without the header
// lines. The tests use it to check the model reader.
func (m *Model) writeInfo(out io.Writer) error {
	w := bufio.NewWriter(out)
	d := &m.dict
	for i := int32(0); i < d.nwords; i++ {
		fmt.Fprintf(w, "word\t%d\t%d\t%s\n", i, d.counts[i], d.entries[i])
	}
	for i := int32(0); i < d.nlabels; i++ {
		fmt.Fprintf(w, "label\t%d\t%d\t%s\n", i, d.counts[d.nwords+i], m.labels[i])
	}
	if !m.quant {
		fmt.Fprintf(w, "matrix\tinput\t%d\t%d\t%016x\n", m.in.rows, m.in.cols, fnv64Floats(m.in.data))
	}
	if !m.quant || !m.qout {
		data := m.out.data
		if m.outT != nil {
			// Undo the transposition.
			dim := int(m.args.dim)
			data = make([]float32, m.outRows*dim)
			for i := 0; i < m.outRows; i++ {
				for j := 0; j < dim; j++ {
					data[i*dim+j] = m.outT[j*m.outPad+i]
				}
			}
		}
		fmt.Fprintf(w, "matrix\toutput\t%d\t%d\t%016x\n", m.outRows, m.args.dim, fnv64Floats(data))
	}
	return w.Flush()
}

// fnv64Floats is 64-bit FNV-1a over the bytes of f as stored in a model file.
func fnv64Floats(f []float32) uint64 {
	h := uint64(14695981039346656037)
	if len(f) == 0 {
		return h
	}
	if hostLittleEndian {
		for _, b := range unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(f))), len(f)*4) {
			h = (h ^ uint64(b)) * 1099511628211
		}
		return h
	}
	for _, v := range f {
		u := math.Float32bits(v)
		for s := 0; s < 32; s += 8 {
			h = (h ^ uint64(byte(u>>s))) * 1099511628211
		}
	}
	return h
}
