package fasttext

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
)

const (
	fileMagic   = 793712314
	fileVersion = 12

	modelSupervised = 3

	lossHS      = 1
	lossNS      = 2
	lossSoftmax = 3
	lossOVA     = 4
)

// args is fastText's Args as stored in a model file.
type args struct {
	dim, ws, epoch, minCount, neg, wordNgrams int32
	loss, model                               int32
	bucket, minn, maxn, lrUpdateRate          int32
	t                                         float64
}

// Model is a loaded fastText classifier. It is read-only after loading and
// safe for use by any number of goroutines.
type Model struct {
	version int32
	args    args
	dict    dictionary
	labels  []string

	// Input matrix: one of the two.
	quant bool
	in    denseMatrix
	qin   quantMatrix
	// copyFirst repeats a detail of fastText on x86: for these vector sizes
	// the hidden vector starts as a copy of the first row instead of zero
	// plus the first row. The two differ only for a row that holds -0.
	copyFirst bool

	// Output matrix: out if dense, qoutM if quantized. When a vector kernel
	// is in use and the model scores every label (softmax, one-vs-all,
	// negative sampling), the dense matrix is kept transposed in outT
	// instead: outT[j*outPad+i] is weight j of label i, with the label count
	// padded to outPad, a multiple of axpyBlock.
	qout    bool
	outRows int
	out     denseMatrix
	outT    []float32
	outPad  int
	qoutM   quantMatrix

	// Hierarchical softmax tree: children of each node, -1 for a leaf.
	osz         int32
	left, right []int32

	pool sync.Pool
}

// scratch holds the working memory of one prediction.
type scratch struct {
	ids    []int32
	hashes []int32
	word   []byte
	hidden []float32
	output []float32
	exps   []float64 // softmax only: the exponentials before rounding
	heap   []scored
	touch  float32 // keeps the reads in computeHidden alive
}

// Errors returned by Predict.
var (
	// ErrNewline is returned when the text holds a newline. fastText
	// predicts one line at a time; replace newlines with spaces to classify
	// a text of several lines as one.
	ErrNewline = errors.New("fasttext: text contains a newline")
	// ErrNaN is returned when a score is not a number, which means the
	// model holds NaN or infinite weights. fastText throws "Encountered NaN".
	ErrNaN = errors.New("fasttext: encountered NaN")
	// ErrK is returned for k of 0 or below -1.
	ErrK = errors.New("fasttext: k needs to be 1 or higher, or -1 for all labels")
)

// Open loads the model file at path, a .bin or .ftz file written by fastText.
func Open(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	size := int64(-1)
	if st, err := f.Stat(); err == nil && st.Mode().IsRegular() {
		size = st.Size()
	}
	m, err := load(f, size)
	if err != nil {
		return nil, fmt.Errorf("%w (%s)", err, path)
	}
	return m, nil
}

// Load reads a model from r. The whole model is held in memory.
//
// Sizes found in the file are checked against the length of the input when r
// can tell it (a file, a bytes.Reader, anything with Len or Seek), so a
// truncated or damaged file gives an error instead of a huge allocation.
func Load(r io.Reader) (*Model, error) {
	size := int64(-1)
	switch l := r.(type) {
	case interface{ Len() int }:
		size = int64(l.Len())
	case io.Seeker:
		if cur, err := l.Seek(0, io.SeekCurrent); err == nil {
			if end, err := l.Seek(0, io.SeekEnd); err == nil {
				if _, err := l.Seek(cur, io.SeekStart); err != nil {
					return nil, err
				}
				size = end - cur
			}
		}
	}
	return load(r, size)
}

func load(src io.Reader, size int64) (*Model, error) {
	r := newReader(src, size)
	if magic := r.int32(); r.err != nil || magic != fileMagic {
		if r.err != nil {
			return nil, r.err
		}
		return nil, errNotModel
	}
	m := &Model{}
	m.version = r.int32()
	if r.err == nil && m.version > fileVersion {
		return nil, fmt.Errorf("fasttext: model file version %d is newer than %d", m.version, fileVersion)
	}

	a := &m.args
	a.dim = r.int32()
	a.ws = r.int32()
	a.epoch = r.int32()
	a.minCount = r.int32()
	a.neg = r.int32()
	a.wordNgrams = r.int32()
	a.loss = r.int32()
	a.model = r.int32()
	a.bucket = r.int32()
	a.minn = r.int32()
	a.maxn = r.int32()
	a.lrUpdateRate = r.int32()
	a.t = r.float64()
	if r.err != nil {
		return nil, r.err
	}
	if m.version == 11 && a.model == modelSupervised {
		// Old supervised models do not use character n-grams.
		a.maxn = 0
	}
	if a.model != modelSupervised {
		return nil, errors.New("fasttext: not a supervised model; only classifiers can predict")
	}
	if a.dim < 1 {
		return nil, fmt.Errorf("fasttext: invalid dimension %d", a.dim)
	}
	dim := int(a.dim)

	if err := m.dict.load(r, a); err != nil {
		return nil, err
	}
	d := &m.dict
	m.labels = d.entries[d.nwords:]
	if d.nlabels < 1 {
		return nil, errors.New("fasttext: model has no labels")
	}

	// Input matrix.
	m.quant = r.bool()
	var inRows, inCols int
	if m.quant {
		if err := m.qin.load(r); err != nil {
			return nil, err
		}
		inRows, inCols = m.qin.rows, m.qin.cols
	} else {
		if err := m.in.load(r); err != nil {
			return nil, err
		}
		inRows, inCols = m.in.rows, m.in.cols
		if d.pruneidxSize >= 0 {
			return nil, errors.New("fasttext: invalid model file: pruned dictionary with a dense input matrix")
		}
		switch dim {
		case 16, 32, 64, 256, 512:
			m.copyFirst = true
		}
	}
	if r.err != nil {
		return nil, r.err
	}
	if inCols != dim {
		return nil, fmt.Errorf("fasttext: input matrix has %d columns, want %d", inCols, dim)
	}
	if need := m.maxInputRow() + 1; int64(inRows) < need {
		return nil, fmt.Errorf("fasttext: input matrix has %d rows, the dictionary needs %d", inRows, need)
	}

	// Output matrix.
	m.qout = r.bool()
	var outCols int
	if m.quant && m.qout {
		if err := m.qoutM.load(r); err != nil {
			return nil, err
		}
		m.outRows, outCols = m.qoutM.rows, m.qoutM.cols
	} else {
		if err := m.out.load(r); err != nil {
			return nil, err
		}
		m.outRows, outCols = m.out.rows, m.out.cols
	}
	if r.err != nil {
		return nil, r.err
	}
	if outCols != dim {
		return nil, fmt.Errorf("fasttext: output matrix has %d columns, want %d", outCols, dim)
	}

	nlabels := int(d.nlabels)
	switch a.loss {
	case lossHS:
		if m.outRows < nlabels-1 {
			return nil, fmt.Errorf("fasttext: output matrix has %d rows, the label tree needs %d", m.outRows, nlabels-1)
		}
		if err := m.buildTree(d.counts[d.nwords:]); err != nil {
			return nil, err
		}
	case lossNS, lossSoftmax, lossOVA:
		if m.outRows != nlabels {
			return nil, fmt.Errorf("fasttext: output matrix has %d rows for %d labels", m.outRows, nlabels)
		}
		if !(m.quant && m.qout) && vec != nil {
			m.outPad = (m.outRows + axpyBlock - 1) / axpyBlock * axpyBlock
			m.outT = transpose(&m.out, m.outPad)
			m.out = denseMatrix{}
		}
	default:
		return nil, fmt.Errorf("fasttext: unknown loss %d", a.loss)
	}

	m.pool.New = func() any {
		s := &scratch{
			hidden: make([]float32, dim),
			output: make([]float32, max(m.outRows, m.outPad)),
		}
		if m.args.loss == lossSoftmax {
			s.exps = make([]float64, m.outRows)
		}
		return s
	}
	return m, nil
}

// maxInputRow returns the largest input row the dictionary can produce.
func (m *Model) maxInputRow() int64 {
	d := &m.dict
	max := int64(d.nwords) - 1
	ngrams := d.maxn != 0 || d.wordNgrams > 1
	switch {
	case d.pruneidxSize > 0:
		for i, k := range d.prune.keys {
			if k >= 0 && int64(d.nwords)+int64(d.prune.vals[i]) > max {
				max = int64(d.nwords) + int64(d.prune.vals[i])
			}
		}
	case d.pruneidxSize < 0 && ngrams:
		max = int64(d.nwords) + int64(d.bucket) - 1
	}
	return max
}

// transpose returns m with rows and columns exchanged and every row of the
// result padded with zeros to pad values.
func transpose(m *denseMatrix, pad int) []float32 {
	t := make([]float32, m.cols*pad)
	for i := 0; i < m.rows; i++ {
		row := m.row(i)
		for j, v := range row {
			t[j*pad+i] = v
		}
	}
	return t
}

// buildTree is HierarchicalSoftmaxLoss::buildTree: the Huffman tree over the
// label counts. Leaves are the labels; node osz+i uses output row i.
func (m *Model) buildTree(counts []int64) error {
	osz := int32(len(counts))
	m.osz = osz
	n := 2*osz - 1
	m.left = make([]int32, n)
	m.right = make([]int32, n)
	count := make([]int64, n)
	for i := range m.left {
		m.left[i], m.right[i] = -1, -1
		count[i] = 1e15
	}
	copy(count, counts)
	leaf := osz - 1
	node := osz
	for i := osz; i < n; i++ {
		var mini [2]int32
		for j := range mini {
			if leaf >= 0 && count[leaf] < count[node] {
				mini[j] = leaf
				leaf--
			} else {
				if node >= i {
					// Only possible with label counts no training run
					// produces; fastText would read past the tree.
					return errors.New("fasttext: invalid label counts")
				}
				mini[j] = node
				node++
			}
		}
		m.left[i] = mini[0]
		m.right[i] = mini[1]
		count[i] = count[mini[0]] + count[mini[1]]
	}
	return nil
}

// Prediction is one label with its probability.
type Prediction struct {
	// Label is the label as stored in the model, including its prefix:
	// "__label__en".
	Label string
	// Prob is the probability fastText reports. For softmax models it can
	// exceed 1 by about 1e-5.
	Prob float32
}

// Predict returns the k most likely labels of text, best first, leaving out
// labels with a probability below threshold. A k of -1 means all labels.
//
// The result is what fastText's Python predict(text, k, threshold) returns:
// the text is read as one line. No result at all (nil, nil) means the text
// held nothing the model knows; with the usual models this cannot happen,
// because the end of the line counts as a token.
func (m *Model) Predict(text string, k int, threshold float32) ([]Prediction, error) {
	return m.AppendPredict(nil, text, k, threshold)
}

// AppendPredict is Predict appending to dst. With a dst of enough capacity it
// does not allocate.
func (m *Model) AppendPredict(dst []Prediction, text string, k int, threshold float32) ([]Prediction, error) {
	if strings.IndexByte(text, '\n') >= 0 {
		return dst, ErrNewline
	}
	s := m.pool.Get().(*scratch)
	err := m.run(s, text, k, threshold)
	if err == nil {
		for _, p := range s.heap {
			dst = append(dst, Prediction{Label: m.labels[p.label], Prob: expf(p.score)})
		}
	}
	m.pool.Put(s)
	return dst, err
}

// run is FastText::predictLine up to the heap: it leaves the candidates in
// s.heap, best first.
func (m *Model) run(s *scratch, text string, k int, threshold float32) error {
	s.heap = s.heap[:0]
	m.dict.line(text, s)
	if len(s.ids) == 0 {
		return nil
	}
	if k == -1 {
		k = m.outRows
	} else if k <= 0 {
		return ErrK
	}
	m.computeHidden(s)
	return m.predict(s, k, threshold)
}

// computeHidden is Model::computeHidden: the average of the input rows.
func (m *Model) computeHidden(s *scratch) {
	h := s.hidden
	ids := s.ids
	switch {
	case m.quant:
		clear(h)
		for _, id := range ids {
			m.qin.addRow(h, int(id))
		}
	default:
		add, add4 := addTo, addTo4
		if vec != nil {
			add, add4 = vec.add, vec.add4
		}
		// The rows of a large input matrix are rarely in the processor's
		// cache, and the additions below reach them one after another.
		// Reading one value of each row first costs almost nothing and
		// lets the processor fetch many rows at the same time.
		var touch float32
		data, dim := m.in.data, m.in.cols
		for _, id := range ids {
			touch += data[int(id)*dim]
		}
		s.touch = touch
		if m.copyFirst {
			copy(h, m.in.row(int(ids[0])))
			ids = ids[1:]
		} else {
			clear(h)
		}
		// Four rows at a time. Each element still receives the rows one
		// by one in order, so the sum is the same.
		for ; len(ids) >= 4; ids = ids[4:] {
			add4(h, m.in.row(int(ids[0])), m.in.row(int(ids[1])), m.in.row(int(ids[2])), m.in.row(int(ids[3])))
		}
		for _, id := range ids {
			add(h, m.in.row(int(id)))
		}
	}
	a := float32(1.0 / float64(len(s.ids)))
	for j := range h {
		h[j] *= a
	}
}

// Labels returns the labels of the model in the order of the model file. The
// slice must not be modified.
func (m *Model) Labels() []string { return m.labels }

// Dim returns the size of the model's vectors.
func (m *Model) Dim() int { return int(m.args.dim) }
