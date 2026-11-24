package fasttext

import "math"

// sigmoidTable is fastText's t_sigmoid_: the logistic function at 513 points
// in [-8, 8]. One-vs-all and negative-sampling models read their
// probabilities from it.
var sigmoidTable = func() (t [513]float32) {
	for i := range t {
		x := float32(i*16)/512 - 8
		t[i] = float32(1.0 / (1.0 + float64(expf(-x))))
	}
	return
}()

// sigmoid is fastText's Loss::sigmoid.
func sigmoid(x float32) float32 {
	if x < -8 {
		return 0
	}
	if x > 8 {
		return 1
	}
	return sigmoidTable[int64((x+8)*512/8/2)]
}

// predict is Model::predict after the hidden vector: it leaves the best k
// labels at or above threshold in s.heap, best first.
func (m *Model) predict(s *scratch, k int, threshold float32) error {
	if m.args.loss == lossHS {
		w := hsWalk{m: m, s: s, k: k, logThreshold: stdLog(threshold)}
		w.dfs(2*m.osz-2, 0)
		if w.nan {
			s.heap = s.heap[:0]
			return ErrNaN
		}
		sortHeap(s.heap)
		return nil
	}

	out := s.output[:m.outRows]
	if err := m.dots(s, out); err != nil {
		return err
	}
	if m.args.loss == lossSoftmax {
		// SoftmaxLoss::computeOutput. The exponential is C's exp on a
		// double, rounded to float; the sum and the division are float.
		max := out[0]
		for _, v := range out {
			if !(v < max) {
				max = v
			}
		}
		// Each value is float32(exp(double(v - max))), as in fastText. The
		// three steps run as three loops because on amd64 a conversion
		// between float32 and float64 waits for the previous content of
		// its target register. In a single loop that made every exp64
		// call wait for the one before it, which cost about 12 ns per
		// label on a Xeon 8581C.
		d := s.exps[:len(out)]
		for i, v := range out {
			d[i] = float64(v - max)
		}
		for i, x := range d {
			d[i] = exp64(x)
		}
		var z float32
		for i, x := range d {
			e := float32(x)
			out[i] = e
			z += e
		}
		for i := range out {
			out[i] /= z
		}
	} else {
		for i, v := range out {
			out[i] = sigmoid(v)
		}
	}
	s.findKBest(out, k, threshold)
	sortHeap(s.heap)
	return nil
}

// dots fills out with the dot product of the hidden vector and every output
// row.
//
// fastText computes one dot product after another, each a chain of additions
// in which every step waits for the one before. Both dense forms below
// advance many of those chains together. Every label still receives the same
// products in the same order, so the results are the same bit for bit.
func (m *Model) dots(s *scratch, out []float32) error {
	h := s.hidden
	switch {
	case m.quant && m.qout:
		m.qoutM.dotRows(out, h)
	case m.outT != nil:
		// Transposed matrix: one hidden value at a time is multiplied
		// into the sums of all labels, with vector instructions.
		acc := s.output[:m.outPad]
		clear(acc)
		for j, hj := range h {
			vec.axpy(acc, m.outT[j*m.outPad:(j+1)*m.outPad], hj)
		}
	default:
		dotRows(out, m.out.data, h)
	}
	return checkNaN(out)
}

// checkNaN reports a score that is not a number. fastText throws for a dense
// output matrix. For a quantized one it does not check and goes on to index
// a table with the NaN; this package returns the error in both cases.
func checkNaN(out []float32) error {
	for _, v := range out {
		if v != v {
			return ErrNaN
		}
	}
	return nil
}

// findKBest is Loss::findKBest over the probabilities in output.
func (s *scratch) findKBest(output []float32, k int, threshold float32) {
	heap := s.heap[:0]
	// skip is the largest probability already turned down for being worse
	// than the worst candidate kept. The worst candidate only gets better,
	// and stdLog never decreases as its argument grows from zero (checked
	// for every float in the tests; probabilities are never negative), so
	// anything at or below skip is turned down without computing its
	// logarithm. fastText computes the logarithm every time; the outcome is
	// the same.
	skip := float32(math.Inf(-1))
	for i, p := range output {
		if p < threshold {
			continue
		}
		full := len(heap) == k
		if full && p <= skip {
			continue
		}
		score := stdLog(p)
		if full && score < heap[0].score {
			skip = p
			continue
		}
		heap = append(heap, scored{score, int32(i)})
		pushHeap(heap)
		if len(heap) > k {
			popHeap(heap)
			heap = heap[:len(heap)-1]
		}
	}
	s.heap = heap
}

// hsWalk is one run of HierarchicalSoftmaxLoss::dfs.
type hsWalk struct {
	m            *Model
	s            *scratch
	k            int
	logThreshold float32
	nan          bool
}

func (w *hsWalk) dfs(node int32, score float32) {
	if score < w.logThreshold {
		return
	}
	heap := w.s.heap
	if len(heap) == w.k && score < heap[0].score {
		return
	}
	m := w.m
	if m.left[node] == -1 && m.right[node] == -1 {
		heap = append(heap, scored{score, node})
		pushHeap(heap)
		if len(heap) > w.k {
			popHeap(heap)
			heap = heap[:len(heap)-1]
		}
		w.s.heap = heap
		return
	}

	var f float32
	row := int(node - m.osz)
	if m.quant && m.qout {
		f = m.qoutM.dotRow(w.s.hidden, row)
	} else {
		f = dotRow(m.out.row(row), w.s.hidden)
	}
	if f != f {
		w.nan = true
		return
	}
	// f = 1. / (1 + std::exp(-f)): a float exponential and addition, a
	// double division.
	f = float32(1.0 / float64(1+expf(-f)))

	w.dfs(m.left[node], score+stdLog(float32(1.0-float64(f))))
	if w.nan {
		return
	}
	w.dfs(m.right[node], score+stdLog(f))
}
