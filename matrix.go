package fasttext

import (
	"fmt"
)

// All arithmetic in this file is single precision and written so that every
// operation rounds on its own, like the reference build of fastText
// (-ffp-contract=off). A product that feeds an addition is wrapped in
// float32(...): the explicit conversion forbids the compiler from fusing the
// two into one multiply-add, which would change the last bit.

// denseMatrix is fastText's DenseMatrix: rows × cols floats, row after row.
type denseMatrix struct {
	rows, cols int
	data       []float32
}

func (m *denseMatrix) load(r *reader) error {
	rows := r.int64()
	cols := r.int64()
	if r.err != nil {
		return r.err
	}
	n, err := r.elements(rows, cols, 4)
	if err != nil {
		return err
	}
	m.rows, m.cols = int(rows), int(cols)
	m.data = r.float32s(n)
	return r.err
}

func (m *denseMatrix) row(i int) []float32 {
	return m.data[i*m.cols : (i+1)*m.cols : (i+1)*m.cols]
}

// productQuantizer is fastText's ProductQuantizer: a vector is cut into
// nsubq pieces of dsub values (the last piece has lastdsub), and each piece
// is replaced by one of 256 centroids.
type productQuantizer struct {
	dim, nsubq, dsub, lastdsub int
	centroids                  []float32
}

const ksub = 256

func (q *productQuantizer) load(r *reader) error {
	dim := r.int32()
	nsubq := r.int32()
	dsub := r.int32()
	lastdsub := r.int32()
	if r.err != nil {
		return r.err
	}
	if dim < 1 || nsubq < 1 || dsub < 1 || lastdsub < 1 ||
		(int64(nsubq)-1)*int64(dsub)+int64(lastdsub) != int64(dim) {
		return fmt.Errorf("fasttext: invalid product quantizer (dim %d, nsubq %d, dsub %d, lastdsub %d)", dim, nsubq, dsub, lastdsub)
	}
	n, err := r.elements(int64(dim), ksub, 4)
	if err != nil {
		return err
	}
	q.dim, q.nsubq, q.dsub, q.lastdsub = int(dim), int(nsubq), int(dsub), int(lastdsub)
	q.centroids = r.float32s(n)
	return r.err
}

// centroid is ProductQuantizer::get_centroids: the centroid with number code
// of piece m.
func (q *productQuantizer) centroid(m int, code uint8) []float32 {
	if m == q.nsubq-1 {
		off := m*ksub*q.dsub + int(code)*q.lastdsub
		return q.centroids[off : off+q.lastdsub]
	}
	off := (m*ksub + int(code)) * q.dsub
	return q.centroids[off : off+q.dsub]
}

// quantMatrix is fastText's QuantMatrix: a matrix whose rows are stored as
// product-quantizer codes, optionally with the row norm quantized separately.
type quantMatrix struct {
	rows, cols int
	qnorm      bool
	codes      []uint8
	pq         productQuantizer
	normCodes  []uint8
	norms      [ksub]float32 // norm for each norm code
}

func (m *quantMatrix) load(r *reader) error {
	m.qnorm = r.bool()
	rows := r.int64()
	cols := r.int64()
	codesize := r.int32()
	if r.err != nil {
		return r.err
	}
	if _, err := r.elements(rows, cols, 0); err != nil {
		return err
	}
	if codesize < 0 || int64(codesize) > r.remaining() {
		return fmt.Errorf("fasttext: invalid code size %d", codesize)
	}
	m.rows, m.cols = int(rows), int(cols)
	m.codes = r.bytes(int(codesize))
	if err := m.pq.load(r); err != nil {
		return err
	}
	if m.pq.dim != m.cols || int64(m.rows)*int64(m.pq.nsubq) != int64(codesize) {
		return fmt.Errorf("fasttext: quantized matrix %d×%d does not match its codes (dim %d, %d codes per row, %d codes)",
			rows, cols, m.pq.dim, m.pq.nsubq, codesize)
	}
	if m.qnorm {
		if rows > r.remaining() {
			return fmt.Errorf("fasttext: invalid row count %d", rows)
		}
		m.normCodes = r.bytes(m.rows)
		var npq productQuantizer
		if err := npq.load(r); err != nil {
			return err
		}
		for c := 0; c < ksub; c++ {
			m.norms[c] = npq.centroid(0, uint8(c))[0]
		}
	}
	return r.err
}

func (m *quantMatrix) norm(row int) float32 {
	if m.qnorm {
		return m.norms[m.normCodes[row]]
	}
	return 1
}

// addRow is QuantMatrix::addRowToVector: x += row.
func (m *quantMatrix) addRow(x []float32, row int) {
	q := &m.pq
	alpha := m.norm(row)
	code := m.codes[row*q.nsubq : (row+1)*q.nsubq]
	for s, c := range code {
		cen := q.centroid(s, c)
		xs := x[s*q.dsub:]
		xs = xs[:len(cen)]
		for n, v := range cen {
			xs[n] += float32(alpha * v)
		}
	}
}

// dotRow is QuantMatrix::dotRow: the dot product of x with row.
func (m *quantMatrix) dotRow(x []float32, row int) float32 {
	q := &m.pq
	alpha := m.norm(row)
	code := m.codes[row*q.nsubq : (row+1)*q.nsubq]
	var res float32
	for s, c := range code {
		cen := q.centroid(s, c)
		xs := x[s*q.dsub:]
		xs = xs[:len(cen)]
		for n, v := range cen {
			res += float32(xs[n] * v)
		}
	}
	return float32(res * alpha)
}

// dotRows fills out[i] with dotRow(x, i), four rows at a time. As in the
// dense dotRows, every sum receives its products in the order of dotRow.
func (m *quantMatrix) dotRows(out, x []float32) {
	q := &m.pq
	ns := q.nsubq
	i := 0
	for ; i+4 <= len(out); i += 4 {
		code := m.codes[i*ns : (i+4)*ns : (i+4)*ns]
		var r0, r1, r2, r3 float32
		for s := 0; s < ns; s++ {
			c0 := q.centroid(s, code[s])
			c1 := q.centroid(s, code[ns+s])
			c2 := q.centroid(s, code[2*ns+s])
			c3 := q.centroid(s, code[3*ns+s])
			xs := x[s*q.dsub:]
			xs = xs[:len(c0)]
			c1, c2, c3 = c1[:len(xs)], c2[:len(xs)], c3[:len(xs)]
			for n, v := range xs {
				r0 += float32(v * c0[n])
				r1 += float32(v * c1[n])
				r2 += float32(v * c2[n])
				r3 += float32(v * c3[n])
			}
		}
		out[i] = float32(r0 * m.norm(i))
		out[i+1] = float32(r1 * m.norm(i+1))
		out[i+2] = float32(r2 * m.norm(i+2))
		out[i+3] = float32(r3 * m.norm(i+3))
	}
	for ; i < len(out); i++ {
		out[i] = m.dotRow(x, i)
	}
}
