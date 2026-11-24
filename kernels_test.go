package fasttext

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestKernels compares every vector kernel and the eight-row dot product
// with the plain loops, bit for bit, on random data of many lengths.
func TestKernels(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 20261120))
	random := func(n int) []float32 {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32(rng.NormFloat64())
			switch rng.IntN(40) {
			case 0:
				f[i] = 0
			case 1:
				f[i] = float32(math.Copysign(0, -1))
			case 2:
				f[i] *= 1e-30
			}
		}
		return f
	}
	same := func(a, b []float32) bool {
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				return false
			}
		}
		return true
	}

	for _, k := range kernels {
		for n := 0; n <= 300; n++ {
			dst, src := random(n), random(n+3)
			want := append([]float32(nil), dst...)
			addTo(want, src)
			k.add(dst, src)
			if !same(dst, want) {
				t.Fatalf("%s: add differs from the plain loop at length %d", k.name, n)
			}
			a, b, c, d := random(n), random(n+1), random(n), random(n+2)
			for _, r := range [][]float32{a, b, c, d} {
				addTo(want, r)
			}
			k.add4(dst, a, b, c, d)
			if !same(dst, want) {
				t.Fatalf("%s: add4 differs from four plain loops at length %d", k.name, n)
			}
		}
		for n := 0; n <= 2080; n += axpyBlock {
			y, x, a := random(n), random(n), float32(rng.NormFloat64())
			want := append([]float32(nil), y...)
			axpy(want, x, a)
			k.axpy(y, x, a)
			if !same(y, want) {
				t.Fatalf("%s: axpy differs from the plain loop at length %d", k.name, n)
			}
		}
	}

	for n := 0; n <= 70; n++ {
		dst := random(n)
		want := append([]float32(nil), dst...)
		a, b, c, d := random(n), random(n+1), random(n), random(n+2)
		for _, r := range [][]float32{a, b, c, d} {
			addTo(want, r)
		}
		addTo4(dst, a, b, c, d)
		if !same(dst, want) {
			t.Fatalf("addTo4 differs from four plain loops at length %d", n)
		}
	}

	for _, dim := range []int{1, 10, 16, 20, 96, 256} {
		for _, n := range []int{0, 1, 7, 8, 9, 176, 323} {
			w, h := random(n*dim), random(dim)
			got := make([]float32, n)
			dotRows(got, w, h)
			want := make([]float32, n)
			for i := range want {
				want[i] = dotRow(w[i*dim:(i+1)*dim], h)
			}
			if !same(got, want) {
				t.Fatalf("dotRows differs from dotRow for %d rows of %d", n, dim)
			}
		}
	}

	// Quantized rows: four at a time against one at a time.
	for _, c := range []struct{ dim, dsub, rows int }{{64, 4, 323}, {16, 2, 176}, {10, 3, 9}, {32, 2, 3}, {7, 7, 5}} {
		var m quantMatrix
		q := &m.pq
		q.dim, q.dsub = c.dim, c.dsub
		q.nsubq = c.dim / c.dsub
		q.lastdsub = c.dim % c.dsub
		if q.lastdsub == 0 {
			q.lastdsub = c.dsub
		} else {
			q.nsubq++
		}
		q.centroids = random(c.dim * ksub)
		m.rows, m.cols = c.rows, c.dim
		m.codes = make([]uint8, c.rows*q.nsubq)
		for i := range m.codes {
			m.codes[i] = uint8(rng.Uint32())
		}
		for _, qnorm := range []bool{false, true} {
			m.qnorm = qnorm
			m.normCodes = make([]uint8, c.rows)
			for i := range m.normCodes {
				m.normCodes[i] = uint8(rng.Uint32())
			}
			copy(m.norms[:], random(ksub))
			x := random(c.dim)
			got := make([]float32, c.rows)
			m.dotRows(got, x)
			want := make([]float32, c.rows)
			for i := range want {
				want[i] = m.dotRow(x, i)
			}
			if !same(got, want) {
				t.Fatalf("quantized dotRows differs from dotRow for %+v", c)
			}
		}
	}
	t.Logf("kernels: %v, in use: %s", KernelNames(), Kernel())
}
